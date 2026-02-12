import numpy as np
import os 
from datetime import datetime
from PIL import Image
import shutil
import json
import scipy.ndimage 
import gc
import tempfile
from utils import save_state, load_state, clear_log_file, log_error
from ...helpers import (
    generate_check_file_image_data,
    retrieve_existing_warnings_image_data,
    load_patient_series_seg_data,
    load_patient_mappings,
    load_series_mappings
)

LOG_FILENAME = "2D_segmentation_preprocessing_error.log"

class SegmentImagePreprocessor:

    def __init__(self, protocol, local_config, mapping_file, num_image_patients, num_slices_group, total_series, total_slices, series_group_name):
        self.protocol = protocol
        self.local_config = local_config
        self.mapping_file = mapping_file
        self.num_image_patients = num_image_patients
        self.num_slices_group = num_slices_group
        self.total_series = total_series
        self.total_slices = total_slices
        self.series_group_name = series_group_name
        self.study_path = os.getenv("ROOT_NAME")
        self.host_study_path = os.getenv("HOST_ROOT_DIR")
        self.input_dir = os.getenv("INPUT_DIR")
        self.images_dir = os.path.join(self.input_dir, "IMAGES")
        self.host_input_dir = os.getenv("HOST_BASE_DATA_DIR")
        self.state_file = os.path.join(self.input_dir, "image_validation_state.json")
        self.series_progress_file = os.path.join(self.input_dir, "validation_progress_by_series.json")
        self.output_directory_report = os.path.join(self.input_dir, "Reports", f"Image Data Reports {self.series_group_name}")
        self.host_output_directory_report = os.path.join(self.host_input_dir, "Reports", f"Image Data Reports {self.series_group_name}")
        self.output_directory_checkfile = os.path.join(self.study_path, "CHECKFILE")
        self.output_directory_data = os.path.join(self.study_path, "IMAGES")
        self.host_output_directory_data = os.path.join(self.host_study_path, "IMAGES")
        os.makedirs(self.output_directory_report, exist_ok=True)  
        os.makedirs(self.output_directory_checkfile, exist_ok=True)
        os.makedirs(self.output_directory_data, exist_ok=True)


    def map_segmentation_values(self, segmentation_codes): 
        """
        Load each segmentation file, map unique pixel values to segmentation codes,
        and save the mapped array as .npy in the output segmentation folder.
        """
        series_group = self.series_group_name
        patient_id_map = load_patient_mappings(self.local_config, self.input_dir)
        series_id_map = load_series_mappings(self.images_dir, series_group)
    
        series_group_mapping = self.local_config["Local config"]["radiological data"]["series_mapping"]
        series_group_subfolder = series_group_mapping.get(series_group)
    
        patient_folders = [f for f in os.listdir(self.images_dir)
                           if os.path.isdir(os.path.join(self.images_dir, f)) and f != "Reports"]
    
        for orig_patient_id in patient_folders:
            patient_path = os.path.join(self.images_dir, orig_patient_id)
            series_folder = os.path.normpath(series_group_subfolder).split(os.sep)[-1].strip()
            series_path = os.path.join(patient_path, series_folder)
            full_series_key = os.path.join(orig_patient_id, series_folder)
            # If mappings use Windows-style separators, normalize only for lookup
            if "\\" in list(series_id_map.keys())[0]:  # Detect if mapping keys use '\'
                full_series_key = full_series_key.replace("/", "\\")

            new_series_name = series_id_map.get(full_series_key)
            new_patient_id = patient_id_map.get(orig_patient_id)

            if new_series_name and "_" in new_series_name:
                parts = new_series_name.split("_")
                new_series_name2 = f"{parts[1]}_{parts[2]}_{parts[0]}"
                seg_folder_name = f"seg_{parts[1]}_{parts[2]}_{parts[0]}"
            else:
                continue 
                
            # Locate segmentation folder (first subfolder)
            subfolders = [f for f in os.listdir(series_path) if os.path.isdir(os.path.join(series_path, f))]
            seg_folder_path = os.path.join(series_path, subfolders[0])

            # Add this assert to ensure only one file exists in the folder
            assert len(os.listdir(seg_folder_path)) == 1, \
                f"Expected only one segmentation file in {seg_folder_path}, but found multiple."
    
            filename = os.listdir(seg_folder_path)[0]
            file_path = os.path.join(seg_folder_path, filename)

            # Load image (assuming PNG/JPG/TIFF)
            try:
                img = Image.open(file_path).convert("L")
                seg_array = np.array(img)
            except Exception as e:
                raise ValueError(
                    f"Could not open: {e}"
                )

            unique_values = np.unique(seg_array)
            sorted_values = np.sort(unique_values)

            if len(sorted_values) > len(segmentation_codes):
                raise ValueError(
                    f"Error in {filename}: Found more unique values in the mask ({sorted_values.tolist()}) "
                    f"than provided segmentation codes ({segmentation_codes})."
                )

            # Create mapping
            value_mapping = {old: new for old, new in zip(sorted_values, segmentation_codes)}

            # Map pixel values
            mapped_array = np.vectorize(value_mapping.get)(seg_array)
            mapped_array = mapped_array.astype(np.int16)

            # Prepare output directory, naming convention similar to check_and_reorient_niftiseg
            patient_out_path = os.path.join(self.output_directory_data, new_patient_id)
            series_out_path = os.path.join(patient_out_path, new_series_name)
            seg_out_path = os.path.join(series_out_path, seg_folder_name)
            os.makedirs(series_out_path, exist_ok=True)

            # Save .npy with similar name but .npy extension
            npy_filename = f"seg_{new_series_name2}.npy"
            
            save_path = os.path.join(seg_out_path, npy_filename)
            mapped_array = np.expand_dims(mapped_array, axis=0)  # adds a new axis at position 0
            np.save(save_path, mapped_array)
            print(f"Saved mapped segmentation: {save_path}")


    def single_to_multi_label_conversion_2D(self, multiple_seg_file_flag): 
        """
        Convert multiple single-label 2D segmentation files into a single multi-label .npy mask
        and save the result in the output directory.
        """
        if multiple_seg_file_flag is None:
            raise ValueError("Missing required 'multiple_seg_file_flag'. Cannot proceed with multi-label conversion.")
        
        series_group = self.series_group_name
        series_group_mapping = self.local_config["Local config"]["radiological data"]["series_mapping"]
        series_group_subfolder = series_group_mapping.get(series_group) #input name of the series
        seg_group_config = self.protocol.get(series_group, {}).get("segmentation", {}) #original self.protocol["segmentation"]
        
        seg_input_format = seg_group_config["segmentation_input_format"]["selected"]
        
        # Case 1: Input already multi-label mask
        if seg_input_format == "multi-mask":
            segmentation_codes = sorted(seg_group_config["segmentation_code"])

            # Map pixel values to segmentation codes
            self.map_segmentation_values(segmentation_codes)
            
            single_to_multilabel_message = "Multi-label conversion skipped: input format is already 'multi-mask'. Pixel values were mapped to protocol segmentation codes."
            return single_to_multilabel_message

        # Case 2: Single-mask but only 1 label (2 segments including background)
        if seg_input_format == "single-mask" and len(seg_group_config["segments_labels"]) == 2:
            segmentation_codes = sorted(seg_group_config["segmentation_code"])
            
            # Map pixel values to segmentation codes
            self.map_segmentation_values(segmentation_codes)
                    
            single_to_multilabel_message = "Multi-label conversion skipped: single-mask with only 1 label. Pixel values were mapped to protocol segmentation code."
            return single_to_multilabel_message
    
        if seg_input_format == "single-mask" and multiple_seg_file_flag:
            
            patient_id_map = load_patient_mappings(self.local_config, self.input_dir)
            series_id_map = load_series_mappings(self.images_dir, series_group)
            
            seg_file_format = seg_group_config["segmentation file format"]["selected"]

            # Get expected labels and their assigned segmentation codes
            segment_labels = [label.lower().replace("_", " ").replace("-", " ").replace("  ", " ") 
                              for label in seg_group_config["segments_labels"][1:]]  # Skip "background"
            segment_codes = seg_group_config["segmentation_code"][1:]  # Skip background code (assumed 0)

            # Create the label_map dictionary
            label_map = {f"{label}{seg_file_format}".lower(): code for label, code in zip(segment_labels, segment_codes)}

            patient_folders = [f for f in os.listdir(self.images_dir)
                       if os.path.isdir(os.path.join(self.images_dir, f)) and f != "Reports"]

            for orig_patient_id in patient_folders:
                patient_path = os.path.join(self.images_dir, orig_patient_id)
                series_folder = os.path.normpath(series_group_subfolder).split(os.sep)[-1].strip()
                series_path = os.path.join(patient_path, series_folder)
                full_series_key = os.path.join(orig_patient_id, series_folder)
                # If mappings use Windows-style separators, normalize only for lookup
                if "\\" in list(series_id_map.keys())[0]:  # Detect if mapping keys use '\'
                    full_series_key = full_series_key.replace("/", "\\")

                new_series_name = series_id_map.get(full_series_key)
                new_patient_id = patient_id_map.get(orig_patient_id)
                if new_series_name and "_" in new_series_name:
                    parts = new_series_name.split("_")
                    new_series_name2 = f"{parts[1]}_{parts[2]}_{parts[0]}"
                    seg_folder_name = f"seg_{parts[1]}_{parts[2]}_{parts[0]}"
                else:
                    continue 
                
                # Find segmentation folder (first subfolder)
                subfolders = [f for f in os.listdir(series_path) if os.path.isdir(os.path.join(series_path, f))]
                seg_folder_path = os.path.join(series_path, subfolders[0])

                # Initialize a dict to store masks by patient/series
                masks_for_series = []

                for filename in os.listdir(seg_folder_path):
                    normalized_filename = filename.lower().replace("_", " ").replace("-", " ").replace("  ", " ")
                    label_value = label_map.get(normalized_filename)
                    if label_value is None:
                        raise ValueError(
                            f"File {filename} does not match any expected label."
                        )
                        
                    file_path = os.path.join(seg_folder_path, filename)
                    try:
                        img = Image.open(file_path).convert("L")
                        seg_array = np.array(img)
                    except Exception as e:
                        raise ValueError(
                            f"Could not open: {e}"
                        )

                    # Create mask with label value
                    mask = (seg_array != 0).astype(np.int16) * label_value
                    masks_for_series.append(mask)

                if not masks_for_series:
                    raise ValueError(f"No valid masks found for patient {orig_patient_id} series {series_folder}.")
                                
                # Combine all masks into one multi-label mask
                multi_label_mask = np.zeros_like(masks_for_series[0])
                for mask in masks_for_series:
                    multi_label_mask = np.where(mask > 0, mask, multi_label_mask)
        
                # Prepare output directory, similar naming conventions
                patient_out_path = os.path.join(self.output_directory_data, new_patient_id)
                series_out_path = os.path.join(patient_out_path, new_series_name)
                seg_out_path = os.path.join(series_out_path, seg_folder_name)
                os.makedirs(series_out_path, exist_ok=True)
        
                # Save .npy with similar name but .npy extension
                npy_filename = f"seg_{new_series_name2}.npy"
                save_path = os.path.join(seg_out_path, npy_filename)
                
                multi_label_mask = np.expand_dims(multi_label_mask, axis=0)  # adds a new axis at position 0
                np.save(save_path, multi_label_mask)
                print(f"Saved multi-label mask for {new_patient_id}/{new_series_name} as {npy_filename}")

            single_to_multilabel_message = "Multi-label conversion successful."
            return single_to_multilabel_message
        

    def resize_seg(self):  
        """
        Loads and resizes all .npy segmentation files in the output directory to the target size defined in the protocol.
        Only overwrites the files if resizing is needed.
        
        Returns:
        str: Message indicating whether resizing was performed.
        """
        series_group = self.series_group_name
        series_group_mapping = self.local_config["Local config"]["radiological data"]["series_mapping"]
        series_group_subfolder = series_group_mapping.get(series_group) #input name of the series
        seg_group_config = self.protocol.get(series_group, {}).get("segmentation", {}) #original self.protocol["segmentation"]
        output_directory_data = self.output_directory_data

        # Retrieve the target size from the protocol
        target_size = seg_group_config.get("segments_size", [256, 256])
        output_shape = (target_size[0], target_size[1])  # Set the desired output shape
        any_resized = False   # Flag to check if any slices were resized
        final_message_resize = ""

        for patient_id, series_id, npy_path, tensor, _ in load_patient_series_seg_data(series_group, output_directory_data):
            print(f"Processing {patient_id}/{series_id} - File: {os.path.basename(npy_path)}")

            # Check if resizing is necessary
            if tensor.shape[1] == output_shape[0] and tensor.shape[2] == output_shape[1]:
                #any_resized = False
                print(f"Already correct shape: {tensor.shape}. Skipping.")
                continue

            # Mark that resizing occurred
            any_resized = True
            
            # Reshape each slice independently along the third dimension
            num_slices = tensor.shape[0]
            resized_slices = []
            
            for slice_idx in range(num_slices):
                # Extract the 2D slice
                slice_2d = tensor[slice_idx, :, :]
                
                # Calculate zoom factors for resizing the slice
                zoom_factors = (output_shape[0] / slice_2d.shape[0], output_shape[1] / slice_2d.shape[1])
                
                # Resize the 2D slice using nearest-neighbor interpolation (order=0)
                resized_slice = scipy.ndimage.zoom(slice_2d, zoom_factors, order=0)
                
                # Ensure the resized slice is of integer type
                resized_slices.append(resized_slice) 
            
            # Stack the resized slices back into a 3D array
            resized_segmentation = np.stack(resized_slices, axis=0)

            # Overwrite the file with the resized data
            np.save(npy_path, resized_segmentation)
            any_resized = True
            print(f"Resized and saved: {npy_path} -> New shape: {resized_segmentation.shape}")

            # Manually clean up memory to avoid RAM buildup
            del resized_slices
            del resized_segmentation
            gc.collect()

        # Construct the final message based on whether any slices were resized
        if any_resized:
            final_message_resize += "Resizing completed."
        else:
            final_message_resize += "Resizing not needed: segmentations slices are already of the correct size."
    
        print(final_message_resize)

        return final_message_resize
    

    def check_seg_replicates_after_processing(self): #47 
        """
        Checks for identical 3D segmentation masks within a central slice range,
        excluding fully black slices.
    
        Returns:
        - str: Success message if no replicates are found.
        """
        
        local_config = self.local_config
        output_directory_data = self.output_directory_data
        series_group = self.series_group_name
        
        patient_id_map = load_patient_mappings(local_config, self.input_dir)
        series_id_map = load_series_mappings(self.images_dir, series_group)

        # Reverse maps to retrieve original keys from standardized IDs
        reversed_patient_id_map = {v: k for k, v in patient_id_map.items()}
        reversed_series_id_map = {v: k for k, v in series_id_map.items()}
        
        phase_number = 47  
        phase_data = self.mapping_file.get("check_phase", {}).get(str(phase_number), {})
        error_descriptions = phase_data.get("errors", {})
        warning_descriptions = phase_data.get("warnings", {})
        phase_name = phase_data.get("name", f"Phase {phase_number}")  
        
        # Format check and report names dynamically
        check_name = phase_name.lower().replace(" ", "_")
        report_name = f"4.047.{check_name}_report"
        series_group = self.series_group_name
        # Retrieve existing warnings
        check_file_path = os.path.join(self.output_directory_checkfile, "image_data_check_file.json")
        existing_warnings = retrieve_existing_warnings_image_data(check_file_path)
        
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        checked_pairs = set()
        grouped_series = []
        seg_index = {}
        total_replicates = 0
        error_counts = {"E4701": None}

        temp_dir = tempfile.mkdtemp()

        try:
            # Step 1: Save all segmentations temporarily, track by label
            for patient_id, series_id, npy_path, tensor, _ in load_patient_series_seg_data(series_group, output_directory_data):
                slices = tensor
                
                orig_patient_id = reversed_patient_id_map.get(patient_id, patient_id)
                orig_series_id = reversed_series_id_map.get(series_id, series_id)
                
                # For labels used in reporting
                label = f"{orig_patient_id}/{os.path.basename(orig_series_id)}"
                temp_path = os.path.join(temp_dir, f"{label.replace('/', '_')}.npy")
                np.save(temp_path, tensor)
                seg_index[label] = temp_path

            labels = list(seg_index.keys())
    
            for i, label1 in enumerate(labels):
                vol1 = np.load(seg_index[label1])
            
                # Discard fully black slices
                non_black_slices_1 = [z for z in range(vol1.shape[0]) if not np.all(vol1[z, :, :] == 0)]
                if not non_black_slices_1:
                    continue  # Skip if fully black
            
                middle_idx_1 = non_black_slices_1[len(non_black_slices_1) // 2]
                start_1 = max(middle_idx_1 - 5, 0)
                end_1 = min(middle_idx_1 + 6, vol1.shape[0])
                cropped_vol1 = vol1[start_1:end_1, :, :]
            
                for j in range(i + 1, len(labels)):
                    label2 = labels[j]
                    pair = tuple(sorted([label1, label2]))
            
                    if pair in checked_pairs:
                        continue
                    checked_pairs.add(pair)
            
                    vol2 = np.load(seg_index[label2])
            
                    non_black_slices_2 = [z for z in range(vol2.shape[0]) if not np.all(vol2[z, :, :] == 0)]
                    if not non_black_slices_2:
                        continue  # Skip if fully black
            
                    middle_idx_2 = non_black_slices_2[len(non_black_slices_2) // 2]
                    start_2 = max(middle_idx_2 - 5, 0)
                    end_2 = min(middle_idx_2 + 6, vol2.shape[0])
                    cropped_vol2 = vol2[start_2:end_2, :, :]
            
                    if cropped_vol1.shape != cropped_vol2.shape:
                        continue  # Skip if shapes don't match
            
                    if np.array_equal(cropped_vol1, cropped_vol2):
                        # Identical: add to group
                        group_found = False
                        for group in grouped_series:
                            if label1 in group or label2 in group:
                                group.add(label1)
                                group.add(label2)
                                group_found = True
                                break
            
                        if not group_found:
                            grouped_series.append(set([label1, label2]))
    
            total_replicates = sum(len(group) - 1 for group in grouped_series)
            error_counts["E4701"] = total_replicates if total_replicates > 0 else None
        
            # Generate Check File
            generate_check_file_image_data(
                check_name=phase_name,
                phase_number=phase_number,
                series_group_name=series_group,
                num_slices_group=self.num_slices_group,
                num_patients_with_image_data=self.num_image_patients,
                num_series=self.total_series,
                num_tot_slices=self.total_slices,
                timestamp=current_datetime,
                error_counts=error_counts,
                warning_counts={},
                error_descriptions=error_descriptions,
                warning_descriptions={},
                output_dir=self.output_directory_checkfile,
                existing_warnings=existing_warnings
            )
        
            # Generate Report
            if grouped_series:
                os.makedirs(self.output_directory_report, exist_ok=True)
                report_path = os.path.join(self.output_directory_report, f"{report_name}.txt")
                host_report_path = os.path.join(self.host_output_directory_report, f"{report_name}.txt")
        
                with open(report_path, "w") as report_file:
                    report_file.write(f"Report generated on: {current_datetime}\n\n")
                    report_file.write(f"{phase_name} report:\n\n")
                    report_file.write(f"Error E4701: {error_descriptions.get('E4701', None)}\n")
                    report_file.write("- Details: The following patient/series groups have identical segmentations:\n") #adapted to 2D configuration
        
                    for i, group in enumerate(grouped_series):
                        group_list = sorted(list(group))
                        group_report = " and ".join(group_list)
                        report_file.write(f"  - {group_report}.\n")
        
                    report_file.write(f"\nTotal Errors: {total_replicates}\n")
        
                raise ValueError(f"Identical segmentations detected. "
                                 f"See the detailed report at: {host_report_path}") #!! Path del container, deve essere quello locale.OK
    
            no_equal_seg_message = "No identical segmentations detected."
        
            return no_equal_seg_message
            
        finally:
            # Clean up temp directory
            shutil.rmtree(temp_dir)


    def validate_npy_dimension_match(self):
        """
        Validates that the final .npy volumes files in series folders within each patient folder in output_directory_data
        have matching dimensions with the corresponding .npy files in segmentation folders.
        Generates a report if mismatches are found and a check file regardless.
        """
        phase_number = 48  
        phase_data = self.mapping_file.get("check_phase", {}).get(str(phase_number), {})
        error_descriptions = phase_data.get("errors", {})
        phase_name = phase_data.get("name", f"Phase {phase_number}")  
        
        # Format check and report names dynamically
        check_name = phase_name.lower().replace(" ", "_") 
        report_name = f"4.048.{check_name}_report.txt"

        series_group = self.series_group_name
        series_group_mapping = self.local_config["Local config"]["radiological data"]["series_mapping"]
        series_group_subfolder = series_group_mapping.get(series_group) #input name of the series
        
        num_errors = 0
        has_error = False
        discrepancy_info = ""  
        npy_dim_success_message = ""
        error_counts = {"E4801": None}
    
        # Retrieve existing warnings
        check_file_path = os.path.join(self.output_directory_checkfile, "image_data_check_file.json")
        existing_warnings = retrieve_existing_warnings_image_data(check_file_path)
        
        # Get the current datetime for file naming and report generation
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Iterate over patient folders in output_directory_data
        patient_folders = [f for f in os.listdir(self.output_directory_data) if os.path.isdir(os.path.join(self.output_directory_data, f))]
        
        for patient_folder in patient_folders:
            patient_folder_path = os.path.join(self.output_directory_data, patient_folder)
            
            if os.path.isdir(patient_folder_path):
                # Find the series folder that starts with series_group
                series_folder = next(
                    (folder for folder in os.listdir(patient_folder_path)
                     if folder.startswith(series_group) and os.path.isdir(os.path.join(patient_folder_path, folder))),
                    None
                )
                    
                if series_folder:
                    series_folder_path = os.path.join(patient_folder_path, series_folder)
            
                    if os.path.isdir(series_folder_path): 
                        # Locate the .npy files in both the image and segmentation subfolders
                        image_npy_files = [f for f in os.listdir(series_folder_path) if f.endswith('.npy') and f.startswith("tensor")]
                        if not image_npy_files:
                            discrepancy_info += f"  - Missing image .npy file for patient {patient_folder}, series {series_folder}.\n"
                            num_errors += 1
                            has_error = True
                            continue
        
                        # Locate the segmentation folder dynamically
                        seg_folder = None
                        for subfolder in os.listdir(series_folder_path):
                            subfolder_path = os.path.join(series_folder_path, subfolder)
                            if os.path.isdir(subfolder_path):  # Check if it's a folder
                                seg_npy_files = [f for f in os.listdir(subfolder_path) if f.endswith('.npy')]
                                if seg_npy_files:  # Found segmentation .npy files
                                    seg_folder = subfolder_path
                                    break
                        
                        if not seg_folder:
                            discrepancy_info += f"  - Missing segmentation folder for patient {patient_folder}, series {series_folder}.\n"
                            num_errors += 1
                            has_error = True
                            continue
        
                        # Load the first image and segmentation .npy files for dimension matching
                        image_npy_path = os.path.join(series_folder_path, image_npy_files[0])
                        seg_npy_path = os.path.join(seg_folder, seg_npy_files[0]) 
                        
                        try:
                            image_data = np.load(image_npy_path)
                            seg_data = np.load(seg_npy_path)
                        except Exception as e:
                            discrepancy_info += f"  - Failed to load .npy file for patient {patient_folder}, series {series_folder}. Error: {str(e)}\n"
                            num_errors += 1
                            has_error = True
                            continue
                            
                        #print("image data shape:", image_data.shape)
                        #print("seg data shape:", seg_data.shape)
                        # Check if dimensions match
                        if image_data.shape != seg_data.shape:
                            discrepancy_info += (
                                f"  - Dimension mismatch for patient {patient_folder}, series {series_folder}. "
                                f"Image dimensions: {image_data.shape}, Segmentation dimensions: {seg_data.shape}\n"
                            )
                            num_errors += 1
                            has_error = True

        error_counts = {"E4801": num_errors if num_errors > 0 else None}
        # Use the helper function to generate the report and check file
        # Update the JSON check file
        generate_check_file_image_data(
            check_name=phase_name,
            phase_number=phase_number,
            series_group_name=series_group,
            num_slices_group=self.num_slices_group,
            num_patients_with_image_data=self.num_image_patients,
            num_series=self.total_series,
            num_tot_slices=self.total_slices,
            timestamp=current_datetime,
            error_counts=error_counts,
            warning_counts={},
            error_descriptions=error_descriptions,
            warning_descriptions={},
            output_dir=self.output_directory_checkfile,
            existing_warnings=existing_warnings
        )
    
        # Return appropriate message
        if has_error:
            report_content = ""
            report_filename = os.path.join(self.output_directory_report, report_name)
            host_report_filename = os.path.join(self.host_output_directory_report, report_name)
            report_content += f"Report generated on: {current_datetime}\n\n"
            report_content += f"{phase_name} report:\n\n"
            report_content += f"Error E4801: {error_descriptions.get('E4801', None)}\n"
            report_content += f"- Details:\n"
            report_content += discrepancy_info
            report_content += f"\nTotal Errors: {num_errors}"
            with open(report_filename, "w") as report_file:
                report_file.write(report_content)
            raise Exception(
                 f"Dimension mismatch detected between image and segmentation .npy files. "
                 f"See the detailed report at: {host_report_filename}" #!! Path del container, deve essere quello locale.OK
            )
        else:
            npy_dim_success_message += "Dimension validation successful: all corresponding .npy volumes and segmentation files match in dimensions."
            return npy_dim_success_message
        

    def generate_SegmentImagePreprocess_final_report(self, single_to_multilabel_message, final_message_resize): #riaggiungo no_equal_seg_message,npy_dim_success_message 
        """
        Generate a comprehensive final report for SegmentImagePreprocessor.
    
        Parameters:
        - single_to_multilabel_message: Message about multi-label conversion.
        - final_message_resize: Message about resizing status.
        - no_equal_seg_message: Message about segmentation equality check.
        - npy_dim_success_message: Message about NPY dimensions check.
        """
        
        # Get the current datetime
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
        
        # Define the report filename and path
        report_filename = "4.SegmentImagePreprocess_final_report.txt"
        report_filepath = os.path.join(self.output_directory_report, report_filename)
        
        # Initialize report content with ordered titles and messages
        report_content = []
        
        # Add report header
        report_content.append(f"Report generated on: {formatted_datetime}\n")
        report_content.append("Segmentation preprocessing report:\n")
        
        # Single to multi-label conversion status
        report_content.append("Multi-label conversion status:")
        report_content.append("- " + single_to_multilabel_message + "\n")

        # Add resizing status
        report_content.append("Segmentation resizing status:")
        if final_message_resize:
            report_content.append("- " + final_message_resize + "\n")

        # Replicate check status
        #report_content.append("Replicate segmentations check:")
        #report_content.append("- " + no_equal_seg_message + "\n")
        
         # Export to NPY format status
        report_content.append("Segmentation tensor save status:")
        final_message_npy = f"Tensors saved in .npy format into segmentation folders inside series folders within patient folders in {self.host_output_directory_data}." #!! Path del container, deve essere quello locale.OK
        report_content.append("- " + final_message_npy + "\n")

        # NPY dimension check status
        #report_content.append("NPY dimensions check status:")
        #if npy_dim_success_message:
            #report_content.append("- " + npy_dim_success_message + "\n")
            
        # Final report summary
        report_content.append("Summary:")
        report_content.append("All checks and transformations completed.\n")
        
        # Write the report to the specified file
        with open(report_filepath, 'w') as report_file:
            report_file.write("\n".join(report_content))
        
        # Print a message indicating that the final report has been generated
        print(f"SegmentImagePreprocess final report saved to {report_filepath}")


def run_segimage_preprocessor(protocol, local_config, mapping_file, series_group_name):
    # Define the input directory path with 
    input_dir = os.getenv("INPUT_DIR")

    # Clear log at the start of validation
    clear_log_file(input_dir, LOG_FILENAME)

    # Load the input validation state to extract num_image_patients
    input_state_file = os.path.join(input_dir, "input_validation_state.json")
    image_state_file = os.path.join(input_dir, "image_validation_state.json")

    try:
        with open(input_state_file, "r") as f:
            input_state = json.load(f)
        num_image_patients = input_state.get("num_patients_with_image_data", 0)
    except Exception as e:
        raise RuntimeError(f"Failed to load input state file '{input_state_file}': {e}")
    
    try:
        with open(image_state_file, "r") as f:
            image_state = json.load(f)
        num_slices_group = image_state.get("num_slices_group", 0)
        total_series = image_state.get("total_series", 0)
        total_slices = image_state.get("total_slices", 0)
    except Exception as e:
        raise RuntimeError(f"Failed to load image state file '{image_state_file}': {e}")
    
    segimage_preprocessor = SegmentImagePreprocessor(protocol, local_config, mapping_file, num_image_patients,  num_slices_group, total_series, total_slices, series_group_name)

    series_progress_state =  load_state(segimage_preprocessor.series_progress_file)

    # Access or create the per-series progress dictionary
    series_state = series_progress_state.setdefault(series_group_name, {})
    last_phase_done = series_state.get("last_successful_phase", 0)

    if last_phase_done < 48:

        print("Running single_to_multi_label_conversion_2D function...")
        try:
            single_to_multilabel_message = segimage_preprocessor.single_to_multi_label_conversion_2D(image_state.get("multiple_segmentation_flag"))
        except Exception as e:
            log_error(input_dir, "single_to_multi_label_conversion_2D", e, LOG_FILENAME)
            print(f"An unexpected error occurred during single_to_multi_label_conversion_2D. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
            raise

        print("Running resize_seg function...")
        try:
            final_message_resize = segimage_preprocessor.resize_seg()
        except Exception as e:
            log_error(input_dir, "resize_seg", e, LOG_FILENAME)
            print(f"An unexpected error occurred during resize_seg. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
            raise

        #print("Running check_seg_replicates_after_processing function...") # check phase 47
        #try:
            #no_equal_seg_message = segimage_preprocessor.check_seg_replicates_after_processing()
            #no_equal_seg_message = "-"
            #series_state["last_successful_phase"] = 47
            #series_progress_state[series_group_name] = series_state
            #save_state(series_progress_state, segimage_preprocessor.series_progress_file)
        #except Exception as e:
            #log_error(input_dir, "check_seg_replicates_after_processing", e, LOG_FILENAME)
            #print(f"An unexpected error occurred during check_seg_replicates_after_processing. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
            #raise 
        
        #print("Running validate_npy_dimension_match function...") # check phase 48
        #try:
            #npy_dim_success_message = segimage_preprocessor.validate_npy_dimension_match()

            # Phase 48: Final step of imageseg preprocessing; all prior validation assumed complete.
            #series_state["last_successful_phase"] = 48
            #series_progress_state[series_group_name] = series_state
            #save_state(series_progress_state, segimage_preprocessor.series_progress_file)
        #except Exception as e:
            #log_error(input_dir, "validate_npy_dimension_match", e, LOG_FILENAME)
            #print(f"An unexpected error occurred during validate_npy_dimension_match. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
            #raise 

        print("Running generate_SegmentImagePreprocess_final_report function...")
        try:
            segimage_preprocessor.generate_SegmentImagePreprocess_final_report(single_to_multilabel_message, final_message_resize) ##riaggiungo no_equal_seg_message,npy_dim_success_message
        except Exception as e:
            log_error(input_dir, "generate_SegmentImagePreprocess_final_report", e, LOG_FILENAME)
            print(f"An unexpected error occurred during generate_SegmentImagePreprocess_final_report. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
            raise



    

