import pandas as pd
import numpy as np
import os
import pydicom
from datetime import datetime
import json
from collections import defaultdict
import SimpleITK as sitk
import scipy.ndimage 
import re
import gc
from utils import save_state, load_state, clear_log_file, log_error
from ...helpers import (
    generate_check_file_image_data,
    retrieve_existing_warnings_image_data,
    load_patient_series_seg_data,
    load_patient_mappings,
    load_series_mappings,
    get_orientation
)

LOG_FILENAME = "dicomseg_preprocessing_error.log"

class DicomSegReorientPreprocessor:

    def __init__(self, protocol, local_config, mapping_file, num_image_patients, num_slices_group, total_series, total_slices, series_group_name):
        self.protocol = protocol
        self.local_config = local_config
        self.mapping_file = mapping_file
        self.num_image_patients = num_image_patients
        self.num_slices_group = num_slices_group
        self.total_series = total_series
        self.total_slices = total_slices
        self.series_group_name = series_group_name
        self.input_dir = os.getenv("INPUT_DIR")
        self.host_input_dir = os.getenv("HOST_BASE_DATA_DIR")
        self.images_dir = os.path.join(self.input_dir, "IMAGES")
        self.state_file = os.path.join(self.input_dir, "image_validation_state.json")
        self.series_progress_file = os.path.join(self.input_dir, "validation_progress_by_series.json")
        self.study_path = os.getenv("ROOT_NAME")
        self.host_study_path = os.getenv("HOST_ROOT_DIR")
        self.output_directory_report = os.path.join(self.input_dir, "Reports", f"Image Data Reports {self.series_group_name}")
        self.host_output_directory_report = os.path.join(self.host_input_dir, "Reports", f"Image Data Reports {self.series_group_name}")
        self.output_directory_checkfile = os.path.join(self.study_path, "CHECKFILE")
        self.output_directory_data = os.path.join(self.study_path, "IMAGES")
        self.host_output_directory_data = os.path.join(self.host_study_path, "IMAGES")
        os.makedirs(self.output_directory_report, exist_ok=True)  
        os.makedirs(self.output_directory_checkfile, exist_ok=True)
        os.makedirs(self.output_directory_data, exist_ok=True)


    def check_and_reorient_dicomseg(self):
        """
        This function checks the orientation of DICOM SEG files,
        reorients them to LPS if necessary, and saves each segment as
        .npy and metadata .json metadata files in the corresponding segmentation subfolder within the output_directory_data structure.
        """
        local_config = self.local_config
        series_group = self.series_group_name
        
        patient_id_map = load_patient_mappings(local_config, self.input_dir)
        series_id_map = load_series_mappings(self.images_dir, series_group)
        
        # Flag to indicate if any slices were reoriented
        reoriented_flag_global  = False
        
        series_group_mapping = self.local_config["Local config"]["radiological data"]["series_mapping"]
        series_group_subfolder = series_group_mapping.get(series_group) #input name of the series
        
        # List all patient folders
        patient_folders = [
            f for f in os.listdir(self.images_dir)
            if os.path.isdir(os.path.join(self.images_dir, f)) and f != "Reports"
        ]
        
        for orig_patient_id  in patient_folders:
            patient_folder_path = os.path.join(self.images_dir, orig_patient_id)
            if os.path.isdir(patient_folder_path):

                # Map to new patient ID
                new_patient_id = patient_id_map.get(orig_patient_id)
                
                series_folder_name = os.path.normpath(series_group_subfolder).split(os.sep)[-1].strip()
                series_folder_path = os.path.join(patient_folder_path, series_folder_name)

                if os.path.isdir(series_folder_path):
                    full_series_key = os.path.join(orig_patient_id, series_folder_name)
                    # If mappings use Windows-style separators, normalize only for lookup
                    if "\\" in list(series_id_map.keys())[0]:  # Detect if mapping keys use '\'
                        full_series_key = full_series_key.replace("/", "\\")

                    new_series_name = series_id_map.get(full_series_key)

                    if new_series_name and "_" in new_series_name:
                        parts = new_series_name.split("_")
                        if len(parts) >= 3:
                            # Reorder: ["series001", "ct", "YWPXH"] → "ct_YWPXH_series001"
                            new_series_name2 = f"{parts[1]}_{parts[2]}_{parts[0]}"

                    seg_folder_name = f"seg_{parts[1]}_{parts[2]}_{parts[0]}"
                            
                    # Get the segmentation file (assuming .dcm files)
                    subfolders = [
                        sf for sf in os.listdir(series_folder_path)
                        if os.path.isdir(os.path.join(series_folder_path, sf))
                    ]
                    
                    segmentation_folder_path = os.path.join(series_folder_path, subfolders[0])
                    segmentation_file = next(
                        (f for f in os.listdir(segmentation_folder_path) if f.endswith(".dcm")),
                        None
                    )
    
                    segmentation_path = os.path.join(segmentation_folder_path, segmentation_file)
                    # Read DICOM file
                    ds = pydicom.dcmread(segmentation_path)
                    
                    # Initialize a dictionary to store anatomical structures by segment number
                    anatomical_structures = {}

                    # Iterate through each frame in the DICOM SEG dataset
                    for i, frame in enumerate(ds.PerFrameFunctionalGroupsSequence):
                        try:
                            # Get the segment number (anatomical structure ID)
                            segment_number = frame.SegmentIdentificationSequence[0].ReferencedSegmentNumber

                            # Extract the z-value from ImagePositionPatient (third element in the position list)
                            z_value = float(frame.PlanePositionSequence[0].ImagePositionPatient[2])

                            # If this segment number is not yet in the dictionary, add it
                            if segment_number not in anatomical_structures:
                                anatomical_structures[segment_number] = []

                            # Append the frame index and z-value to the corresponding segment's list
                            anatomical_structures[segment_number].append((i, z_value))

                        except AttributeError:
                            print(f"Frame {i+1} is missing PlanePositionSequence.")
                    
                    # Now check the orientation for each anatomical structure
                    for segment_number, frames in anatomical_structures.items():
                        # Check if frames are already sorted by z-value
                        is_sorted = all(frames[i][1] <= frames[i+1][1] for i in range(len(frames) - 1))

                        # Infer the S/I orientation based on z-values
                        orientation = 'S' if is_sorted else 'I'
                        
                        # Get the x (L/R) and y (P/A) orientations using the helper function
                        x_orientation, y_orientation = get_orientation(ds.SharedFunctionalGroupsSequence[0].PlaneOrientationSequence[0].ImageOrientationPatient)

                        # Full orientation is a combination of x, y, and z
                        full_orientation = f"{x_orientation}{y_orientation}{orientation}"

                        print(f"Segment {segment_number} orientation: {full_orientation}")

                        # Check if reorientation is needed
                        if full_orientation != 'LPS':
                            print(f"Segment {segment_number} needs reorientation.")
                            
                            # Extract the pixel data for this segment (get the indices of frames belonging to the segment)
                            segment_indices = [frame[0] for frame in frames]  # Get the indices of frames belonging to the segment
                            segmentation_tensor = ds.pixel_array[segment_indices, :, :]
                            
                            # Validate the number of slices for this segment
                            num_segment_slices = segmentation_tensor.shape[0] # num slices in the first position
                            print(f"Segment {segment_number} has {num_segment_slices} slices.")

                            # Reorient this segment using SimpleITK
                            sitk_image = sitk.GetImageFromArray(segmentation_tensor)
                            reoriented_segmentation = sitk.DICOMOrient(sitk_image, desiredCoordinateOrientation='LPS')
                            # Convert back to NumPy array
                            reoriented_array = sitk.GetArrayFromImage(reoriented_segmentation)
                            segmentation_tensor = reoriented_array
                            
                            reoriented_flag_global = True
                        else:
                            print(f"Segment {segment_number} is already in LPS orientation.")
                            
                            # If the segment doesn't need reorientation, store the original pixel data
                            segment_indices = [frame[0] for frame in frames]
                            segmentation_tensor = ds.pixel_array[segment_indices, :, :]
                            
                        # Prepare output path and filenames
                        patient_out_path = os.path.join(self.output_directory_data, new_patient_id)
                        series_out_path = os.path.join(patient_out_path, new_series_name)
                        seg_out_path = os.path.join(series_out_path, seg_folder_name)
                        
                        os.makedirs(seg_out_path, exist_ok=True)

                        segmentation_tensor = segmentation_tensor.astype(np.int16)
                        npy_filename = f"seg_{new_series_name2}_segment{segment_number}.npy"
                        np.save(os.path.join(seg_out_path, npy_filename), segmentation_tensor)
            
                        print(f"Saved segment {segment_number} tensor for {new_patient_id}/{new_series_name}")
            
        return reoriented_flag_global
    
    
    def unify_multiple_segments(self, duplicate_label_flag, duplicate_labels, protocol_dict):
        """
        Unifies multiple segmentations on disk for each patient/series, based on duplicate label info.
        Merges tensors, saves to disk using protocol codes, and deletes original segment files.
        
        Parameters:
            duplicate_label_flag (bool): If True, resolve duplicate labels. Otherwise, skip this step.
            duplicate_labels (defaultdict): Duplicate labels and associated segment codes,
                e.g., {'Label': [{1, 2}, {1, 2}, {1, 2}]}.
            protocol_dict (dict): Protocol mapping segment codes to labels,
                e.g., {1: 'Label'}.
        Returns:
            duplicate_labels_message (str): Message describing the outcome of the operation.
        """
        duplicate_labels_message = ""
        current_group = None
        segmentations = {}
        series_group = self.series_group_name
        output_directory_data = self.output_directory_data
        
        if not duplicate_label_flag:
            duplicate_labels_message = "No presence of labels with multiple segments."
            # If the flag is False, skip processing and return the input as-is
            return duplicate_labels_message
        
        protocol_dict = {int(k): v for k, v in protocol_dict.items()}

        def process_and_save(current_group, segmentations):
            patient_id, series_id = current_group
            for label, segment_sets in duplicate_labels.items():
                unique_seg_sets = {frozenset(s) for s in segment_sets}  # deduplicate sets
                for seg_set in unique_seg_sets:
                    seg_set = set(seg_set)
                    existing = [s for s in seg_set if s in segmentations]
                    if not existing:
                        continue
    
                    # Merge tensors
                    merged_tensor = None
                    for seg_code in existing:
                        tensor = segmentations[seg_code]['tensor']
                        if merged_tensor is None:
                            merged_tensor = np.zeros_like(tensor, dtype=np.int16)
                        merged_tensor |= tensor
    
                    # Get protocol code
                    protocol_code = next((code for code, lbl in protocol_dict.items() if lbl == label), None)
                    if protocol_code is None:
                        raise ValueError(f"Missing protocol code for label: {label}")
    
                    # Set label value in tensor
                    merged_tensor[merged_tensor > 0] = protocol_code
    
                    # Remove old files
                    for seg_code in existing:
                        old_path = segmentations[seg_code]['path']
                        if os.path.exists(old_path):
                            os.remove(old_path)
                            print(f"[{patient_id}] Removed segment{seg_code}")

                    parts = series_id.split("_")
                    if len(parts) >= 3:
                        reordered_series_id = "_".join(parts[1:]) + "_" + parts[0]  # ct_YWPXH_series001
                    else:
                        reordered_series_id = series_id  # fallback in case format is unexpected

                    # Save merged tensor
                    out_folder = os.path.dirname(segmentations[existing[0]]['path'])
                    out_filename = f"seg_{reordered_series_id}.npy"
                    out_path = os.path.join(out_folder, out_filename)
                    np.save(out_path, merged_tensor)
                    print(f"[{patient_id}] Merged {label} → segment{protocol_code}")
    
        # Iterate with streaming logic
        for patient_id, series_id, npy_path, tensor, segment_number in load_patient_series_seg_data(series_group, output_directory_data):
            group = (patient_id, series_id)
    
            if current_group is not None and group != current_group:
                process_and_save(current_group, segmentations)
                segmentations.clear()
    
            current_group = group
            segmentations[segment_number] = {
                'path': npy_path,
                'tensor': tensor
            }
    
        # Final flush
        if current_group:
            process_and_save(current_group, segmentations)
                
        duplicate_labels_message = "Labels with multiple segments were successfully unified according to the protocol."
    
        return duplicate_labels_message
    

    def _build_multilabel(self, segmentations):
        """
        Helper method to build multi-label tensor within one patient-series.
        
        Args:
            segmentations (dict): {segment_number: (tensor, npy_path), ...}
        
        Returns:
            multi_label_tensor (np.ndarray): Combined multi-label segmentation tensor.
        """
        # Assumes all tensors have same shape
        first_tensor = next(iter(segmentations.values()))[0]
        num_slices, height, width = first_tensor.shape
        multi_label_tensor = np.zeros((num_slices, height, width), dtype=np.int16)
    
        for segment_number, (tensor, _) in segmentations.items():
            for z in range(num_slices):
                binary_mask = tensor[z] != 0 # instead of == 1
                multi_label_tensor[z][binary_mask] = segment_number
    
        return multi_label_tensor
    

    def single_to_multi_label_conversion(self): 
        """
        Converts single-label segmentations on disk to multi-label format if needed.
        """
        series_group = self.series_group_name
        seg_group_config = self.protocol.get(series_group, {}).get("segmentation", {})
        input_format = seg_group_config["segmentation_input_format"]["selected"]
        output_directory_data = self.output_directory_data
        
        # Check if multi-label conversion is necessary
        if input_format == "multi-mask":
            single_to_multilabel_message = "Multi-label conversion skipped: input format is already 'multi-mask'."
            return single_to_multilabel_message

        if input_format == "single-mask" and len(seg_group_config["segments_labels"]) == 2:
            for patient_id, series_id, npy_path, tensor, segment_number in load_patient_series_seg_data(series_group, output_directory_data):
                if segment_number is None:
                    continue
                    
                # Get directory and filename
                directory, filename = os.path.split(npy_path)
            
                # Remove "_segment{n}" using regex
                new_filename = re.sub(r'_segment\d+', '', filename)
            
                # Only rename if different
                if filename != new_filename:
                    new_path = os.path.join(directory, new_filename)
                    os.rename(npy_path, new_path)
                    print(f"[{patient_id}] Renamed: {filename} → {new_filename}")
                    
            single_to_multilabel_message = "Multi-label conversion skipped: input format is 'single-mask' with 1 label."
            return single_to_multilabel_message
    
        all_segment_file_paths = []
        current_key = None
        segmentations = {}

        for patient_id, series_id, npy_path, tensor, segment_number in load_patient_series_seg_data(series_group, output_directory_data):
            if segment_number is None:
                continue  # Skip entries with no segment label
                
            key = (patient_id, series_id)
            
            # When patient-series changes, process previous group
            if current_key is not None and key != current_key:
                # Process multi-label tensor for current_key
                multi_label_tensor = self._build_multilabel(segmentations)
                
                # Save multi-label tensor immediately
                example_path = next(iter(segmentations.values()))[1]
                directory, filename = os.path.split(example_path)
                base_filename = re.sub(r'_segment\d+', '', filename)
                
                output_path = os.path.join(directory, base_filename)
                np.save(output_path, multi_label_tensor)
                
                # Clear for next patient-series
                segmentations.clear()
            
            # Accumulate segmentations for current patient-series
            segmentations[segment_number] = (tensor, npy_path)
            if re.search(r'_segment\d+\.npy$', npy_path):
                all_segment_file_paths.append(npy_path)
            current_key = key
    
        # Process last patient-series after loop ends
        if current_key is not None and segmentations:
            multi_label_tensor = self._build_multilabel(segmentations)
            example_path = next(iter(segmentations.values()))[1]
            directory, filename = os.path.split(example_path)
            base_filename = re.sub(r'_segment\d+', '', filename)
            
            output_path = os.path.join(directory, base_filename)
            np.save(output_path, multi_label_tensor)

        # Remove original single-label files
        for path in all_segment_file_paths:
            try:
                os.remove(path)
                print(f"Deleted: {path}")
            except Exception as e:
                print(f"Could not delete {path}: {e}")
                    
        single_to_multilabel_message = "Multi-label conversion successful."
        return single_to_multilabel_message


    def discard_black_seg_slices(self):
        """
        Reads discarded_slices.csv to identify black slices removed from volume data,
        and removes the corresponding slices from segmentation .npy files on disk.
        If no changes are needed, leaves the files untouched.
        """
        removal_message = ""
        discarded_slices_file = os.path.join(self.images_dir, "discarded_slices.csv")
        series_group = self.series_group_name

        if not os.path.exists(discarded_slices_file):
            # If the CSV is not found, return a message indicating no slices were removed
            removal_message += "No segmentation slices were removed."
            return removal_message

        # Load discarded slices data from CSV
        try:
            discarded_slices_df = pd.read_csv(discarded_slices_file)
        except Exception as e:
            raise Exception(f"Failed to read discarded slices CSV: {str(e)}")

        # Load patient and series ID maps
        patient_id_map = load_patient_mappings(self.local_config, self.input_dir)
        series_id_map = load_series_mappings(self.images_dir, series_group)

        # Iterate through each row in the CSV
        discarded_by_key = defaultdict(list)
        for _, row in discarded_slices_df.iterrows():
            orig_patient = row['Patient Name']
            orig_series = row['Series Folder Name']
            lookup_key = f"{orig_patient}\\{orig_series}"
            discarded_index = row['Discarded Slice Index'] - 1  # Convert 1-based to 0-based

            patient_id = patient_id_map.get(orig_patient)
            series_id = series_id_map.get(lookup_key)

            if patient_id is None or series_id is None:
                continue
            
            key = (patient_id, series_id)
            discarded_by_key[key].append(discarded_index)

        total_removed = 0
        total_invalid = 0

        for (patient, series), indices in discarded_by_key.items():
            # Locate the segmentation folder
            series_folder  = os.path.join(self.output_directory_data, patient, series)
            if not os.path.exists(series_folder ):
                print(f"Warning: Series folder not found for {patient}/{series}")
                continue

            # Find the segmentation folder inside the series folder (e.g. folder starting with "seg")
            segmentation_folder = next(
                (sf for sf in os.listdir(series_folder) if sf.startswith("seg") and os.path.isdir(os.path.join(series_folder, sf))),
                None
            )
            if not segmentation_folder:
                print(f"Warning: Segmentation folder not found in {series_folder}")
                continue

            seg_folder = os.path.join(series_folder, segmentation_folder)

            # List .npy segmentation files
            npy_files = [f for f in os.listdir(seg_folder) if f.endswith('.npy')]
            if not npy_files:
                print(f"Warning: No .npy files in {seg_folder}")
                continue
    
            npy_path = os.path.join(seg_folder, npy_files[0])

            try:
                tensor = np.load(npy_path)
                if len(tensor.shape) < 3:
                    print(f"Warning: tensor in {npy_path} has less than 3 dimensions: {tensor.shape}, skipping slice removal.")
                    continue
            except Exception as e:
                print(f"Failed to load {npy_path}: {e}")
                continue

            # Validate and remove only valid slice indices
            valid_indices = [i for i in indices if 0 <= i < tensor.shape[0]] #2
            invalid_indices = [i for i in indices if i < 0 or i >= tensor.shape[0]] #2
            total_invalid += len(invalid_indices)

            if valid_indices:
                # Sort in reverse to avoid reindexing issues
                valid_indices.sort(reverse=True)
                for idx in valid_indices:
                    tensor = np.delete(tensor, idx, axis=0)
                    
                tensor = tensor.astype(np.int16)
                # Overwrite .npy file with modified tensor
                np.save(npy_path, tensor)
                total_removed += len(valid_indices)
                print(f"[{patient}/{series}] Removed {len(valid_indices)} black slice(s) from {os.path.basename(npy_path)}")

        # Build final message
        if total_removed > 0:
            removal_message += f"Number of removed segmentation slices: {total_removed}."
        else:
            removal_message += "No segmentation slices were removed."
    
        if total_invalid > 0:
            removal_message += f" (Warning: {total_invalid} slice indices were out of bounds.)"
    
        return removal_message   
    

    def resize_seg(self):  
        """
        Loads and resizes all .npy segmentation files in the output directory to the target size defined in the protocol.
        Only overwrites the files if resizing is needed.
        
        Returns:
        str: Message indicating whether resizing was performed.
        """
        series_group = self.series_group_name
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

            resized_segmentation = resized_segmentation.astype(np.int16)
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
                 f"See the detailed report at: {host_report_filename}" #!! Path del container, deve essere quello locale.ok
            )
        else:
            npy_dim_success_message += "Dimension validation successful: all corresponding .npy volumes and segmentation files match in dimensions."
            return npy_dim_success_message
    

    def generate_DicomSegReorientPreprocess_final_report(self, reoriented_flag_global, duplicate_labels_message, single_to_multilabel_message, removal_message, final_message_resize, npy_dim_success_message):
        """
        Generate a comprehensive final report for DicomSegReorientTransform process.
    
        Parameters:
        - reoriented_flag_global: Boolean variable about reorientation status.
        - duplicate_labels_message: Message about multiple segment codes union.
        - single_to_multilabel_message: Message about single-to-multilabel conversion.
        - removal_message: Message about slice removal.
        - final_message_resize: Message about resizing status.
        - npy_dim_success_message: Message about NPY dimensions check. 
        """
        
        # Get the current datetime
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
        
        # Define the report filename and path
        report_filename = "4.DicomSegReorientPreprocess_final_report.txt"
        report_filepath = os.path.join(self.output_directory_report, report_filename)
        
        # Initialize report content with ordered titles and messages
        report_content = []
        
        # Add report header
        report_content.append(f"Report generated on: {formatted_datetime}\n")
        report_content.append("DICOM segmentation reorientation and transformation report:\n")
        
        # Segmentation reorientation status
        report_content.append("Segmentation reorientation check:")
        if reoriented_flag_global:
            report_content.append("- Segmentations reorientation occurred.")
        report_content.append("- Reference segmentation orientation: LPS.\n")

        if duplicate_labels_message:
            report_content.append("Labels with multiple segments union:")
            report_content.append("- " + duplicate_labels_message + "\n")
        
        # Single to multi-label conversion status
        report_content.append("Multi-label conversion status:")
        report_content.append("- " + single_to_multilabel_message + "\n")

        # Segmentation slice removal status
        report_content.append("Segmentation slice removal status:")
        if removal_message:
            report_content.append("- " + removal_message + "\n")
        else:
            report_content.append("- No slices were removed." + "\n")

        # Add resizing status
        report_content.append("Segmentation resizing status:")
        if final_message_resize:
            report_content.append("- " + final_message_resize + "\n")
        
        # Export to NPY format status
        report_content.append("Segmentation tensor save status:")
        final_message_npy = f"Tensors saved in .npy format into segmentation folders inside series folders within patient folders in {self.host_output_directory_data}." #!! Path del container, deve essere quello locale.ok
        report_content.append("- " + final_message_npy + "\n")

        # NPY dimension check status
        report_content.append("NPY dimensions check status:")
        if npy_dim_success_message:
            report_content.append("- " + npy_dim_success_message + "\n")
            
        # Final report summary
        report_content.append("Summary:")
        report_content.append("All checks and transformations completed.\n")
        
        # Write the report to the specified file
        with open(report_filepath, 'w') as report_file:
            report_file.write("\n".join(report_content))
        
        # Print a message indicating that the final report has been generated
        print(f"DicomSegReorientPreprocess final report saved to {report_filepath}")


def run_dicomseg_reorient_preprocessor(protocol, local_config, mapping_file, series_group_name):
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
    
    dicomseg_preprocessor = DicomSegReorientPreprocessor(protocol, local_config, mapping_file, num_image_patients, num_slices_group, total_series, total_slices, series_group_name)

    series_progress_state =  load_state(dicomseg_preprocessor.series_progress_file)

    # Access or create the per-series progress dictionary
    series_state = series_progress_state.setdefault(series_group_name, {})
    last_phase_done = series_state.get("last_successful_phase", 0)

    if last_phase_done < 48:

        print("Running check_and_reorient_dicomseg function...")
        try:
            reoriented_flag = dicomseg_preprocessor.check_and_reorient_dicomseg()
        except Exception as e:
            log_error(input_dir, "check_and_reorient_dicomseg", e, LOG_FILENAME)
            print(f"An unexpected error occurred during check_and_reorient_dicomseg. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
            raise 

        print("Running unify_multiple_segments function...")
        try:
            duplicate_labels_message = dicomseg_preprocessor.unify_multiple_segments(
                    image_state.get("duplicate_label_flag"), 
                    image_state.get("duplicate_labels"), 
                    image_state.get("protocol_dict")
                )
        except Exception as e:
            log_error(input_dir, "unify_multiple_segments", e, LOG_FILENAME)
            print(f"An unexpected error occurred during unify_multiple_segments. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
            raise 
        
        print("Running single_to_multi_label_conversion function...")
        try:
            single_to_multilabel_message = dicomseg_preprocessor.single_to_multi_label_conversion()
        except Exception as e:
            log_error(input_dir, "single_to_multi_label_conversion", e, LOG_FILENAME)
            print(f"An unexpected error occurred during single_to_multi_label_conversion. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
            raise 

        print("Running discard_black_seg_slices function...")
        try:
            removal_message = dicomseg_preprocessor.discard_black_seg_slices()
        except Exception as e:
            log_error(input_dir, "discard_black_seg_slices", e, LOG_FILENAME)
            print(f"An unexpected error occurred during discard_black_seg_slices. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
            raise 

        print("Running resize_seg function...")
        try:
            final_message_resize = dicomseg_preprocessor.resize_seg()
        except Exception as e:
            log_error(input_dir, "resize_seg", e, LOG_FILENAME)
            print(f"An unexpected error occurred during resize_seg. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
            raise 

        print("Running validate_npy_dimension_match function...") # check phase 48
        try:
            npy_dim_success_message = dicomseg_preprocessor.validate_npy_dimension_match()
        
            # Phase 48: Final step of dicomseg preprocessing; all prior validation assumed complete.
            series_state["last_successful_phase"] = 48
            series_progress_state[series_group_name] = series_state
            save_state(series_progress_state, dicomseg_preprocessor.series_progress_file)
        except Exception as e:
            log_error(input_dir, "validate_npy_dimension_match", e, LOG_FILENAME)
            print(f"An unexpected error occurred during validate_npy_dimension_match. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
            raise 

        print("Running generate_DicomSegReorientPreprocess_final_report function...")
        try:
            dicomseg_preprocessor.generate_DicomSegReorientPreprocess_final_report(reoriented_flag, duplicate_labels_message, single_to_multilabel_message, removal_message, final_message_resize, npy_dim_success_message)
        except Exception as e:
            log_error(input_dir, "generate_DicomSegReorientPreprocess_final_report", e, LOG_FILENAME)
            print(f"An unexpected error occurred during generate_DicomSegReorientPreprocess_final_report. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
            raise 










        