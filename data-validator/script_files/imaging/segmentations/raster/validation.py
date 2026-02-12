import numpy as np
import os 
from datetime import datetime
from PIL import Image
import json
from utils import save_state, load_state, clear_log_file, log_error
from ...helpers import (
    generate_check_file_image_data,
    retrieve_existing_warnings_image_data
)

LOG_FILENAME = "2D_segmentation_validation_error.log"

class SegmentImageValidator:

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
        self.input_dir = os.getenv("INPUT_DIR")
        self.images_dir = os.path.join(self.input_dir, "IMAGES")
        self.host_input_dir = os.getenv("HOST_BASE_DATA_DIR")
        self.state_file = os.path.join(self.input_dir, "image_validation_state.json")
        self.series_progress_file = os.path.join(self.input_dir, "validation_progress_by_series.json")
        self.output_directory_report = os.path.join(self.input_dir, "Reports", f"Image Data Reports {self.series_group_name}")
        self.host_output_directory_report = os.path.join(self.host_input_dir, "Reports", f"Image Data Reports {self.series_group_name}")
        self.output_directory_checkfile = os.path.join(self.study_path, "CHECKFILE")
        os.makedirs(self.output_directory_report, exist_ok=True)  
        os.makedirs(self.output_directory_checkfile, exist_ok=True)       


    def check_uniform_seg(self): #43
        """
        Checks for completely uniform segmentations (all-black, all-white, or all-gray) in patient series folders.
        Generates a check file and, if any uniform images are found, an error report.
        """
        phase_number = 43  
        phase_data = self.mapping_file.get("check_phase", {}).get(str(phase_number), {})
        error_descriptions = phase_data.get("errors", {})
        phase_name = phase_data.get("name", f"Phase {phase_number}")  
        
        # Format check and report names dynamically
        check_name = phase_name.lower().replace(" ", "_") 
        report_name = f"3.043.{check_name}_report"

        series_group = self.series_group_name
        series_group_mapping = self.local_config["Local config"]["radiological data"]["series_mapping"]
        series_group_subfolder = series_group_mapping.get(series_group) #input name of the series
        seg_group_config = self.protocol.get(series_group, {}).get("segmentation", {}) #original self.protocol["segmentation"]
        
        segmentation_format = seg_group_config['segmentation file format']['selected']
        seg_input_format = seg_group_config["segmentation_input_format"]["selected"]
        segments_number = seg_group_config["segments_number"]
        multiple_segmentation_flag = seg_input_format == "single-mask" and segments_number > 1

        error_details = []  # List to store affected files
        total_affected_segmentations = 0
        
        # Retrieve existing warnings
        check_file_path = os.path.join(self.output_directory_checkfile, "image_data_check_file.json")
        existing_warnings = retrieve_existing_warnings_image_data(check_file_path)
        
        # Get patient folders (excluding non-directories like "Reports")
        patient_folders = [
            f for f in os.listdir(self.images_dir)
            if os.path.isdir(os.path.join(self.images_dir, f)) and f != "Reports"
        ]
        
        # Iterate through patients
        for patient_folder in patient_folders:
            patient_path = os.path.join(self.images_dir, patient_folder)
    
            if os.path.isdir(patient_path):
                series_folder = os.path.normpath(series_group_subfolder).split(os.sep)[-1].strip()
                series_path = os.path.join(patient_path, series_folder)

                if os.path.isdir(series_path):
                    # Find the unique subfolder (segmentation folder)
                    subfolders = [f for f in os.listdir(series_path) if os.path.isdir(os.path.join(series_path, f))]
                    
                    if len(subfolders) != 1:
                        continue  

                    segmentation_path = os.path.join(series_path, subfolders[0])

                    # Ensure segmentation folder contains files
                    if not os.path.exists(segmentation_path) or not os.listdir(segmentation_path):
                        continue  # Skip empty folders

                    # Get segmentation files
                    segmentation_files = [
                        f for f in os.listdir(segmentation_path)
                        if f.lower().endswith(segmentation_format)
                    ]
                    
                    for segmentation_file in segmentation_files:
                        segmentation_file_path = os.path.join(segmentation_path, segmentation_file)
                        
                        try:
                            # Open image and convert to numpy array
                            segmentation_image  = Image.open(segmentation_file_path).convert("L")  
                            seg_array = np.array(segmentation_image)
        
                            # Check if the image has uniform pixel values (all-black, all-white, or all-gray)
                            if np.std(seg_array) == 0:
                                error_details.append((patient_folder, series_folder, subfolders[0], segmentation_file))
                                total_affected_segmentations += 1
                        except Exception as e:
                            print(f"Error processing {segmentation_file_path}: {str(e)}")

        error_counts={"E4301": total_affected_segmentations if total_affected_segmentations > 0 else None}
        # Store the current datetime for report generation
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")                        
        # Generate JSON check file
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
        # If no issues found, no need to generate a report
        if not error_details:
            uniform_seg_message = "No uniform (completely black, white, or gray) segmentations detected."
            return uniform_seg_message, multiple_segmentation_flag
    
        # Generate report file
        report_path = os.path.join(self.output_directory_report, f"{report_name}.txt")
        host_report_path = os.path.join(self.host_output_directory_report, f"{report_name}.txt")
    
        with open(report_path, "w") as report_file:
            report_file.write(f"Report generated on: {current_datetime}\n\n")
            report_file.write(f"{phase_name} report:\n\n")
            report_file.write(f"Error E4301: {error_descriptions.get('E4301', None)}\n")
            report_file.write("- Details:\n")
    
            for patient, series, subfolder, segmentation in error_details:
                report_file.write(f"  - Patient: {patient}, Series: {series}, Segmentation Folder: {subfolder}, File: {segmentation}\n")
    
            report_file.write(f"\nTotal Errors: {total_affected_segmentations}\n")
    
        print(f"Uniform segmentation report saved at: {report_path}")
        raise Exception(
            f"Uniform segmentations detected. "
            f"See the detailed report at: {host_report_path}"
        ) #!! Path del container, deve essere quello locale.OK
    

    def validate_segment_code_conformity_2D(self, multiple_seg_file_flag): #44
        """
        Validates segmentation conformity for various image formats (PNG, JPG, TIFF, etc.) in a 2D configuration.
        - If single-mask with multiple segmentation files, checks that the number of files matches `segments_number` 
          and that filenames correspond to `segments_labels`.
        - If multi-mask, checks that the unique pixel values in the segmentation images match the expected `segmentation_code`.
        """
        phase_number = 44  
        phase_data = self.mapping_file.get("check_phase", {}).get(str(phase_number), {})
        error_descriptions = phase_data.get("errors", {})
        phase_name = phase_data.get("name", f"Phase {phase_number}")  
        
        # Format check and report names dynamically
        check_name = phase_name.lower().replace(" ", "_") 
        report_name = f"3.044.{check_name}_report"

        series_group = self.series_group_name
        series_group_mapping = self.local_config["Local config"]["radiological data"]["series_mapping"]
        series_group_subfolder = series_group_mapping.get(series_group) #input name of the series
        seg_group_config = self.protocol.get(series_group, {}).get("segmentation", {}) #original self.protocol["segmentation"]

        seg_file_format = seg_group_config["segmentation file format"]["selected"]
        seg_input_format = seg_group_config["segmentation_input_format"]["selected"]
        segments_number = seg_group_config["segments_number"]
        segments_labels = seg_group_config["segments_labels"]
  
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        check_file_path = os.path.join(self.output_directory_checkfile, "image_data_check_file.json")
        existing_warnings = retrieve_existing_warnings_image_data(check_file_path)

        # Initializations   
        num_errors = 0
        has_error = False
        count_errors_content = ""
        label_errors_content = ""
        multi_mask_file_error_content = ""
        report_content = ""
        discrepancy_info = ""
        error_counts = {"E4401": None}

        patient_folders = [f for f in os.listdir(self.images_dir) if os.path.isdir(os.path.join(self.images_dir, f)) and f != "Reports"]

        # Iterate through all patient folders
        for patient_folder in patient_folders:
            patient_folder_path = os.path.join(self.images_dir, patient_folder)
            if os.path.isdir(patient_folder_path):
                series_folder = os.path.normpath(series_group_subfolder).split(os.sep)[-1].strip()
                series_folder_path = os.path.join(patient_folder_path, series_folder)
                
                if os.path.isdir(series_folder_path):
                    
                    # Find the unique folder in the series folder 
                    segmentation_folder = None
                    for subfolder in os.listdir(series_folder_path):
                        subfolder_path = os.path.join(series_folder_path, subfolder)
                        if os.path.isdir(subfolder_path):
                            segmentation_folder = subfolder_path
                            break  # Select the first (and only) subfolder containing the png/jpg/tiff file

                    if seg_input_format == "single-mask":
                        seg_files = [f for f in os.listdir(segmentation_folder) if f.endswith(seg_file_format)]

                        # Check if number of files matches segments_number
                        if len(seg_files) != segments_number:
                            count_errors_content += (f"  - Incorrect number of segmentation files ({len(seg_files)}) "
                                                     f"in folder {segmentation_folder}.\n")
                            num_errors += 1
                            has_error = True
                        
                        if multiple_seg_file_flag: 
                            # Create a mapping of original filenames to their normalized versions
                            filename_mapping = {
                                f: f.lower().replace("_", " ").replace("-", " ").replace("  ", " ")  # Replace double spaces if they appear
                                for f in seg_files
                            }
                            # Normalize actual filenames for comparison
                            normalized_actual_filenames = set(filename_mapping.values())
                            
                            # Keep original expected filenames for reporting
                            original_expected_filenames = {f"{label}{seg_file_format}" for label in segments_labels if label.lower() != "background"}
                            
                            # Check if filenames match expected segment labels
                            filtered_labels = [label.lower().replace("_", " ").replace("-", " ").replace("  ", " ") for label in segments_labels if label.lower() != "background"]
                            expected_filenames = {f"{label}{seg_file_format}".lower() for label in filtered_labels}
                            
                            if normalized_actual_filenames != expected_filenames:
                                #missing_files = expected_filenames - normalized_actual_filenames
                                extra_files = normalized_actual_filenames - expected_filenames

                                if extra_files:
                                    # Retrieve original filenames corresponding to the mismatched normalized names
                                    original_extra_files = {orig for orig, norm in filename_mapping.items() if norm in extra_files}
                                    label_errors_content += (f"  - Incorrect segmentation filenames in {segmentation_folder}. "
                                                             f"Mismatched filenames: {original_extra_files}.\n")
                                    num_errors += len(original_extra_files) 
                                    has_error = True
                    else:
                        # Process the image segmentation files (PNG, JPG, TIFF, etc.)
                        image_files = [f for f in os.listdir(segmentation_folder) if f.endswith(seg_file_format)]
                        for image_file in image_files:
                            image_file_path = os.path.join(segmentation_folder, image_file)

                            try:
                                # Open the image file and convert to numpy array
                                image = Image.open(image_file_path).convert("L")
                                seg_array = np.array(image)

                                unique_values_before = np.unique(seg_array)
                                filtered_unique_values = unique_values_before[unique_values_before != 0]
        
                                # Verify the number of unique values matches segments_number
                                if len(filtered_unique_values) != segments_number:
                                    count_errors_content += f"  - Number of unique values ({len(filtered_unique_values)}) in segmentation array does not match protocol segments_number ({segments_number}) for patient {patient_folder}, series {series_folder}\n"
                                    num_errors += 1
                                    has_error = True
                            except Exception as e:
                                print(f"Error processing {image_file_path}: {str(e)}")

                    if seg_input_format == "multi-mask":
                        # Check if segmentation folder contains exactly one segmentation file
                        seg_files = [f for f in os.listdir(segmentation_folder) if f.endswith(seg_file_format)]
                    
                        if len(seg_files) != 1:  # If there's not exactly one segmentation file
                            multi_mask_file_error_content += (f"  - Incorrect number of segmentation files ({len(seg_files)}) "
                                                     f"in folder {segmentation_folder}. Expected exactly 1 file.\n")
                            num_errors += 1
                            has_error = True

        # Assemble the report content with separate sections
        if count_errors_content:
            count_errors_content_title = f"  Expected number of segmentation files: {segments_number}\n"
            discrepancy_info += "  --- Segment Count Errors ---\n\n" + count_errors_content_title + count_errors_content + "\n"
        if label_errors_content:
            label_errors_content_title = f"  Expected segmentation filenames: {original_expected_filenames}\n"
            discrepancy_info += "  --- Segmentation Label Errors ---\n\n" + label_errors_content_title + label_errors_content + "\n"
        if multi_mask_file_error_content:
            discrepancy_info += "  --- Multi-mask Segmentation File Count Errors ---\n\n" + multi_mask_file_error_content + "\n"
        

        error_counts = {"E4401": num_errors if num_errors > 0 else None}
        # Generate the JSON check file using the helper function
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
    
        if has_error:
            report_name = f"{report_name}.txt"
            report_filename = os.path.join(self.output_directory_report, report_name)
            host_report_filename = os.path.join(self.host_output_directory_report, report_name)
            report_content += f"Report generated on: {current_datetime}\n\n"
            report_content += f"{phase_name} report:\n\n"
            report_content += f"Error E4401: {error_descriptions.get('E4401', None)}\n"
            report_content += "- Details:\n\n"
            report_content += discrepancy_info
            report_content += f"\nTotal Errors: {num_errors}"
    
            with open(report_filename, "w") as report_file:
                report_file.write(report_content)
            raise Exception(
                f"Segment code conformity check failed. "
                f"See the detailed report at: {host_report_filename}") #!! Path del container, deve essere quello locale.OK
        else:
            if seg_input_format == "single-mask" and multiple_seg_file_flag:
                segment_code_conformity_message = (
                    "Segmentation validation successful: all segmentation files conform to the expected labels and count."
                )
            else:
                segment_code_conformity_message = (
                    "Segmentation validation successful: segmentation count conform to the protocol."
                )
            
            return segment_code_conformity_message
        

    def single_label_seg_2Dshape_check(self, multiple_segmentation_flag):
        """
        Checks for inconsistent shapes among 2D segmentations in the input directory structure.
        Only performs the check if the segmentation input format is 'single-mask' and multiple segments are present.
    
        Returns:
        - Success message string if no inconsistencies are detected.
    
        Raises:
        - Exception if shape inconsistencies are detected.
        """
        # Only proceed if condition is met
        if not multiple_segmentation_flag:
            shape_inconsistency_message = "Shape consistency check is not needed. Segmentations are not single-mask with multiple segments."
            return shape_inconsistency_message

        phase_number = 45  
        phase_data = self.mapping_file.get("check_phase", {}).get(str(phase_number), {})
        error_descriptions = phase_data.get("errors", {})
        phase_name = phase_data.get("name", f"Phase {phase_number}")  
        
        # Format check and report names dynamically
        check_name = phase_name.lower().replace(" ", "_") 
        report_name = f"3.045.{check_name}_report" 

        series_group = self.series_group_name
        series_group_mapping = self.local_config["Local config"]["radiological data"]["series_mapping"]
        series_group_subfolder = series_group_mapping.get(series_group)
        seg_group_config = self.protocol.get(series_group, {}).get("segmentation", {})
        seg_file_format = seg_group_config.get("segmentation file format", {}).get("selected").lower()
        
        shape_inconsistencies = {}
        inconsistent_shape_detected = False

        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        check_file_path = os.path.join(self.output_directory_checkfile, "image_data_check_file.json")
        existing_warnings = retrieve_existing_warnings_image_data(check_file_path)

        patient_folders = [
            f for f in os.listdir(self.images_dir)
            if os.path.isdir(os.path.join(self.images_dir, f)) and f != "Reports"
        ]
    
        for patient_folder in patient_folders:
            patient_folder_path = os.path.join(self.images_dir, patient_folder)
            series_folder = os.path.normpath(series_group_subfolder).split(os.sep)[-1].strip()
            series_folder_path = os.path.join(patient_folder_path, series_folder)
    
            if os.path.isdir(series_folder_path):
                # Search for segmentation folder within series
                segmentation_folder = None
                for subfolder in os.listdir(series_folder_path):
                    subfolder_path = os.path.join(series_folder_path, subfolder)
                    if os.path.isdir(subfolder_path):
                        segmentation_folder = subfolder_path
                        break
    
                # Filter image files based on selected format
                image_files = [f for f in os.listdir(segmentation_folder) if f.lower().endswith(seg_file_format)]

                reference_shape = None
                for image_file in image_files:
                    image_path = os.path.join(segmentation_folder, image_file)
                    image = Image.open(image_path)
                    image_shape = image.size  # (width, height)
    
                    if reference_shape is None:
                        reference_shape = image_shape
                    elif image_shape != reference_shape:
                        inconsistent_shape_detected = True
                        key = (patient_folder, series_folder)
                        if key not in shape_inconsistencies:
                            shape_inconsistencies[key] = []
                        shape_inconsistencies[key].append((image_file, image_shape))
        
        num_errors = len(shape_inconsistencies)
        error_counts = {"E4501": num_errors if num_errors > 0 else None}
    
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
    
        if inconsistent_shape_detected:
            report_filename = os.path.join(self.output_directory_report, f"{report_name}.txt")
            host_report_filename = os.path.join(self.host_output_directory_report, f"{report_name}.txt")
            report_content = f"Report generated on: {current_datetime}\n\n"
            report_content += f"{phase_name} report:\n\n"
            report_content += f"Error E4501: {error_descriptions.get('E4501', 'Segmentation slice shape inconsistency detected.')}\n"
            report_content += "- Details: single-label segmentations with inconsistent shapes were detected in:\n"
            for (patient_folder, series_folder), details in shape_inconsistencies.items():
                report_content += f"  - Patient: {patient_folder}, Series: {series_folder}\n"
            report_content += f"\nTotal Errors: {num_errors}"
    
            with open(report_filename, "w") as report_file:
                report_file.write(report_content)
    
            raise Exception(
                f"Shape inconsistency detected in single-label segmentations within at least one series. "
                f"See the detailed report at: {host_report_filename}") #!! Path del container, deve essere quello locale.OK
            
        shape_inconsistency_message = "All single-label segmentations within each series have consistent shapes."
        return shape_inconsistency_message
    

    def check_pixel_overlap_single_label_seg_2D(self, multiple_segmentation_flag):
        """
        Checks for overlapping pixels between segmentation masks in single-label 2D segmentations.
    
        Only performs the check if the segmentation input format is 'single-mask' and multiple segments are present.
    
        Args:
            multiple_segmentation_flag (bool): Whether to perform the overlap check.
    
        Returns:
            str: Status message indicating result of overlap check.
    
        Raises:
            Exception: When overlapping pixels are detected, with report file location.
        """
        # Only proceed if condition is met
        if not multiple_segmentation_flag:
            overlapping_pixel_message = "Overlapping pixels check is not needed. Segmentations are not single-mask with multiple segments."
            return overlapping_pixel_message

        series_group = self.series_group_name
        series_group_mapping = self.local_config["Local config"]["radiological data"]["series_mapping"]
        series_group_subfolder = series_group_mapping.get(series_group)
        seg_group_config = self.protocol.get(series_group, {}).get("segmentation", {})
        seg_file_format = seg_group_config.get("segmentation file format", {}).get("selected").lower()
        
        phase_number = 46  
        phase_data = self.mapping_file.get("check_phase", {}).get(str(phase_number), {})
        error_descriptions = phase_data.get("errors", {})
        phase_name = phase_data.get("name", f"Phase {phase_number}")  
        
        # Format check and report names dynamically
        check_name = phase_name.lower().replace(" ", "_") 
        report_name = f"3.046.{check_name}_report"
        
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        check_file_path = os.path.join(self.output_directory_checkfile, "image_data_check_file.json")
        existing_warnings = retrieve_existing_warnings_image_data(check_file_path)

        report_content = ""
        num_errors = 0
        error_counts = {"E4601": None}
        overlap_detected = False
        overlapping_pairs = {}
        overlapping_files_map = {}
        overlapping_positions_map = {}
    
        patient_folders = [
            f for f in os.listdir(self.images_dir)
            if os.path.isdir(os.path.join(self.images_dir, f)) and f != "Reports"
        ]
    
        for patient_folder in patient_folders:
            patient_folder_path = os.path.join(self.images_dir, patient_folder)
    
            series_folder = os.path.normpath(series_group_subfolder).split(os.sep)[-1].strip()
            series_folder_path = os.path.join(patient_folder_path, series_folder)
            if not os.path.isdir(series_folder_path):
                continue
    
            # Find segmentation folder (assuming same logic as before)
            segmentation_folder = None
            for subfolder in os.listdir(series_folder_path):
                subfolder_path = os.path.join(series_folder_path, subfolder)
                if os.path.isdir(subfolder_path):
                    segmentation_folder = subfolder_path
                    break
    
            # Filter image files based on selected format
            image_files = [f for f in os.listdir(segmentation_folder) if f.lower().endswith(seg_file_format)] 
            if len(image_files) <= 1:
                continue  # No overlap possible
    
            processed_masks_single_series = {}
            for image_filename in image_files:
                image_file_path = os.path.join(segmentation_folder, image_filename)
                try:
                    img = Image.open(image_file_path).convert("L")
                    mask = (np.array(img) > 0).astype(np.int16)
                except Exception:
                    continue  # skip unreadable/corrupt images
    
                current_file_key = f"{patient_folder}/{series_folder}/{image_filename}"
    
                for other_key, other_mask in processed_masks_single_series.items():
                    pair = tuple(sorted([current_file_key, other_key]))
                    if pair in overlapping_pairs:
                        continue
    
                    overlap = (mask > 0) & (other_mask > 0)
                    if np.any(overlap):
                        overlap_detected = True
                        overlapping_files_map.setdefault(current_file_key, set()).add(other_key)
                        overlapping_files_map.setdefault(other_key, set()).add(current_file_key)
    
                        overlap_positions = np.argwhere(overlap)
                        overlapping_pairs[pair] = overlap_positions
    
                        for key in pair:
                            overlapping_positions_map.setdefault(key, []).extend([tuple(pos) for pos in overlap_positions])
    
                processed_masks_single_series[current_file_key] = mask
    
        files_with_overlap = set()
        for file1, file2 in overlapping_pairs.keys():
            files_with_overlap.add(file1)
            files_with_overlap.add(file2)
    
        num_errors = len(files_with_overlap)
        error_counts["E4601"] = num_errors if overlap_detected else None
    
        # Generate summary check file (same as your 3D function)
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
    
        report_filename = os.path.join(self.output_directory_report, f"{report_name}.txt")
        host_report_filename = os.path.join(self.host_output_directory_report, f"{report_name}.txt")
        if overlap_detected:
            report_content += f"Report generated on: {current_datetime}\n\n"
            report_content += f"{phase_name} report:\n\n"
            report_content += f"Error E4601: {error_descriptions.get('E4601', '')}\n"
            report_content += f"- Details:\n"
    
            reported_files = set()
            for file, overlaps in overlapping_files_map.items():
                if file in reported_files:
                    continue
    
                report_content += f"  - {file} overlaps with:\n"
                for other_file in sorted(overlaps):
                    report_content += f"  - {other_file}\n"
    
                if file in overlapping_positions_map:
                    unique_positions = sorted(set(overlapping_positions_map[file]))
                    report_content += f"\n    Overlapping Pixel Locations (Y, X):\n"
                    for y, x in unique_positions:
                        report_content += f"      - Y: {y}, X: {x}\n"
    
                report_content += "\n" + "-" * 80 + "\n"
                reported_files.add(file)
                reported_files.update(overlaps)
    
            report_content += f"\nTotal Errors: {num_errors}"
    
            with open(report_filename, "w") as f:
                f.write(report_content)
    
            raise Exception(
                f"Overlapping 2D segmentations found. "
                f"See the detailed report at: {host_report_filename}") #!! Path del container, deve essere quello locale.OK

        overlapping_pixel_message = "No overlapping pixels found between segmentation masks within the same series."
        return overlapping_pixel_message
    

    def generate_SegmentImageValidation_final_report(self, uniform_seg_message, segment_code_conformity_message, shape_inconsistency_message, overlapping_pixel_message): 
        """
        Generate a comprehensive final report for the SegmentImageValidator process.
        
        Parameters:
        - label_conformity_message: Message about label conformity check. 
        """
        
        # Get the current datetime
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")

        # Construct the final report
        final_report_lines = [
            f"Report generated on: {formatted_datetime}",
            "",
            "Final report on segmentation conformity checks:",
            "",
            "Uniform image check:",
            f"- {uniform_seg_message}",
            "",
            "Label conformity check:",
            f"- {segment_code_conformity_message}",
            "",
            "Shape consistency check:",
            f"- {shape_inconsistency_message}",
            "",
            "Overlapping pixel check:",
            f"- {overlapping_pixel_message}",
            "",
            "Summary:",
            "All checks completed."
        ]
    
        # Join all lines into a single string
        final_report = "\n".join(final_report_lines)
        
        # Define the report filename
        report_filename = os.path.join(self.output_directory_report, "3.SegmentImageValidation_final_report.txt")
        
        # Write the final report to a file
        with open(report_filename, "w") as report_file:
            report_file.write(final_report)
        
        # Print a message indicating that the final report has been generated
        print("3.SegmentImageValidation final report has been generated.")
    

def run_segimage_validator(protocol, local_config, mapping_file, series_group_name):
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
    
    segimage_validator = SegmentImageValidator(protocol, local_config, mapping_file, num_image_patients, num_slices_group, total_series, total_slices, series_group_name)

    series_progress_state =  load_state(segimage_validator.series_progress_file)

    # Access or create the per-series progress dictionary
    series_state = series_progress_state.setdefault(series_group_name, {})
    last_phase_done = series_state.get("last_successful_phase", 0)

    if last_phase_done < 43:
        print("Running check_uniform_seg function...") # check phase 43
        try:
            image_state["uniform_seg_message"], image_state["multiple_segmentation_flag"] = segimage_validator.check_uniform_seg()
            save_state(image_state, segimage_validator.state_file)

            # Update series progress state
            series_state["last_successful_phase"] = 43
            series_progress_state[series_group_name] = series_state
            save_state(series_progress_state, segimage_validator.series_progress_file)
        except Exception as e:
            log_error(input_dir, "check_uniform_seg", e, LOG_FILENAME)
            print(f"An unexpected error occurred during check_uniform_seg. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
            raise

    if last_phase_done < 44:
        print("Running validate_segment_code_conformity_2D function...") # check phase 44
        try:
            image_state["segment_code_conformity_message"] = segimage_validator.validate_segment_code_conformity_2D(image_state.get("multiple_segmentation_flag"))
            save_state(image_state, segimage_validator.state_file)

            # Update series progress state
            series_state["last_successful_phase"] = 44
            series_progress_state[series_group_name] = series_state
            save_state(series_progress_state, segimage_validator.series_progress_file)
        except Exception as e:
            log_error(input_dir, "validate_segment_code_conformity_2D", e, LOG_FILENAME)
            print(f"An unexpected error occurred during validate_segment_code_conformity_2D. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
            raise
    
    if last_phase_done < 45:
        print("Running single_label_seg_2Dshape_check function...") # check phase 45
        try:
            image_state["shape_inconsistency_message"] = segimage_validator.single_label_seg_2Dshape_check(image_state.get("multiple_segmentation_flag"))
            save_state(image_state, segimage_validator.state_file)

            # Update series progress state
            series_state["last_successful_phase"] = 45
            series_progress_state[series_group_name] = series_state
            save_state(series_progress_state, segimage_validator.series_progress_file)
        except Exception as e:
            log_error(input_dir, "single_label_seg_2Dshape_check", e, LOG_FILENAME)
            print(f"An unexpected error occurred during single_label_seg_2Dshape_check. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
            raise

    if last_phase_done < 46:
        print("Running check_pixel_overlap_single_label_seg_2D function...") # check phase 46
        try:
            image_state["overlapping_pixel_message"] = segimage_validator.check_pixel_overlap_single_label_seg_2D(image_state.get("multiple_segmentation_flag"))
            save_state(image_state, segimage_validator.state_file)

            # Update series progress state
            series_state["last_successful_phase"] = 46
            series_progress_state[series_group_name] = series_state
            save_state(series_progress_state, segimage_validator.series_progress_file)
        except Exception as e:
            log_error(input_dir, "check_pixel_overlap_single_label_seg_2D", e, LOG_FILENAME)
            print(f"An unexpected error occurred during check_pixel_overlap_single_label_seg_2D. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
            raise

        print("Running generate_SegmentImageValidation_final_report function...")
        try:
            segimage_validator.generate_SegmentImageValidation_final_report(
                    image_state.get("uniform_seg_message"), 
                    image_state.get("segment_code_conformity_message"), 
                    image_state.get("shape_inconsistency_message"), 
                    image_state.get("overlapping_pixel_message")
                )
        except Exception as e:
            log_error(input_dir, "generate_SegmentImageValidation_final_report", e, LOG_FILENAME)
            print(f"An unexpected error occurred during generate_SegmentImageValidation_final_report. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
            raise
        



    


