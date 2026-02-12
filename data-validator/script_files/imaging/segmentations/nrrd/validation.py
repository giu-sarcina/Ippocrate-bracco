import numpy as np
import os 
from datetime import datetime
import json
import nibabel as nib
import nrrd
from utils import save_state, load_state, clear_log_file, log_error
from ...helpers import (
    generate_check_file_image_data,
    retrieve_existing_warnings_image_data
)

LOG_FILENAME = "nrrdseg_validation_error.log"

class NrrdSegValidator:

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
        self.host_input_dir = os.getenv("HOST_BASE_DATA_DIR")
        self.images_dir = os.path.join(self.input_dir, "IMAGES")
        self.state_file = os.path.join(self.input_dir, "image_validation_state.json")
        self.series_progress_file = os.path.join(self.input_dir, "validation_progress_by_series.json")
        self.output_directory_report = os.path.join(self.input_dir, "Reports", f"Image Data Reports {self.series_group_name}")
        self.host_output_directory_report = os.path.join(self.host_input_dir, "Reports", f"Image Data Reports {self.series_group_name}")
        self.output_directory_checkfile = os.path.join(self.study_path, "CHECKFILE")
        self.output_directory_data = os.path.join(self.study_path, "IMAGES")
        os.makedirs(self.output_directory_report, exist_ok=True)  
        os.makedirs(self.output_directory_checkfile, exist_ok=True)
        os.makedirs(self.output_directory_data, exist_ok=True)


    def check_num_slices_nrrd_seg(self): #OK#41
        """
        Checks if the number of segmentation slices in each NRRD file in the segmentation folder(s)
        matches the number of volume slices in the corresponding series folder.
        If there are multiple NRRD segmentation files for a series, it processes each one.
        Also sets a flag (multiple_segmentation_flag) to True if the protocol specifies single-mask segmentation with more than one segment.
        Generates a JSON check file, and a text report if discrepancies are found.
        """
        phase_number = 41  
        phase_data = self.mapping_file.get("check_phase", {}).get(str(phase_number), {})
        error_descriptions = phase_data.get("errors", {})
        phase_name = phase_data.get("name", f"Phase {phase_number}")  
        
        # Format check and report names dynamically
        check_name = phase_name.lower().replace(" ", "_") 
        report_name = f"3.041.{check_name}_report"
        
        # Generate formatted date-time strings
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        series_group = self.series_group_name
        series_group_mapping = self.local_config["Local config"]["radiological data"]["series_mapping"]
        series_group_subfolder = series_group_mapping.get(series_group) #input name of the series
        image_group_config = self.protocol.get(series_group, {}).get("image", {}) #original self.protocol["image"]
        seg_group_config = self.protocol.get(series_group, {}).get("segmentation", {}) #original self.protocol["segmentation"]

        seg_input_format = seg_group_config["segmentation_input_format"]["selected"]
        segments_number = seg_group_config["segments_number"]

        # Retrieve existing warnings
        check_file_path = os.path.join(self.output_directory_checkfile, "image_data_check_file.json")
        existing_warnings = retrieve_existing_warnings_image_data(check_file_path)
        multiple_segmentation_flag = seg_input_format == "single-mask" and segments_number > 1

        report_content = ""
        discrepancy_info = ""
        num_errors = 0
        has_error = False
        error_counts = {"E4101": None}

        patient_folders = [f for f in os.listdir(self.images_dir) if os.path.isdir(os.path.join(self.images_dir, f)) and f != "Reports"]
        
        # Iterate through all patient folders
        for patient_folder in patient_folders:
            patient_folder_path = os.path.join(self.images_dir, patient_folder)
            if os.path.isdir(patient_folder_path):
                series_folder = os.path.normpath(series_group_subfolder).split(os.sep)[-1].strip()
                series_folder_path = os.path.join(patient_folder_path, series_folder)
                
                if os.path.isdir(series_folder_path):

                    if image_group_config["image file format"]["selected"] == ".dcm": 
                        # Find DICOM files in the series folder
                        dicom_files = [f for f in os.listdir(series_folder_path) if f.endswith('.dcm')]
                        num_slices = len(dicom_files)     
                    elif image_group_config["image file format"]["selected"] in [".nii", ".nii.gz"]:
                        # Find NIfTI image files in the series folder for slice count comparison
                        nifti_files = [f for f in os.listdir(series_folder_path) if f.endswith('.nii') or f.endswith('.nii.gz')] # output = list 
                        nifti_image_path = os.path.join(series_folder_path, nifti_files[0])
                        nifti_img = nib.load(nifti_image_path)
                        nifti_data = nifti_img.get_fdata()
                        num_slices = nifti_data.shape[2]  # Use the third dimension as the slice count for images        
                    elif image_group_config["image file format"]["selected"] == ".nrrd":
                        # Handle NRRD files
                        nrrd_files_volume = [f for f in os.listdir(series_folder_path) if f.endswith('.nrrd')]
                        nrrd_image_path = os.path.join(series_folder_path, nrrd_files_volume[0])
                        nrrd_data, _ = nrrd.read(nrrd_image_path)
                        num_slices = nrrd_data.shape[2]  # Use the third dimension as the slice count
                    else:
                        raise ValueError(f"Invalid image file format specified in protocol: {image_group_config['image file format']['selected']}.")

                    # Find the unique folder in the series folder (which should contain the NRRD file)
                    segmentation_folder = None
                    for subfolder in os.listdir(series_folder_path):
                        subfolder_path = os.path.join(series_folder_path, subfolder)
                        if os.path.isdir(subfolder_path):
                            segmentation_folder = subfolder_path
                            break  # We only need to select the first (and only) subfolder containing the NNRD file
                            
                    # CONSIDERA LA POSSIBILITA' DI AVERE PIU' FILE DI SEGMENTAZIONE
                    # Assuming there is exactly one NRRD file in the segmentation folder
                    nrrd_files  = [f for f in os.listdir(segmentation_folder) if f.endswith('.nrrd')]

                    for nrrd_filename in nrrd_files:
                        nrrd_file_path = os.path.join(segmentation_folder, nrrd_filename)
                        
                        # Load the NRRD file 
                        nrrd_data, nrrd_header = nrrd.read(nrrd_file_path)
                        nrrd_num_slices = nrrd_data.shape[2]
                        
                        # Check that the number of slices matches between NRRD and DICOM
                        if nrrd_num_slices != num_slices:
                            discrepancy_info += f"  - Number of slices mismatch for patient {patient_folder}, series {series_folder}, segmentation file {nrrd_filename}: segmentations ({nrrd_num_slices}) vs volumes ({num_slices})\n"
                            num_errors += 1
                            has_error = True
                            
                            
        error_counts = {"E4101": num_errors if num_errors > 0 else None}
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

        if has_error:
            report_name = f"{report_name}.txt"
            report_filename = os.path.join(self.output_directory_report, report_name)
            host_report_filename = os.path.join(self.host_output_directory_report, report_name)

            report_content += f"Report generated on: {current_datetime}\n\n"
            report_content += f"{phase_name} report:\n\n"
            report_content += f"Error E4101: {error_descriptions.get('E4101', None)}\n"
            report_content += f"- Details:\n"
            report_content += discrepancy_info
            report_content += f"\nTotal Errors: {num_errors}"
            with open(report_filename, "w") as report_file:
                report_file.write(report_content)
            raise Exception(
                f"Discrepancies found in segmentation vs. volume slice count for one or more segments. "
                f"See the detailed report at: {host_report_filename}") #!! Path del container, deve essere quello locale.ok
        else:
            success_size_nrrdseg = "All segmentation files match the slice count of their corresponding volumes."
            return success_size_nrrdseg, multiple_segmentation_flag
        

    def validate_segment_code_conformity(self, multiple_seg_file_flag): 
        """
        Validates segmentation conformity for multi-mask and single-mask segmentations.
        If single-mask with multiple segmentation files, checks that the number of files matches segments_number
        and that filenames correspond to segments_labels.
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
        expected_codes = seg_group_config["segmentation_code"]
        segments_number = seg_group_config["segments_number"]

        segments_labels = seg_group_config["segments_labels"]
        
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        check_file_path = os.path.join(self.output_directory_checkfile, "image_data_check_file.json")
        existing_warnings = retrieve_existing_warnings_image_data(check_file_path)
    
        # Initializations   
        num_errors = 0
        has_error = False
        code_errors_content = ""
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
                    
                    # Find the unique folder in the series folder (which should contain the NRRD file)
                    segmentation_folder = None
                    for subfolder in os.listdir(series_folder_path):
                        subfolder_path = os.path.join(series_folder_path, subfolder)
                        if os.path.isdir(subfolder_path):
                            segmentation_folder = subfolder_path
                            break  # Select the first (and only) subfolder containing the NRRD file

                    seg_files = [f for f in os.listdir(segmentation_folder) if f.endswith(seg_file_format)]
                            
                    # CONSIDERA LA POSSIBILITA' DI AVERE PIU' FILE DI SEGMENTAZIONE 
                    if seg_input_format == "single-mask":

                        # Check if number of files matches segments_number
                        if len(seg_files) != segments_number:
                            count_errors_content += (f"  - Incorrect number of segmentation files ({len(seg_files)}) "
                                                     f"in folder {segmentation_folder}.\n")
                            num_errors += 1
                            has_error = True

                        if multiple_seg_file_flag:
                            # Create a mapping of original filenames to their normalized versions (to add flexibility)
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
                                extra_files = normalized_actual_filenames - expected_filenames

                                if extra_files:
                                    # Retrieve original filenames corresponding to the mismatched normalized names
                                    original_extra_files = {orig for orig, norm in filename_mapping.items() if norm in extra_files}
                                    label_errors_content += (f"  - Incorrect segmentation filenames in {segmentation_folder}. "
                                                             f"Mismatched filenames: {original_extra_files}.\n")
                                    num_errors += len(original_extra_files) 
                                    has_error = True
                                    
                    if (seg_input_format == "multi-mask") or (seg_input_format == "single-mask" and not multiple_seg_file_flag):            
                        # Load the NRRD file
                        nrrd_files = [f for f in os.listdir(segmentation_folder) if f.endswith('.nrrd')]
                        nrrd_file_path = os.path.join(segmentation_folder, nrrd_files[0])
                        nrrd_data, _ = nrrd.read(nrrd_file_path)
    
                        # Check unique values in the NRRD data
                        unique_values = np.unique(nrrd_data)
                        filtered_unique_values = unique_values[unique_values != 0]
                        
                        # Verify the unique values match the expected codes
                        unexpected_values = [val for val in unique_values if val not in expected_codes]
                        if unexpected_values:
                            code_errors_content += (f"  - Unexpected segmentation codes in NRRD for patient {patient_folder}, "
                                                     f"series {series_folder}. Unexpected codes: {unexpected_values}.\n")
                            num_errors += len(unexpected_values)
                            has_error = True
                            
                    if seg_input_format == "multi-mask":
                        # Verify the number of unique values matches segments_number
                        if len(filtered_unique_values) != segments_number:
                            count_errors_content += (f"  - Number of unique values ({len(filtered_unique_values)}) in NRRD segmentation array does not match protocol segments_number ({segments_number}) for patient {patient_folder}, series {series_folder}.\n")
                            num_errors += 1
                            has_error = True
                    
                        if len(seg_files) != 1:  # If there's not exactly one segmentation file
                            multi_mask_file_error_content += (f"  - Incorrect number of segmentation files ({len(seg_files)}) "
                                                     f"in folder {segmentation_folder}. Expected exactly 1 file.\n")
                            num_errors += 1
                            has_error = True
    
        # Assemble the report content with separate sections
        if code_errors_content:
            code_errors_content_title = f"  Expected segmentation codes: {expected_codes}\n"
            discrepancy_info += "  --- Segmentation Code Errors ---\n\n" + code_errors_content_title + code_errors_content + "\n"
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
                f"See the detailed report at: {host_report_filename}") #!! Path del container, deve essere quello locale.ok
        else:
            if seg_input_format == "single-mask" and multiple_seg_file_flag:
                segment_code_conformity_message = (
                    "Segmentation validation successful: all segmentation files conform to the expected labels and count."
                )
            else:
                segment_code_conformity_message = (
                    "Segmentation validation successful: segmentation codes and count conform to the protocol."
                )
            
            return segment_code_conformity_message
        

    def single_label_seg_3Dshape_check(self, multiple_segmentation_flag): #45
        """
        Checks for inconsistent shapes among multi-dimensional (NRRD) segmentations in the input directory structure.
        Only performs the check if the segmentation input format is 'single-mask' and the number of segments is > 1.
        
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
        seg_input_format = seg_group_config.get("segmentation_input_format", {}).get("selected")
        segments_number = seg_group_config.get("segments_number")
        
        shape_inconsistencies = {}
        inconsistent_shape_detected = False

        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        check_file_path = os.path.join(self.output_directory_checkfile, "image_data_check_file.json")
        existing_warnings = retrieve_existing_warnings_image_data(check_file_path)

        patient_folders = [
            f for f in os.listdir(self.images_dir)
            if os.path.isdir(os.path.join(self.images_dir, f)) and f != "Reports"
        ]
        
        # Iterate over patients and series
        for patient_folder in patient_folders:
            patient_folder_path = os.path.join(self.images_dir, patient_folder)
            series_folder = os.path.normpath(series_group_subfolder).split(os.sep)[-1].strip()
            series_folder_path = os.path.join(patient_folder_path, series_folder)
    
            if os.path.isdir(series_folder_path):
                # Find the unique folder in the series folder (which should contain the NRRD file)
                segmentation_folder = None
                for subfolder in os.listdir(series_folder_path):
                    subfolder_path = os.path.join(series_folder_path, subfolder)
                    if os.path.isdir(subfolder_path):
                        segmentation_folder = subfolder_path
                        break  # Select the first (and only) subfolder containing the NRRD file
                        
                nrrd_files = [f for f in os.listdir(segmentation_folder)
                           if f.endswith('.nrrd')]
                
                reference_shape = None
                for nrrd_filename in nrrd_files:
                    nrrd_file_path = os.path.join(segmentation_folder, nrrd_filename)
                    try:
                        seg_array, _ = nrrd.read(nrrd_file_path)
                    except Exception as e:
                        print(f"Failed to read NRRD file {nrrd_filename}: {e}")
                        continue

                    # Check for shape consistency (3D or 4D)
                    if reference_shape is None:
                        reference_shape = seg_array.shape  # First shape encountered becomes the reference
                    elif seg_array.shape != reference_shape:
                        inconsistent_shape_detected = True
                        key = (patient_folder, series_folder)
                        if key not in shape_inconsistencies:
                            shape_inconsistencies[key] = []
                        shape_inconsistencies[key].append((nrrd_filename, seg_array.shape))

        num_errors = len(shape_inconsistencies)
        error_counts = {"E4501": num_errors if num_errors > 0 else None}
        # Generate check file
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
        # If inconsistencies are found, generate a report and raise an error
        if inconsistent_shape_detected: 
            report_filename = os.path.join(self.output_directory_report, f"{report_name}.txt")
            host_report_filename = os.path.join(self.host_output_directory_report, f"{report_name}.txt")

            # Generate report content
            report_content = f"Report generated on: {current_datetime}\n\n"
            report_content += f"{phase_name} report:\n\n"
            report_content += f"Error E4501: {error_descriptions.get('E4501', None)}\n"
            report_content += "- Details: single-label segmentations with inconsistent shapes were detected in:\n"
            for (patient_folder, series_folder), _ in shape_inconsistencies.items():
                report_content += f"  - Patient: {patient_folder}, Series: {series_folder}\n"
            report_content += f"\nTotal Errors: {num_errors}"
    
            # Save the report
            with open(report_filename, "w") as report_file:
                report_file.write(report_content)

            # Raise an error to halt the process
            raise Exception(
                f"Shape inconsistency detected in single-label segmentations within at least one series. "
                f"See the detailed report at: {host_report_filename}") #!! Path del container, deve essere quello locale
        else:
            shape_inconsistency_message = "All single-label segmentations within each series have consistent shapes."
            return shape_inconsistency_message
        

    def check_pixel_overlap_single_label_seg(self, multiple_segmentation_flag):
        """
        Checks for overlapping pixels between segmentation masks in single-label segmentations.
    
        If overlaps are found, generates a detailed report and raises an exception.
        Skips the check if segmentations are not single-mask with multiple segments.
    
        Args:
            multiple_segmentation_flag (bool): Whether to perform the overlap check.
    
        Returns:
            str: Status message indicating if overlaps were found or check skipped.
    
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

            segmentation_folder = None
            for subfolder in os.listdir(series_folder_path):
                subfolder_path = os.path.join(series_folder_path, subfolder)
                if os.path.isdir(subfolder_path):
                    segmentation_folder = subfolder_path
                    break
            if segmentation_folder is None:
                continue

            nrrd_files = [f for f in os.listdir(segmentation_folder) if f.endswith(".nrrd")]
            if len(nrrd_files) <= 1:
                continue  # No overlap possible

            processed_masks_single_series = {}
            for nrrd_filename in nrrd_files:
                nrrd_file_path = os.path.join(segmentation_folder, nrrd_filename)
                try:
                    seg_array, _ = nrrd.read(nrrd_file_path)
                except Exception as e:
                    print(f"Failed to read NRRD file {nrrd_filename}: {e}")
                    continue
                    
                seg_mask = (seg_array > 0).astype(np.int16)

                current_file_key = f"{patient_folder}/{series_folder}/{nrrd_filename}"

                for other_key, other_mask in processed_masks_single_series.items():
                    # Ensure unique pair (sorted)
                    pair = tuple(sorted([current_file_key, other_key]))
                    if pair in overlapping_pairs:
                        continue
                        
                    overlap = (seg_mask > 0) & (other_mask > 0)
                    if np.any(overlap):
                        overlap_detected = True

                        # Log overlapping file names
                        overlapping_files_map.setdefault(current_file_key, set()).add(other_key)
                        overlapping_files_map.setdefault(other_key, set()).add(current_file_key)

                        # Log overlapping pixel positions
                        overlap_positions = np.argwhere(overlap)
                        overlapping_pairs[pair] = overlap_positions

                        for key in pair:
                            overlapping_positions_map.setdefault(key, []).extend([tuple(pos) for pos in overlap_positions])

                processed_masks_single_series[current_file_key] = seg_mask
    
        files_with_overlap = set()
        for file1, file2 in overlapping_pairs.keys():
            files_with_overlap.add(file1)
            files_with_overlap.add(file2)
        
        num_errors = len(files_with_overlap)
        error_counts["E4601"] = num_errors if overlap_detected else None
    
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
                    continue  # Skip already reported files
            
                report_content += f"  - {file} overlaps with:\n"
                for other_file in sorted(overlaps):
                    report_content += f"  - {other_file}\n"
            
                if file in overlapping_positions_map:
                    unique_positions = sorted(set(overlapping_positions_map[file]))
                    report_content += f"\n    Overlapping Pixel Locations (Slice, Y, X):\n"
                    for z, y, x in unique_positions:
                        report_content += f"      - Slice: {z}, X: {x}, Y: {y}\n"
            
                report_content += "\n" + "-" * 80 + "\n"
            
                # Mark all these files as reported to avoid repetition
                reported_files.add(file)
                reported_files.update(overlaps)
        
            report_content += f"\nTotal Errors: {num_errors}"
            
            with open(report_filename, "w") as f:
                f.write(report_content)
    
            raise Exception(
                f"Overlapping segmentations found in DICOM SEG. "
                f"See the detailed report at: {host_report_filename}") #!! Path del container, deve essere quello locale.ok

        overlapping_pixel_message = "No overlapping pixels found between segmentation masks within the same series."
        return overlapping_pixel_message
    

    def generate_NrrdSegValidation_final_report(self, success_size_nrrdseg, segment_code_conformity_message, shape_inconsistency_message, overlapping_pixel_message):
        # Get the current datetime
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")

        # Construct the final report
        final_report_lines = [
            f"Report generated on: {formatted_datetime}",
            "",
            "Final report on NRRD segmentation conformity checks:",
            "",
            "Size check:",
            f"- {success_size_nrrdseg}",
            "",
            "Segment label and code conformity check:",
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
        report_filename = os.path.join(self.output_directory_report, "3.NrrdSegValidation_final_report.txt")
        
        # Write the final report to a file
        with open(report_filename, "w") as report_file:
            report_file.write(final_report)
        
        # Print a message indicating that the final report has been generated
        print("3.NrrdSegValidation final report has been generated.")


def run_nrrdseg_validator(protocol, local_config, mapping_file, series_group_name):
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
    
    nrrdseg_validator = NrrdSegValidator(protocol, local_config, mapping_file, num_image_patients, num_slices_group, total_series, total_slices, series_group_name)

    series_progress_state =  load_state(nrrdseg_validator.series_progress_file)

    # Access or create the per-series progress dictionary
    series_state = series_progress_state.setdefault(series_group_name, {})
    last_phase_done = series_state.get("last_successful_phase", 0)

    if last_phase_done < 41:
        print("Running check_num_slices_nrrd_seg function...") # check phase 41
        try:
            image_state["success_size_nrrdseg"], image_state["multiple_segmentation_flag"] = nrrdseg_validator.check_num_slices_nrrd_seg()
            save_state(image_state, nrrdseg_validator.state_file)  # Save updated state

            # Update series progress state
            series_state["last_successful_phase"] = 41
            series_progress_state[series_group_name] = series_state
            save_state(series_progress_state, nrrdseg_validator.series_progress_file)
        except Exception as e:
            log_error(input_dir, "check_num_slices_nrrd_seg", e, LOG_FILENAME)
            print(f"An unexpected error occurred during check_num_slices_nrrd_seg. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
            raise

        # NOTE: Phases 42 and 43 intentionally skipped (not applicable to NRRD)

    if last_phase_done < 44:
        print("Running validate_segment_code_conformity function...") # check phase 44
        try: 
            image_state["segment_code_conformity_message"] = nrrdseg_validator.validate_segment_code_conformity(image_state.get("multiple_segmentation_flag"))
            save_state(image_state, nrrdseg_validator.state_file)

            # Update series progress state
            series_state["last_successful_phase"] = 44
            series_progress_state[series_group_name] = series_state
            save_state(series_progress_state, nrrdseg_validator.series_progress_file)
        except Exception as e:
            log_error(input_dir, "validate_segment_code_conformity", e, LOG_FILENAME)
            print(f"An unexpected error occurred during validate_segment_code_conformity. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
            raise
    
    if last_phase_done < 45:
        print("Running single_label_seg_3Dshape_check function...") # check phase 45
        try:
            image_state["shape_inconsistency_message"] = nrrdseg_validator.single_label_seg_3Dshape_check(image_state.get("multiple_segmentation_flag"))
            save_state(image_state, nrrdseg_validator.state_file)

            # Update series progress state
            series_state["last_successful_phase"] = 45
            series_progress_state[series_group_name] = series_state
            save_state(series_progress_state, nrrdseg_validator.series_progress_file)
        except Exception as e:
            log_error(input_dir, "single_label_seg_3Dshape_check", e, LOG_FILENAME)
            print(f"An unexpected error occurred during single_label_seg_3Dshape_check. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
            raise

    if last_phase_done < 46:
        print("Running check_pixel_overlap_single_label_seg function...") # check phase 46
        try:
            image_state["overlapping_pixel_message"] = nrrdseg_validator.check_pixel_overlap_single_label_seg(image_state.get("multiple_segmentation_flag"))
            save_state(image_state, nrrdseg_validator.state_file)

            # Update series progress state
            series_state["last_successful_phase"] = 46
            series_progress_state[series_group_name] = series_state
            save_state(series_progress_state, nrrdseg_validator.series_progress_file)
        except Exception as e:
            log_error(input_dir, "check_pixel_overlap_single_label_seg", e, LOG_FILENAME)
            print(f"An unexpected error occurred during check_pixel_overlap_single_label_seg. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
            raise

        print("Running generate_NrrdSegValidation_final_report function...")
        try:
            nrrdseg_validator.generate_NrrdSegValidation_final_report(
                    image_state.get("success_size_nrrdseg"), 
                    image_state.get("segment_code_conformity_message"), 
                    image_state.get("shape_inconsistency_message"), 
                    image_state.get("overlapping_pixel_message")
                )
        except Exception as e:
            log_error(input_dir, "generate_NrrdSegValidation_final_report", e, LOG_FILENAME)
            print(f"An unexpected error occurred during generate_NrrdSegValidation_final_report. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
            raise

    


