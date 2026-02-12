import os 
import pydicom
import numpy as np
from datetime import datetime
import json
import SimpleITK as sitk
from utils import save_state, load_state, clear_log_file, log_error
from ...helpers import (
    generate_check_file_image_data,
    count_tot_num_slices_per_group,
    retrieve_existing_warnings_image_data,
    extract_metadata_from_check_file,
    extract_tag_names_and_values,
    classify_slice_orientation
)

LOG_FILENAME = "dicom_validation_error.log"

class DicomValidator:
    
    def __init__(self, protocol, local_config, tag_list, mapping_file, num_image_patients, series_group_name):
        self.protocol = protocol
        self.local_config = local_config
        self.tag_list = tag_list
        self.mapping_file = mapping_file
        self.series_group_name = series_group_name
        self.num_image_patients = num_image_patients
        self.input_dir = os.getenv("INPUT_DIR")
        self.images_dir = os.path.join(self.input_dir, "IMAGES")
        self.host_input_dir = os.getenv("HOST_BASE_DATA_DIR")
        self.state_file = os.path.join(self.input_dir, "image_validation_state.json")
        self.series_progress_file = os.path.join(self.input_dir, "validation_progress_by_series.json")
        self.study_path = os.getenv("ROOT_NAME")
        self.output_directory_report = os.path.join(self.input_dir, "Reports", f"Image Data Reports {self.series_group_name}")
        self.host_output_directory_report = os.path.join(self.host_input_dir, "Reports", f"Image Data Reports {self.series_group_name}")
        self.output_directory_checkfile = os.path.join(self.study_path, "CHECKFILE")
        os.makedirs(self.output_directory_report, exist_ok=True)  
        os.makedirs(self.output_directory_checkfile, exist_ok=True)

    
    def check_anonymization(self): #2D e 3D

        phase_number = 30  
        phase_data = self.mapping_file.get("check_phase", {}).get(str(phase_number), {})
        error_descriptions = phase_data.get("errors", {})
        phase_name = phase_data.get("name", f"Phase {phase_number}")  
        
        # Format check and report names dynamically
        check_name = phase_name.lower().replace(" ", "_") 
        report_name = f"1.030.{check_name}_report"

        check_file_path = os.path.join(self.output_directory_checkfile, "image_data_check_file.json")
        existing_warnings = retrieve_existing_warnings_image_data(check_file_path)
        _, num_slices_group_old, total_series, total_slices = extract_metadata_from_check_file(self.output_directory_checkfile)

        series_group = self.series_group_name
        series_group_mapping = self.local_config["Local config"]["radiological data"]["series_mapping"]
        series_group_subfolder = series_group_mapping.get(series_group) #input name of the series
        image_group_config = self.protocol.get(series_group, {}).get("image", {}) #original self.protocol["image"]
        image_file_format = image_group_config['image file format']['selected'].lower()

        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report_content  = ""
        non_anonymous_tags_found = False
        non_anonymized_tags_info = {}
        num_non_anonymous_files = 0
        success_anonymization = ""
        total_dicom_files_found = 0

        num_slices = 0
        num_slices_group = 0

        #series_has_non_anonymous_tags = False
        series_with_non_anonymous_files = set()
        error_counts = {"E3001": 0}
        
        patient_folders = [f for f in os.listdir(self.images_dir) if os.path.isdir(os.path.join(self.images_dir, f)) and f != "Reports"]
        
        # Iterate through all patient folders
        for patient_folder in patient_folders:
            patient_folder_path = os.path.join(self.images_dir, patient_folder)
            if os.path.isdir(patient_folder_path):
                
                series_has_non_anonymous_tags = False
                
                series_folder = os.path.normpath(series_group_subfolder).split(os.sep)[-1].strip()
                series_folder_path = os.path.join(patient_folder_path, series_folder)
                
                num_slices = count_tot_num_slices_per_group(series_folder_path, image_file_format)
                num_slices_group += num_slices  # Update total slices
                
                for filename in os.listdir(series_folder_path):
                    if filename.endswith(".dcm"):
                        total_dicom_files_found += 1
                        file_path = os.path.join(series_folder_path, filename)
                        ds = pydicom.dcmread(file_path)

                        tag_names_and_values = extract_tag_names_and_values(ds, self.tag_list)
                                
                        if any(tag_value not in ('', None) for tag_value in tag_names_and_values.values()): 
                            non_anonymous_tags_found = True
                            series_has_non_anonymous_tags = True
                            non_anonymized_tags_info[(patient_folder, series_folder, filename)] = tag_names_and_values
                            num_non_anonymous_files += 1
                            
                        # Check patient identity
                        patient_identity = ds.get((0x0012, 0x0062), None)
                        if patient_identity is not None:
                            if patient_identity.value != 'YES':
                                non_anonymous_tags_found = True
                                series_has_non_anonymous_tags = True
                                patient_identity_value = patient_identity.value
                                tag_names_and_values["Patient Identity Removed"] = patient_identity_value

                if series_has_non_anonymous_tags:
                    series_with_non_anonymous_files.add((patient_folder, series_folder))

        total_slices = total_slices - num_slices_group_old + num_slices_group
        
        if total_dicom_files_found == 0:            
           raise FileNotFoundError("No DICOM files were found in any patient folders. Please check your input directory structure.")         

        num_non_anonymous_series = len(series_with_non_anonymous_files)
        error_counts["E3001"] = num_non_anonymous_files if num_non_anonymous_files else None
            
        if non_anonymous_tags_found:
            report_name = f"{report_name}.txt"
            report_content += f"Report generated on: {current_datetime}\n\n"
            report_content += f"{phase_name} report:\n\n"
            report_content += f"Error E3001: {error_descriptions.get('E3001', None)}\n"
            report_content += "- Details: The following files have non-anonymized tags:\n"
            
            for (patient_name, series_name, filename), tag_names_and_values in non_anonymized_tags_info.items():
                non_anonymous_tags = {key: value for key, value in tag_names_and_values.items() if key != "Patient Identity Removed" and value != ''}
                report_content += f"  - Patient: {patient_name}, Series: {series_name}, File: {filename}, Non-anonymous tags: {non_anonymous_tags}"

                # Check if "Patient Identity Removed" is present in the tags for this file
                if "Patient Identity Removed" in tag_names_and_values:
                    report_content += f"  - 'Patient Identity Removed': {tag_names_and_values['Patient Identity Removed']}\n"
                else:
                    report_content += "\n"  # Add a newline if "Patient Identity Removed" is not present
                    
                # Add a blank line after each file
                #report_content += "\n"
                            
            report_content += f"\nNumber of series with non-anonymous tags: {num_non_anonymous_series}"
            report_content += f"\nNumber of slices with non-anonymous tags: {num_non_anonymous_files}\n\n"
            
            # Generate check file with errors
            generate_check_file_image_data(
                check_name=phase_name,
                phase_number=phase_number,
                series_group_name=series_group,
                num_slices_group=num_slices_group,
                num_patients_with_image_data=len(patient_folders),
                num_series=total_series,
                num_tot_slices=total_slices,
                timestamp=current_datetime,
                error_counts=error_counts,
                warning_counts={},
                error_descriptions=error_descriptions,
                warning_descriptions={},
                output_dir=self.output_directory_checkfile,
                existing_warnings=existing_warnings
            )
            # Write the report content to the file
            non_anonymized_tags_report_path = os.path.join(self.output_directory_report, report_name)
            host_non_anonymized_tags_report_path = os.path.join(self.host_output_directory_report, report_name)
            with open(non_anonymized_tags_report_path, "w") as report_file:
                report_file.write(report_content)

            raise ValueError(
                f"Anonymization check failed: one or more DICOM files contain non-anonymized tags. "
                f"See the detailed report at: {host_non_anonymized_tags_report_path}" #!! Path del container, deve essere quello locale.OK
            )
        else:
            # Generate check file with errors
            generate_check_file_image_data(
                check_name=phase_name,
                phase_number=phase_number,
                series_group_name=series_group,
                num_slices_group=num_slices_group,
                num_patients_with_image_data=len(patient_folders),
                num_series=total_series,
                num_tot_slices=total_slices,
                timestamp=current_datetime,
                error_counts=error_counts,
                warning_counts={},
                error_descriptions=error_descriptions,
                warning_descriptions={},
                output_dir=self.output_directory_checkfile,
                existing_warnings=existing_warnings
            )

            success_anonymization = "All specified tags have been anonymized in all DICOM files."
            return success_anonymization
        

    def check_image_type(self): #2D e 3D

        phase_number = 31  
        phase_data = self.mapping_file.get("check_phase", {}).get(str(phase_number), {})
        error_descriptions = phase_data.get("errors", {})
        phase_name = phase_data.get("name", f"Phase {phase_number}")  
        
        # Format check and report names dynamically
        check_name = phase_name.lower().replace(" ", "_") 
        report_name = f"1.031.{check_name}_report"

        check_file_path = os.path.join(self.output_directory_checkfile, "image_data_check_file.json")
        existing_warnings = retrieve_existing_warnings_image_data(check_file_path)
        _, num_slices_group_old, total_series, total_slices = extract_metadata_from_check_file(self.output_directory_checkfile)
        
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        series_group = self.series_group_name
        series_group_mapping = self.local_config["Local config"]["radiological data"]["series_mapping"]
        series_group_subfolder = series_group_mapping.get(series_group) #input name of the series
        image_group_config = self.protocol.get(series_group, {}).get("image", {}) #original self.protocol["image"]

        # Image and segmentation format validations
        expected_modality = image_group_config["type"]["selected"]
        image_format_selected = image_group_config['image file format']['selected']

        # Normalize the expected modality
        if expected_modality == "MRI":
            expected_modality = "MR"
        else:
            expected_modality = expected_modality
 
        mismatched_files = []
        error_counts = {"E3101": None}
        success_message_image_type = ""
        total_dicom_files_found = 0
        num_slices = 0
        num_slices_group = 0
        
        patient_folders = [f for f in os.listdir(self.images_dir) if os.path.isdir(os.path.join(self.images_dir, f)) and f != "Reports"]
        
        # Iterate through patient folders in the input folder
        for patient_folder_name in patient_folders:
            patient_folder_path = os.path.join(self.images_dir, patient_folder_name)
            if os.path.isdir(patient_folder_path):
                
                series_folder = os.path.normpath(series_group_subfolder).split(os.sep)[-1].strip()
                series_path = os.path.join(patient_folder_path, series_folder)

                num_slices = count_tot_num_slices_per_group(series_path, image_format_selected)
                num_slices_group += num_slices  # Update total slices
                
                # Iterate through DICOM files in the series folder
                for filename in os.listdir(series_path):
                    if filename.endswith(".dcm"):
                        total_dicom_files_found += 1
                        file_path = os.path.join(series_path, filename)
                        ds = pydicom.dcmread(file_path)
                        actual_modality = ds.get((0x0008, 0x0060), None)
                        
                        if actual_modality is None:
                            continue  # Skip if Modality tag is not present
                        
                        actual_modality = actual_modality.value
                        
                        if actual_modality != expected_modality: ##oppure: if expected_modality not in actual_modality:
                            relative_file_path = os.path.relpath(file_path, self.images_dir)
                            mismatched_files.append({
                                "file_path": relative_file_path,
                                "expected_modality": expected_modality,
                                "actual_modality": actual_modality
                            })
                            
        total_slices = total_slices - num_slices_group_old + num_slices_group
        
        if total_dicom_files_found == 0:
            raise FileNotFoundError("No DICOM files were found in any patient folders. Please check your input directory structure.")
            
        if mismatched_files:
            error_counts["E3101"] = len(mismatched_files)
            
            # Generate the report
            report_content = f"Report generated on: {current_datetime}\n\n"
            report_content += f"{phase_name} report:\n\n"
            report_content += f"Error E3101: {error_descriptions.get('E3101', None)}\n"
            report_content += f"- Details: The following files have mismatched modalities (Expected modality: {expected_modality}):\n"
            
            for file_info in mismatched_files:
                report_content += f"  - File: {file_info['file_path']}"
                report_content += f"    Actual modality: {file_info['actual_modality']}\n"

            #report_content = "\n".join(report_lines)
            report_content += f"\nTotal Errors: {len(mismatched_files)}\n"
            
            report_path = os.path.join(self.output_directory_report, f"{report_name}.txt")
            host_report_path = os.path.join(self.host_output_directory_report, f"{report_name}.txt")

            with open(report_path, "w") as report_file:
                report_file.write(report_content)

            # Generate the check file with errors using the standardized method
            generate_check_file_image_data(
                check_name=phase_name,
                phase_number=phase_number,
                series_group_name=series_group,
                num_slices_group=num_slices_group,
                num_patients_with_image_data=len(patient_folders),
                num_series=total_series,
                num_tot_slices=total_slices,
                timestamp=current_datetime,
                error_counts=error_counts,
                warning_counts={},
                error_descriptions=error_descriptions,
                warning_descriptions={},
                output_dir=self.output_directory_checkfile,
                existing_warnings=existing_warnings
            )
                
            raise ValueError(
                f"Some images are not of the correct type according to the protocol. "
                f"See the detailed report at: {host_report_path}" #!! Path del container, deve essere quello locale.OK
            )
        
        else:
            # Generate the check file with no errors
            generate_check_file_image_data(
                check_name=phase_name,
                phase_number=phase_number,
                series_group_name=series_group,
                num_slices_group=num_slices_group,
                num_patients_with_image_data=len(patient_folders),
                num_series=total_series,
                num_tot_slices=total_slices,
                timestamp=current_datetime,
                error_counts=error_counts,
                warning_counts={},
                error_descriptions=error_descriptions,
                warning_descriptions={},
                output_dir=self.output_directory_checkfile,
                existing_warnings=existing_warnings
            )

            success_message_image_type = "All the images are of the correct type according to the protocol. "
        
        return success_message_image_type
    

    def check_for_corrupted_dicom_files(self): #2D e 3D
        """
        Checks each DICOM file in the specified root directory for corruption (e.g., mismatched pixel data length).
        If any corrupted files are found, it generates a detailed report.
        """
        phase_number = 32  
        phase_data = self.mapping_file.get("check_phase", {}).get(str(phase_number), {})
        error_descriptions = phase_data.get("errors", {})
        phase_name = phase_data.get("name", f"Phase {phase_number}")  
        
        # Format check and report names dynamically
        check_name = phase_name.lower().replace(" ", "_") 
        report_name = f"1.032.{check_name}_report"

        check_file_path = os.path.join(self.output_directory_checkfile, "image_data_check_file.json")
        existing_warnings = retrieve_existing_warnings_image_data(check_file_path)
        _, num_slices_group_old, total_series, total_slices = extract_metadata_from_check_file(self.output_directory_checkfile)
        
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        series_group = self.series_group_name
        series_group_mapping = self.local_config["Local config"]["radiological data"]["series_mapping"]
        series_group_subfolder = series_group_mapping.get(series_group) #input name of the series
        image_group_config = self.protocol.get(series_group, {}).get("image", {}) #original self.protocol["image"]

        # Image and segmentation format validations
        image_type = image_group_config["type"]["selected"]
        image_file_format = image_group_config['image file format']['selected']
        
        # Initializations
        report_content = ""
        report_content_specific = ""
        num_errors = 0
        all_files_compressed = True  # Flag to track if all files are compressed
        error_counts = {"E3201": None}
        total_dicom_files_found = 0
        num_slices = 0
        num_slices_group = 0
        
        patient_folders = [f for f in os.listdir(self.images_dir) if os.path.isdir(os.path.join(self.images_dir, f)) and f != "Reports"]
        
        # Iterate through each patient folder in the root directory
        for patient_folder in patient_folders:
            patient_path = os.path.join(self.images_dir, patient_folder)

            if os.path.isdir(patient_path):
                
               series_folder = os.path.normpath(series_group_subfolder).split(os.sep)[-1].strip()
               series_path = os.path.join(patient_path, series_folder)

               num_slices = count_tot_num_slices_per_group(series_path, image_file_format)
               num_slices_group += num_slices  # Update total slices 
                        
               # Iterate over each DICOM file in the series folder
               for dicom_file in os.listdir(series_path):
                   dicom_path = os.path.join(series_path, dicom_file)

                   if dicom_file.lower().endswith('.dcm'):
                        total_dicom_files_found += 1

                   # Skip files that are not DICOM
                   if not dicom_file.lower().endswith('.dcm'):
                       continue

                   try:
                       # Load the DICOM file
                       dicom_data = pydicom.dcmread(dicom_path)

                       # compressed files store pixel data differently, so their byte length doesn't match the uncompressed size
                       # Check if DICOM file contains PixelData
                       if 'PixelData' in dicom_data:
                           if dicom_data.file_meta.TransferSyntaxUID.is_compressed:
                               print(f"Skipping compressed DICOM: {dicom_path}")
                               continue
                           else:
                               all_files_compressed = False  # Found at least one uncompressed file

                       # Attempt to access the pixel data to check for issues
                       pixel_array = dicom_data.pixel_array

                       # Get number of frames (important for multi-frame X-ray or fluoroscopy)
                       num_frames = getattr(dicom_data, "NumberOfFrames", 1)
    
                       # Expected pixel data length calculation
                       expected_length = (
                           num_frames *
                           dicom_data.Rows * 
                           dicom_data.Columns * 
                           dicom_data.SamplesPerPixel *   #number of color channels (samples) per pixel (1 for grayscale images)
                           dicom_data.BitsAllocated // 8
                       )

                       # Check if the length of the pixel data matches the expected size
                       #expected_length = np.prod(pixel_array.shape) * dicom_data.BitsAllocated // 8
                       actual_length = len(dicom_data.PixelData)

                       # If the lengths don't match, it indicates a corruption
                       if actual_length != expected_length:
                           num_errors += 1
                           error_counts["E3201"] = 1 if error_counts["E3201"] is None else error_counts["E3201"] + 1
                           report_content_specific += f"  - Corrupted File: {dicom_path}\n"
                           report_content_specific += f"    Pixel data length mismatch: {actual_length} != {expected_length}\n\n"

                   except Exception as e:
                       num_errors += 1
                       error_counts["E3201"] = 1 if error_counts["E3201"] is None else error_counts["E3201"] + 1
                       report_content_specific += f"  - Corrupted File: {dicom_path}\n"
                       report_content_specific += f"    {str(e)}\n\n"

        total_slices = total_slices - num_slices_group_old + num_slices_group
        
        if total_dicom_files_found == 0:
            raise ValueError("No DICOM files were found in the specified directories. Please check the input folder and series group mapping.")
                            
        if num_errors > 0:
            # Generate the report in the required format
            report_content = f"Report generated on: {current_datetime}\n\n"
            report_content += f"{phase_name} report:\n\n"
            report_content += f"Error E3201: {error_descriptions.get('E3201', None)}\n"
            report_content += "- Details: The following files have corrupted pixel data:\n"
            # Append the details of the corrupted files
            report_content += report_content_specific
            
            # Add total errors at the end
            report_content += f"\nTotal Errors: {num_errors}\n"
        
            # Save the report to the specified output directory
            report_name = f"{report_name}.txt"
            report_path = os.path.join(self.output_directory_report, report_name)
            host_report_path = os.path.join(self.host_output_directory_report, f"{report_name}.txt")

            with open(report_path, "w") as report_file:
                report_file.write(report_content)
                
            # Generate the check file with errors using the standardized method
            generate_check_file_image_data(
                check_name=phase_name,
                phase_number=phase_number,
                series_group_name=series_group,
                num_slices_group=num_slices_group,
                num_patients_with_image_data=len(patient_folders),
                num_series=total_series,
                num_tot_slices=total_slices,
                timestamp=current_datetime,
                error_counts=error_counts,
                warning_counts={},
                error_descriptions=error_descriptions,
                warning_descriptions={},
                output_dir=self.output_directory_checkfile,
                existing_warnings=existing_warnings
            )

    
            raise ValueError(
                f"At least one DICOM file contains corrupted pixel data (length mismatch). "
                f"See the detailed report at: {host_report_path}") #!! Path del container, deve essere quello locale.OK
        
        # Return a success message if no errors were found
        if num_errors == 0:
            generate_check_file_image_data(
                check_name=phase_name,
                phase_number=phase_number,
                series_group_name=series_group,
                num_slices_group=num_slices_group,
                num_patients_with_image_data=len(patient_folders),
                num_series=total_series,
                num_tot_slices=total_slices,
                timestamp=current_datetime,
                error_counts=error_counts,
                warning_counts={},
                error_descriptions=error_descriptions,
                warning_descriptions={},
                output_dir=self.output_directory_checkfile,
                existing_warnings=existing_warnings
            )

            
            if all_files_compressed:
                success_msg_corrupted_files = "All DICOM files are compressed; no validation performed."
            else:
                success_msg_corrupted_files = "DICOM validation complete: no corrupted files detected."
            
            return success_msg_corrupted_files
        
        
    def check_dicom_orientation(self): # SOLO per x-ray DICOM 2D
        """
        Checks Patient Orientation (0020, 0020) and View Position (0018, 5101) metadata in the provided DICOM files to ensure they match
        the expected values in the protocol.
        """

        phase_number = 33  
        phase_data = self.mapping_file.get("check_phase", {}).get(str(phase_number), {})
        error_descriptions = phase_data.get("errors", {})
        warning_descriptions = phase_data.get("warnings", {})
        phase_name = phase_data.get("name", f"Phase {phase_number}")  
        
        # Format check and report names dynamically
        check_name = phase_name.lower().replace(" ", "_") 
        report_name = f"1.033.{check_name}_report"
        check_file_path = os.path.join(self.output_directory_checkfile, "image_data_check_file.json")

        existing_warnings = retrieve_existing_warnings_image_data(check_file_path)
        _, num_slices_group_old, total_series, total_slices = extract_metadata_from_check_file(self.output_directory_checkfile)

        series_group = self.series_group_name
        series_group_mapping = self.local_config["Local config"]["radiological data"]["series_mapping"]
        series_group_subfolder = series_group_mapping.get(series_group) #input name of the series
        image_group_config = self.protocol.get(series_group, {}).get("image", {}) #original self.protocol["image"]
        
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        image_file_format = image_group_config['image file format']['selected']
        missing_metadata_files = []  # Store DICOMs missing metadata
        
        # Initialize counters and report content
        num_errors = 0
        num_warnings = 0  # Track how many files have missing orientation metadata
        
        error_counts = {"E3301": None}
        warning_counts = {"W3301": None}
        
        report_content_specific = ""
        num_slices = 0
        num_slices_group = 0
 
        patient_folders = [f for f in os.listdir(self.images_dir) if os.path.isdir(os.path.join(self.images_dir, f)) and f != "Reports"]
        
        # Iterate through each patient folder
        for patient_folder in patient_folders:
            patient_path = os.path.join(self.images_dir, patient_folder)
            
            if os.path.isdir(patient_path):
                
               series_folder = os.path.normpath(series_group_subfolder).split(os.sep)[-1].strip()
               series_path = os.path.join(patient_path, series_folder)

               num_slices = count_tot_num_slices_per_group(series_path, image_file_format)
               num_slices_group += num_slices  # Update total slices

               # Iterate through each DICOM file in the series folder
               for dicom_file in os.listdir(series_path):
                   dicom_path = os.path.join(series_path, dicom_file)

                   if dicom_file.endswith('.dcm'):
                        try:
                            dicom_data = pydicom.dcmread(dicom_path)

                            # Get Patient Orientation and View Position metadata
                            patient_orientation = dicom_data.get((0x0020, 0x0020))  # Patient Orientation (0020, 0020)
                            view_position = dicom_data.get((0x0018, 0x5101))  # View Position (0018, 5101)
                            
                            # Extract the values, treating empty strings as None
                            patient_orientation_value = patient_orientation.value if patient_orientation and patient_orientation.value else None
                            view_position_value = view_position.value if view_position and view_position.value else None

                            # If both metadata fields are present, check for compliance with the protocol
                            if patient_orientation_value and view_position_value:
                                expected_orientation = image_group_config["image_metadata"]["patient orientation"]["selected"]
                                expected_view_position = image_group_config["image_metadata"]["view position"]["selected"]

                                patient_orientation_str = str(patient_orientation_value)
                                view_position_str = str(view_position_value)

                                incorrect_metadata = []

                                if patient_orientation_str != expected_orientation:
                                    incorrect_metadata.append(f"    Patient Orientation: {patient_orientation_value} (Expected: {expected_orientation})")

                                if view_position_str != expected_view_position:
                                    incorrect_metadata.append(f"    View Position: {view_position_value} (Expected: {expected_view_position})")

                                # Compare metadata with expected values
                                if incorrect_metadata:
                                    num_errors += 1
                                    error_counts["E3301"] = 1 if error_counts["E3301"] is None else error_counts["E3301"] + 1
                                    report_content_specific += f"  - Incorrect Metadata: {dicom_path}\n"
                                    report_content_specific += "\n".join(incorrect_metadata) + "\n\n"
                            else:
                                # Count files missing metadata as warnings
                                num_warnings += 1
                                warning_counts["W3301"] = 1 if warning_counts["W3301"] is None else warning_counts["W3301"] + 1
                                missing_metadata_files.append(f"Patient: {patient_folder}, Series: {series_folder}, File: {dicom_file}")
                            
                        except Exception as e:
                            num_errors += 1
                            error_counts["E3301"] = 1 if error_counts["E3301"] is None else error_counts["E3301"] + 1
                            report_content_specific += f"  - Error in File: {dicom_path}\n"
                            report_content_specific += f"    {str(e)}\n\n"

        total_slices = total_slices - num_slices_group_old + num_slices_group
        error_counts = {"E3301": num_errors if num_errors > 0 else None}
        # Generate check file with error details
        generate_check_file_image_data(
                check_name=phase_name,
                phase_number=phase_number,
                series_group_name=series_group,
                num_slices_group=num_slices_group,
                num_patients_with_image_data=len(patient_folders),
                num_series=total_series,
                num_tot_slices=total_slices,
                timestamp=current_datetime,
                error_counts=error_counts,
                warning_counts={},
                error_descriptions=error_descriptions,
                warning_descriptions={},
                output_dir=self.output_directory_checkfile,
                existing_warnings=existing_warnings
            )
        # If errors are found, generate a report
        if num_errors > 0:
            report_content = f"Report generated on: {current_datetime}\n\n"
            report_content += f"{phase_name} report:\n\n"
            report_content += f"Error E3301: {error_descriptions.get('E3301', None)}\n"
            report_content += "- Details: The following files have metadata issues:\n"
            report_content += report_content_specific
            report_content += f"\nTotal Errors: {num_errors}\n"
            
            report_name = f"{report_name}.txt"
            report_path = os.path.join(self.output_directory_report, report_name)
            host_report_path = os.path.join(self.host_output_directory_report, report_name)
            with open(report_path, "w") as report_file:
                report_file.write(report_content)

            raise ValueError(
                f"Orientation metadata mismatch found in at least one DICOM file. "
                f"See the detailed report at: {host_report_path}" #!! Path del container, deve essere quello locale.OK
            )
            
        # Handle success message based on conditions
        if num_warnings == total_slices:
            orientation_check_msg = "Warning - Orientation metadata not found in any DICOM file. No possible check."
        elif num_warnings > 0:
            missing_files_info = "\n  - ".join(missing_metadata_files)
            orientation_check_msg = ( 
                f"DICOM orientation metadata check complete: no orientation errors detected. Warning W3301: {warning_descriptions.get('W3301', None)}\n"
                f"  Missing metadata details in:\n  - {missing_files_info}"
            )
        else:
            orientation_check_msg = "DICOM orientation metadata check complete: no orientation errors detected."
        
        return orientation_check_msg, num_warnings
    
    
    def check_uniform_images(self): # 2D only 
        """
        Checks for completely uniform DICOM images (all-black, all-white, or all-gray) in patient series folders.
        Generates a check file and, if any uniform images are found, an error report.
        """
        phase_number = 34
        phase_data = self.mapping_file.get("check_phase", {}).get(str(phase_number), {})
        error_descriptions = phase_data.get("errors", {})
        phase_name = phase_data.get("name", f"Phase {phase_number}")
    
        check_name = phase_name.lower().replace(" ", "_")
        report_name = f"1.034.{check_name}_report"
    
        series_group = self.series_group_name
        series_group_mapping = self.local_config["Local config"]["radiological data"]["series_mapping"]
        series_group_subfolder = series_group_mapping.get(series_group)
    
        image_group_config = self.protocol.get(series_group, {}).get("image", {})
        image_format_selected = image_group_config['image file format']['selected'].lower()  # e.g., "dcm"
    
        check_file_path = os.path.join(self.output_directory_checkfile, "image_data_check_file.json")
        existing_warnings = retrieve_existing_warnings_image_data(check_file_path)
    
        output_directory_checkfile = self.output_directory_checkfile
        _, num_slices_group_old, total_series, total_slices = extract_metadata_from_check_file(output_directory_checkfile)
    
        error_details = []
        total_affected_images = 0
        num_slices = 0
        num_slices_group = 0
    
        patient_folders = [
            f for f in os.listdir(self.images_dir)
            if os.path.isdir(os.path.join(self.images_dir, f)) and f != "Reports"
        ]
    
        for patient_folder in patient_folders:
            patient_path = os.path.join(self.images_dir, patient_folder)
            series_folder = os.path.normpath(series_group_subfolder).split(os.sep)[-1].strip()
            series_path = os.path.join(patient_path, series_folder)

            num_slices = count_tot_num_slices_per_group(series_path, image_format_selected)
            num_slices_group += num_slices  # Update total slices
    
            if not os.path.isdir(series_path):
                continue
    
            image_files = [
                f for f in os.listdir(series_path)
                if f.lower().endswith(image_format_selected)
            ]
    
            for image_file in image_files:
                image_path = os.path.join(series_path, image_file)
    
                try:
                    image = sitk.ReadImage(image_path)
                    image_array = sitk.GetArrayFromImage(image)

                    # If 3D (e.g., shape = (1, H, W)), take the 2D slice
                    if image_array.ndim == 3 and image_array.shape[0] == 1:
                        image_array = image_array[0]

                    if np.std(image_array) == 0:
                        error_details.append((patient_folder, series_folder, image_file))
                        total_affected_images += 1
    
                except Exception as e:
                    print(f"Error processing {image_path}: {str(e)}")
    
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        error_counts = {"E3401": total_affected_images if total_affected_images > 0 else None}
        
        total_slices = total_slices - num_slices_group_old + num_slices_group
        generate_check_file_image_data(
            check_name=phase_name,
            phase_number=phase_number,
            series_group_name=series_group,
            num_slices_group=num_slices_group,
            num_patients_with_image_data=len(patient_folders),
            num_series=total_series,
            num_tot_slices=total_slices,
            timestamp=current_datetime,
            error_counts=error_counts,
            warning_counts={},
            error_descriptions=error_descriptions,
            warning_descriptions={},
            output_dir=self.output_directory_checkfile,
            existing_warnings=existing_warnings
        )
    
        if not error_details:
            uniform_image_message = "No uniform (completely black, white, or gray) DICOM images detected."
            return uniform_image_message
    
        report_path = os.path.join(self.output_directory_report, f"{report_name}.txt")
        host_report_path = os.path.join(self.host_output_directory_report, f"{report_name}.txt")
    
        with open(report_path, "w") as report_file:
            report_file.write(f"Report generated on: {current_datetime}\n\n")
            report_file.write(f"{phase_name} report:\n\n")
            report_file.write(f"Error E3401: {error_descriptions.get('E3401', None)}\n")
            report_file.write("- Details:\n")
    
            for patient, series, image in error_details:
                report_file.write(f"  - Patient: {patient}, Series: {series}, DICOM: {image}\n")
    
            report_file.write(f"\nTotal Errors: {total_affected_images}\n")
    
        print(f"Uniform DICOM image report saved at: {host_report_path}") 
        raise Exception(
            f"Uniform images detected. "
            f"See the detailed report at: {host_report_path}" #!! Path del container, deve essere quello locale.OK
        )


    def check_non_axial_slices(self): 
        
        phase_number = 33  
        phase_data = self.mapping_file.get("check_phase", {}).get(str(phase_number), {})
        error_descriptions = phase_data.get("errors", {})
        phase_name = phase_data.get("name", f"Phase {phase_number}")  
        
        # Format check and report names dynamically
        check_name = phase_name.lower().replace(" ", "_") 
        report_name = f"1.033.{check_name}_report"

        total_slices_checked = 0
        num_slices = 0
        num_slices_group = 0
        num_errors = 0
        non_axial_slices = []  # List to store non-axial slices for the report
        non_axial_series_dict = {}  # Dictionary to group non-axial slices by series
        axial_success_message = ""

        # Define error counts
        error_counts = {"E3301": None} 
        total_dicom_files_found = 0

        series_group = self.series_group_name
        series_group_mapping = self.local_config["Local config"]["radiological data"]["series_mapping"]
        series_group_subfolder = series_group_mapping.get(series_group) #input name of the series
        image_group_config = self.protocol.get(series_group, {}).get("image", {}) #original self.protocol["image"]
        
        image_file_format = image_group_config['image file format']['selected']

        # Date formatting for reports and check file
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        check_file_path = os.path.join(self.output_directory_checkfile, "image_data_check_file.json")
        # Retrieve existing warnings
        existing_warnings = retrieve_existing_warnings_image_data(check_file_path)
        _, num_slices_group_old, total_series, total_slices = extract_metadata_from_check_file(self.output_directory_checkfile)
        
        patient_folders = [
            f for f in os.listdir(self.images_dir)
            if os.path.isdir(os.path.join(self.images_dir, f)) and f != "Reports"
        ]

        # Iterate through each patient folder in the root directory
        for patient_folder in patient_folders:
            patient_path = os.path.join(self.images_dir, patient_folder)
                
            series_folder = os.path.normpath(series_group_subfolder).split(os.sep)[-1].strip()
            series_path = os.path.join(patient_path, series_folder)

            num_slices = count_tot_num_slices_per_group(series_path, image_file_format)
            num_slices_group += num_slices  # Update total slices 

            # Iterate over each DICOM file in the series folder
            for dicom_file in os.listdir(series_path):
                dicom_path = os.path.join(series_path, dicom_file)

                if not dicom_file.lower().endswith('.dcm'):
                    continue
                total_dicom_files_found += 1

                # Load the DICOM file
                ds = pydicom.dcmread(dicom_path)
                image_orientation = ds.get("ImageOrientationPatient", None)
                if image_orientation is None:
                    raise ValueError("Missing ImageOrientationPatient tag.")
                plane = classify_slice_orientation(image_orientation)
                total_slices_checked += 1

                if plane != "Axial":
                   non_axial_slices.append((patient_folder, series_folder, dicom_file, plane))
                   key = (patient_folder, series_folder)
                   if key not in non_axial_series_dict:
                        non_axial_series_dict[key] = []
                   non_axial_series_dict[key].append((dicom_file, plane))
                   num_errors += 1

        total_slices = total_slices - num_slices_group_old + num_slices_group
        
        error_counts = {"E3301": num_errors if num_errors > 0 else None}
    
        ## Use generate_check_file_image_data to create check file
        generate_check_file_image_data(
            check_name=phase_name,
            phase_number=phase_number,
            series_group_name=series_group,
            num_slices_group=num_slices_group,
            num_patients_with_image_data=len(patient_folders),
            num_series=total_series,
            num_tot_slices=total_slices,
            timestamp=current_datetime,
            error_counts=error_counts,
            warning_counts={},
            error_descriptions=error_descriptions,
            warning_descriptions={},
            output_dir=self.output_directory_checkfile,
            existing_warnings=existing_warnings
        )

        # Generate report for non-axial slices if any
        if num_errors > 0:
            report_path = os.path.join(self.output_directory_report, f"{report_name}.txt")
            host_report_path = os.path.join(self.host_output_directory_report, f"{report_name}.txt")

            with open(report_path, "w") as report_file:
                report_file.write(f"Generated on {current_datetime}\n\n")
                report_file.write(f"{phase_name} report:\n\n")
                report_file.write(f"Error E3301: {error_descriptions.get('E3301', None)}\n")
                report_file.write("- Details: The following series contain non-axial slices:\n")
                
                for (patient_folder, series), slices in non_axial_series_dict.items():
                    report_file.write(f"  - Patient: {patient_folder}, Series: {series}:\n")
                    for filename, plane in slices:
                        report_file.write(f"    File: {filename} - Orientation: {plane}\n")
                    report_file.write("\n")  # Add a newline after each series block
                report_file.write(f"Total Errors: {num_errors}\n\n")
        
            # After generating the report and check file, raise an error due to non-axial slices
            series_with_errors = ', '.join([f"{series} ({patient})" for patient, series in non_axial_series_dict.keys()])
            raise ValueError(f"Found {len(non_axial_slices)} non-axial slices across the following series: {series_with_errors}. "
                             f"See the detailed report at: {host_report_path}") #!! Path del container, deve essere quello locale.OK

        if not non_axial_slices:
            axial_success_message += "All slices are axial."
        
        print(f"Total slices checked: {total_slices_checked}")
        
        return axial_success_message
    
    
    def check_slice_dimensions_dicom(self): #TESTO SULLE 3D
        """Check if DICOM slices are smaller than the target dimensions specified in the protocol."""
        
        phase_number = 35  
        phase_data = self.mapping_file.get("check_phase", {}).get(str(phase_number), {})
        error_descriptions = phase_data.get("errors", {})
        phase_name = phase_data.get("name", f"Phase {phase_number}")  
        
        # Format check and report names dynamically
        check_name = phase_name.lower().replace(" ", "_") 
        report_name = f"1.035.{check_name}_report"

        series_group = self.series_group_name
        series_group_mapping = self.local_config["Local config"]["radiological data"]["series_mapping"]
        series_group_subfolder = series_group_mapping.get(series_group)
        image_group_config = self.protocol.get(series_group, {}).get("image", {})
        image_file_format = image_group_config['image file format']['selected']

        # Extract size from the protocol
        target_size = tuple(image_group_config["size"])
        target_rows, target_cols = target_size

        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S") 
        check_file_path = os.path.join(self.output_directory_checkfile, "image_data_check_file.json")
        existing_warnings = retrieve_existing_warnings_image_data(check_file_path)
        _, num_slices_group_old, total_series, total_slices = extract_metadata_from_check_file(self.output_directory_checkfile)
    
        smaller_slices_report = []
        all_slices_smaller = True
        total_slices_group = 0
        slice_dim_message = ""
        num_slices = 0
        num_slices_group = 0

        patient_folders = [
            f for f in os.listdir(self.images_dir)
            if os.path.isdir(os.path.join(self.images_dir, f)) and f != "Reports"
        ]

        for patient_folder in patient_folders:
            patient_path = os.path.join(self.images_dir, patient_folder)
            series_folder = os.path.normpath(series_group_subfolder).split(os.sep)[-1].strip()
            series_path = os.path.join(patient_path, series_folder)

            num_slices = count_tot_num_slices_per_group(series_path, image_file_format)
            num_slices_group += num_slices  # Update total slices
    
            if not os.path.exists(series_path):
                continue
    
            for dicom_file in os.listdir(series_path):
                if not dicom_file.lower().endswith('.dcm'):
                    continue

                dicom_path = os.path.join(series_path, dicom_file)
                try:
                    image = sitk.ReadImage(dicom_path)
                    image_array = sitk.GetArrayFromImage(image)

                    # If 3D (e.g., shape = (1, H, W)), take the 2D slice
                    if image_array.ndim == 3 and image_array.shape[0] == 1:
                        image_array = image_array[0]
                        
                    rows, cols = image_array.shape
                    total_slices_group += 1
    
                    if rows < target_rows or cols < target_cols:
                        msg = (
                            f"Patient: {patient_folder}, Series: {series_folder}, "
                            f"File: {dicom_file}, Slice size: {rows}x{cols}"
                        )
                        smaller_slices_report.append(msg)
                    else:
                        all_slices_smaller = False  # Found at least one slice that meets/exceeds size
    
                except Exception as e:
                    raise ValueError(f"Error reading DICOM file {dicom_path}: {e}")
    
    
        error_counts = {"E3501": len(smaller_slices_report) if smaller_slices_report else None}
        total_slices = total_slices - num_slices_group_old + num_slices_group
        # Generate check file
        generate_check_file_image_data(
            check_name=phase_name,
            phase_number=phase_number,
            series_group_name=series_group,
            num_slices_group=num_slices_group,
            num_patients_with_image_data=len(patient_folders),
            num_series=total_series,
            num_tot_slices=total_slices,
            timestamp=current_datetime,
            error_counts=error_counts,
            warning_counts={},
            error_descriptions=error_descriptions,
            warning_descriptions={},
            output_dir=self.output_directory_checkfile,
            existing_warnings=existing_warnings
        )
    
        # Handle error reporting
        if smaller_slices_report:
            os.makedirs(self.output_directory_report, exist_ok=True)
            report_path = os.path.join(self.output_directory_report, f"{report_name}.txt")
            host_report_path = os.path.join(self.host_output_directory_report, f"{report_name}.txt")
    
            with open(report_path, "w") as report_file:
                report_file.write(f"Report generated on: {current_datetime}\n\n")
                report_file.write(f"{phase_name} report:\n\n")
                report_file.write(f"Error E3501: {error_descriptions.get('E3501', 'Slice dimensions are smaller than expected.')}\n\n")
    
                if all_slices_smaller:
                    report_file.write(f"- Details: All slices are smaller than the target size {target_rows}x{target_cols}. Upscaling is not allowed.\n\n")
                else:
                    report_file.write(f"- Details: The following slices are smaller than the target size {target_rows}x{target_cols}:\n\n")
    
                for line in smaller_slices_report:
                    report_file.write("  " + line + "\n")
    
                report_file.write(f"\nTotal Errors: {len(smaller_slices_report)}\n")
    
            if all_slices_smaller:
                raise ValueError(f"All slices are smaller than the target size. "
                                 f"See the detailed report at: {host_report_path}") #!! Path del container, deve essere quello locale.OK
            else:
                raise ValueError(f"Some slices are smaller than the target size. "
                                 f"See the detailed report at: {host_report_path}") #!! Path del container, deve essere quello locale.OK
    
        # No errors found
        slice_dim_message += "All slices meet or exceed the target dimensions. Resizing operations are permitted."

        return slice_dim_message
    

    def generate_DicomValidation3D_final_report(self, success_anonymization, success_message_image_type, success_msg_corrupted_files, axial_success_message, slice_dim_message):
        # Get the current datetime
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")

        # Construct the final report
        final_report_lines = [
        f"Report generated on: {formatted_datetime}",
        "",
        "DICOM compliance final report:",
        "",
        "DICOM anonymization check:",
        f"- {success_anonymization}",
        "",
        "Image type check:",
        f"- {success_message_image_type}",
        "",
        "Corrupted files check:",
        f"- {success_msg_corrupted_files}",
        "",
        "Anatomical plane check:",
        f"- {axial_success_message}",
        "",
        "Slice dimensions check:",
        f"- {slice_dim_message}",
        "",
        "Summary:",
        "All checks completed."
        ]
    
        final_report = "\n".join(final_report_lines)
            
        # Define the report filename
        report_filename = os.path.join(self.output_directory_report, "1.DicomValidation3D_final_report.txt") 
    
        # Write the final report to a file
        with open(report_filename, "w") as report_file:
            report_file.write(final_report)
    
        # Print a message indicating that the final report has been generated
        print("DicomValidation3D final report has been generated.")


    def generate_DicomValidation2D_final_report(self, success_anonymization, success_message_image_type, success_msg_corrupted_files, orientation_check_msg, num_warnings, uniform_image_message, slice_dim_message): 
        # Get the current datetime
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")

        # Construct the final report
        final_report_lines = [
        f"Report generated on: {formatted_datetime}",
        "",
        "DICOM compliance final report:",
        "",
        "DICOM anonymization check:",
        f"- {success_anonymization}",
        "",
        "Image type check:",
        f"- {success_message_image_type}",
        "",
        "Corrupted files check:",
        f"- {success_msg_corrupted_files}",
        "",
        "DICOM orientation check:",
        f"- {orientation_check_msg}",
        "",
        "Uniform image check:",
        f"- {uniform_image_message}",
        "",
        "Slice dimensions check:",
        f"- {slice_dim_message}",
        "",
        "Summary:",
        f"All checks completed. Total warnings: {num_warnings}."
        ]
    
        final_report = "\n".join(final_report_lines)
            
        # Define the report filename
        report_filename = os.path.join(self.output_directory_report, "1.DicomValidation2D_final_report.txt") 
    
        # Write the final report to a file
        with open(report_filename, "w") as report_file:
            report_file.write(final_report)
    
        # Print a message indicating that the final report has been generated
        print("DicomValidation2D final report has been generated.")


# invece di prendere in input num_image_patients lo carico da input_validation_state.json
def run_dicom_validator(protocol, local_config, tag_list, mapping_file, series_group_name):

    # Define the input directory path 
    input_dir = os.getenv("INPUT_DIR")

    # Clear log at the start of validation
    clear_log_file(input_dir, LOG_FILENAME)

    image_group_config = protocol.get(series_group_name, {}).get("image", {}) 
    image_type = image_group_config['type']['selected']
    # Load the input validation state to extract num_image_patients
    input_state_file = os.path.join(input_dir, "input_validation_state.json")

    try:
        with open(input_state_file, "r") as f:
            state = json.load(f)
        num_image_patients = state.get("num_patients_with_image_data", 0)
    except Exception as e:
        raise RuntimeError(f"Failed to load state file '{input_state_file}': {e}")
    
    # Create an instance of DicomValidator
    dicom_validator = DicomValidator(protocol, local_config, tag_list, mapping_file, num_image_patients, series_group_name)

    state = load_state(dicom_validator.state_file)
    series_progress_state =  load_state(dicom_validator.series_progress_file)

    # Access or create the per-series progress dictionary
    series_state = series_progress_state.setdefault(series_group_name, {})
    last_phase_done = series_state.get("last_successful_phase", 0)
    
    if last_phase_done < 30: 
        print("Running check_anonymization function...") # check phase 30
        try:
            state["success_anonymization_dicom"] = dicom_validator.check_anonymization()
            save_state(state, dicom_validator.state_file)  # Save updated state

            # Update series progress state
            series_state["last_successful_phase"] = 30
            series_progress_state[series_group_name] = series_state
            save_state(series_progress_state, dicom_validator.series_progress_file)
        except Exception as e:
            log_error(input_dir, "check_anonymization", e, LOG_FILENAME)
            print(f"An unexpected error occurred during check_anonymization. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
            raise  
        
    if last_phase_done < 31:
        print("Running check_image_type function...") # check phase 31
        try:
            state["success_message_image_type"] = dicom_validator.check_image_type()
            save_state(state, dicom_validator.state_file)  # Save updated state

            series_state["last_successful_phase"] = 31
            series_progress_state[series_group_name] = series_state
            save_state(series_progress_state, dicom_validator.series_progress_file)
        except Exception as e:
            log_error(input_dir, "check_image_type", e, LOG_FILENAME)
            print(f"An unexpected error occurred during check_image_type. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
            raise  

    if last_phase_done < 32:
        print("Running check_for_corrupted_dicom_files function...") # check phase 32
        try:
            state["success_msg_corrupted_files"] = dicom_validator.check_for_corrupted_dicom_files()
            save_state(state, dicom_validator.state_file)  # Save updated state

            series_state["last_successful_phase"] = 32
            series_progress_state[series_group_name] = series_state
            save_state(series_progress_state, dicom_validator.series_progress_file)
        except Exception as e:
            log_error(input_dir, "check_for_corrupted_dicom_files", e, LOG_FILENAME)
            print(f"An unexpected error occurred during check_for_corrupted_dicom_files. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
            raise  

    if last_phase_done < 33:
        if image_type in ["CR", "DX", "RX"]:
            print("Running check_dicom_orientation function...") # check phase 33
            try:
                state["orientation_check_msg"], state["num_warnings"] = dicom_validator.check_dicom_orientation()
                save_state(state, dicom_validator.state_file)  # Save updated state

                series_state["last_successful_phase"] = 33
                series_progress_state[series_group_name] = series_state
                save_state(series_progress_state, dicom_validator.series_progress_file)
            except Exception as e:
                log_error(input_dir, "check_dicom_orientation", e, LOG_FILENAME)
                print(f"An unexpected error occurred during check_dicom_orientation. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
                raise  

        else: #3D
            print("Running check_non_axial_slices function...") # check phase 33
            try:
                state["non_axial_message"] = dicom_validator.check_non_axial_slices()
                save_state(state, dicom_validator.state_file)  # Save updated state

                series_state["last_successful_phase"] = 33
                series_progress_state[series_group_name] = series_state
                save_state(series_progress_state, dicom_validator.series_progress_file)
            except Exception as e:
                log_error(input_dir, "check_non_axial_slices", e, LOG_FILENAME)
                print(f"An unexpected error occurred during check_non_axial_slices. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
                raise  
    
    if last_phase_done < 34 and image_type in ["CR", "DX", "RX"]:
        print("Running check_uniform_images function...") # check phase 34
        try:
            state["uniform_image_message"] = dicom_validator.check_uniform_images()
            save_state(state, dicom_validator.state_file)  # Save updated state

            series_state["last_successful_phase"] = 34
            series_progress_state[series_group_name] = series_state
            save_state(series_progress_state, dicom_validator.series_progress_file)
        except Exception as e:
                log_error(input_dir, "check_uniform_images", e, LOG_FILENAME)
                print(f"An unexpected error occurred during check_uniform_images. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
                raise  

    if last_phase_done < 35:
        print("Running check_slice_dimensions_dicom function...") # check phase 35
        try:
            state["slice_dim_message"] = dicom_validator.check_slice_dimensions_dicom()
            save_state(state, dicom_validator.state_file)  # Save updated state

            series_state["last_successful_phase"] = 35
            series_progress_state[series_group_name] = series_state
            save_state(series_progress_state, dicom_validator.series_progress_file)
        except Exception as e:
                log_error(input_dir, "check_slice_dimensions_dicom", e, LOG_FILENAME)
                print(f"An unexpected error occurred during check_slice_dimensions_dicom. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
                raise  

        if image_type in ["CR", "DX", "RX"]:
            print("Running generate_DicomValidation2D_final_report function")
            try:
                dicom_validator.generate_DicomValidation2D_final_report(
                    state.get("success_anonymization_dicom"),
                    state.get("success_message_image_type"),
                    state.get("success_msg_corrupted_files"),
                    state.get("orientation_check_msg"),
                    state.get("num_warnings"),
                    state.get("uniform_image_message"),
                    state.get("slice_dim_message")
                )
            except Exception as e:
                log_error(input_dir, "generate_DicomValidation2D_final_report", e, LOG_FILENAME)
                print(f"An unexpected error occurred during generate_DicomValidation2D_final_report. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
                raise 
        else:
            print("Running generate_3DDicomValidation_final_report function")
            try:
                dicom_validator.generate_DicomValidation3D_final_report(
                    state.get("success_anonymization_dicom"),
                    state.get("success_message_image_type"),
                    state.get("success_msg_corrupted_files"),
                    state.get("non_axial_message"),
                    state.get("slice_dim_message")
                )
            except Exception as e:
                log_error(input_dir, "generate_3DDicomValidation_final_report", e, LOG_FILENAME)
                print(f"An unexpected error occurred during generate_3DDicomValidation_final_report. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
                raise 

    

    



    

