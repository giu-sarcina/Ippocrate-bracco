import pandas as pd
import os 
import csv
from datetime import datetime
import json
from utils import save_state, load_state, clear_log_file, log_error, numeric_suffix_sort_key
from .helpers import (
    generate_check_file_image_data,
    count_tot_num_slices_per_group,
    retrieve_existing_warnings_image_data,
    extract_metadata_from_check_file,
    count_tot_num_series_per_dataset,
    count_tot_num_slices_per_dataset
)

LOG_FILENAME = "image_structure_and_label_validation_error.log"

# Image structure and label validation
class ImageStructureAndLabelValidator:
    
    def __init__(self, protocol, local_config=None, mapping_file=None, num_image_patients=0, series_group_name="default_group"):
        self.protocol = protocol
        self.local_config = local_config if local_config is not None else {}
        self.mapping_file = mapping_file if mapping_file is not None else {}
        self.series_group_name = series_group_name
        self.num_image_patients = num_image_patients 
        self.study_path = os.getenv("ROOT_NAME")
        self.host_study_path = os.getenv("HOST_ROOT_DIR")
        self.input_dir = os.getenv("INPUT_DIR")
        self.images_dir = os.path.join(self.input_dir, "IMAGES")
        self.host_input_dir = os.getenv("HOST_BASE_DATA_DIR")
        self.host_images_dir = os.path.join(self.host_input_dir, "IMAGES")
        self.state_file = os.path.join(self.input_dir, ".json")
        self.series_progress_file = os.path.join(self.input_dir, "validation_progress_by_series.json")
        self.output_directory_report = os.path.join(self.input_dir, "Reports", f"Image Data Reports {self.series_group_name}")
        self.host_output_directory_report = os.path.join(self.host_input_dir, "Reports", f"Image Data Reports {self.series_group_name}")
        self.output_directory_checkfile = os.path.join(self.study_path, "CHECKFILE")
        self.output_directory_data = os.path.join(self.study_path, "IMAGES")
        os.makedirs(self.output_directory_report, exist_ok=True)  
        os.makedirs(self.output_directory_checkfile, exist_ok=True)
        os.makedirs(self.output_directory_data, exist_ok=True)
    

    def create_patient_folder_structure(self, total_series, total_slices): # total_slices da count_tot_num_slices_per_dataset#ok

        phase_number = 20  
        phase_data = self.mapping_file.get("check_phase", {}).get(str(phase_number), {})
        error_descriptions = phase_data.get("errors", {})
        phase_name = phase_data.get("name", f"Phase {phase_number}") 
        
        # Format check and report names dynamically
        check_name = phase_name.lower().replace(" ", "_") 
        report_name = f"0.020.{check_name}_report"
        
        # Get the current datetime
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
        patients_list_filename = self.local_config["Local config"]["patients list"]
        patients_list_path = os.path.join(self.input_dir, patients_list_filename)

        series_group = self.series_group_name
        series_group_mapping = self.local_config["Local config"]["radiological data"]["series_mapping"]
        image_group_config = self.protocol.get(series_group, {}).get("image", {})
        protocol_group_specific = self.protocol.get(series_group, {})

        check_file_path = os.path.join(self.output_directory_checkfile, "image_data_check_file.json")
        existing_warnings = retrieve_existing_warnings_image_data(check_file_path)

        is_first_check = not os.path.exists(check_file_path)
        
        if not is_first_check:
            previous_series_group, num_slices_group_old, total_series, total_slices = extract_metadata_from_check_file(self.output_directory_checkfile)

        # Read patients_list_filename file
        with open(patients_list_path, 'r', newline='', encoding='utf-8') as csvfile:
            sniffer = csv.Sniffer()
            sample = csvfile.read(1024)  # Read a small sample of the file (maybe need to increase it)
            csvfile.seek(0)  # Reset file pointer to the beginning
            dialect = sniffer.sniff(sample)

            if dialect.delimiter == '\t':
                sep = '\t'
            elif dialect.delimiter == ';':
                sep = ';'  
            elif dialect.delimiter == ',':
                sep = ','
                
        patients_list_df = pd.read_csv(patients_list_path, sep=sep)
        success_creation = ""
    
        # Check that the patients_list_df has at least 4 columns
        if 'clinical data' not in self.protocol:
            if patients_list_df.shape[1] < 4:
                raise ValueError("patients_list_filename should have at least four columns: ID, Clinical Data (0/1), Image Data (0/1), New Patient ID")
    
        # Create a mapping only for patients with image data (second column == 1)
        image_mapping = patients_list_df[patients_list_df.iloc[:, 2] == 1].set_index(patients_list_df.columns[0])[patients_list_df.columns[4]].to_dict()
    
        # Extract the image modality from the protocol and convert it to lowercase
        image_modality = image_group_config.get("type", {}).get("selected", "unknown").lower()
        #print("image_modality", image_modality)
        image_file_format = image_group_config.get("image file format", {}).get("selected", "unknown").lower()
        #print("image_file_format", image_file_format)

        # List directories that represent patient folders (assuming they are named by patient ID)
        patient_folders = [f for f in os.listdir(self.images_dir) if os.path.isdir(os.path.join(self.images_dir, f)) and f != "Reports"]

        # Initialize counts
        num_slices_group = 0
        # Create a list for storing mappings
        series_mapping = []  
        # Create a list for storing wrong associations
        mismatches = []
        # Initialize error count
        num_errors = 0
        error_counts = {"E2001": None}

        # Iterate over the patient folders and attempt to map them to the new patient IDs
        for old_id in patient_folders:
            if old_id.isdigit():
                old_id_int = int(old_id)  # "020" -> 20
                key_to_lookup = old_id_int
            else:
                key_to_lookup = old_id # keep as string

            if key_to_lookup in image_mapping:
                new_id = image_mapping[key_to_lookup]
                old_folder_path = os.path.join(self.images_dir, old_id)
                new_folder_path = os.path.join(self.output_directory_data, new_id)
    
                # Create the new patient folder if it doesn't exist
                if not os.path.exists(new_folder_path):
                    os.makedirs(new_folder_path, exist_ok=True)
                else:
                    print(f"Folder for patient {new_id} already exists. Checking for missing series/segmentation folders...")

                subfolder_path = series_group_mapping.get(series_group) 

                if subfolder_path:
                    expected_subfolder = os.path.normpath(subfolder_path).split(os.sep)[-1].strip()
                    series_full_path = os.path.join(old_folder_path, expected_subfolder)
                
                    if os.path.exists(series_full_path) and os.path.isdir(series_full_path):
                        # Extractpatient_folder_stru image modality and number from series group key
                        series_number = series_group.replace("series", "").split("_")[0]
                
                        five_letters = new_id[-5:]
                        new_series_name = f"series{series_number}_{image_modality}_{five_letters}"
    
                        # Define the source and destination paths for the series folder
                        new_series_path = os.path.join(new_folder_path, new_series_name)
        
                        # Create the new series folder if it doesn't exist
                        if not os.path.exists(new_series_path):
                            os.makedirs(new_series_path, exist_ok=True)
                        else:
                            print(f"Series folder {new_series_name} already exists for patient {new_id}. Skipping creation.")
    
                        # Append mapping to the series_mapping list
                        series_mapping.append((os.path.join(old_id, expected_subfolder), new_series_name))
                        
                        try:
                            num_slices = count_tot_num_slices_per_group(series_full_path, image_file_format)
                            num_slices_group += num_slices
                        except ValueError as e:
                            print(f"Warning: {e}")  # Log errors but continue processing
                        
                        # Check for segmentations and create the segmentation folder if required
                        if "segmentation" in protocol_group_specific: #!!

                            # Create the new segmentation folder name: seg_{image_modality}_{five_letters}_series{series_number}
                            seg_folder_name = f"seg_{image_modality}_{five_letters}_series{series_number}"
                            new_seg_folder_path = os.path.join(new_series_path, seg_folder_name)
    
                            # Create the new segmentation folder if it doesn't exist
                            if not os.path.exists(new_seg_folder_path):
                                os.makedirs(new_seg_folder_path, exist_ok=True)
                            else:
                                print(f"Segmentation folder {seg_folder_name} already exists for patient {new_id}. Skipping creation.")
            else:
                # Add to mismatches if the old ID is not found in the image_mapping
                mismatches.append(old_id)
                
        # Create a DataFrame from the series mapping list
        series_mapping_df = pd.DataFrame(series_mapping, columns=["Original Series Name", "New Series Name"])

        # Sort the DataFrame by the numeric part of the patient ID
        series_mapping_df_sorted = series_mapping_df.copy()
        series_mapping_df_sorted["sort_key"] = series_mapping_df_sorted["Original Series Name"].apply(
            lambda x: numeric_suffix_sort_key(x.split(os.sep)[0])
        )
        
        # Sort by the extracted numeric patient ID
        series_mapping_df_sorted = series_mapping_df_sorted.sort_values("sort_key").drop(columns=["sort_key"])
    
        # Save the series mapping to a CSV file
        series_mapping_csv_path = os.path.join(self.images_dir, "series_mapping.csv")
        if os.path.exists(series_mapping_csv_path):
            existing_df = pd.read_csv(series_mapping_csv_path)
            combined_df = pd.concat([existing_df, series_mapping_df_sorted], ignore_index=True).drop_duplicates()
            combined_df.to_csv(series_mapping_csv_path, index=False)
        else:
            series_mapping_df_sorted.to_csv(series_mapping_csv_path, index=False)
    
        # If there are mismatches or report content, generate a report
        if mismatches:
            report_path = os.path.join(self.output_directory_report, f"{report_name}.txt")
            host_report_path = os.path.join(self.host_output_directory_report, f"{report_name}.txt")

            with open(report_path, "w") as report_file:
                report_file.write(f"Report generated on: {current_datetime}\n\n")
                report_file.write(f"{phase_name} report:\n\n")
                report_file.write(f"Error E2001: {error_descriptions.get('E2001', None)}\n")
                report_file.write("- Details: The following patient folders were not found in the mapping file:\n")
                for mismatch in mismatches:
                    report_file.write(f"  - {mismatch}\n")
                num_errors += len(mismatches)
                error_counts["E2001"] = len(mismatches)
                report_file.write(f"\nTotal Errors: {num_errors}")
                
            print(f"Mismatch report generated at {report_path}")

        if not is_first_check and previous_series_group == series_group:
            total_slices = total_slices - num_slices_group_old + num_slices_group
        # Generate the check file
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
    
        # If mismatches or report content exists, raise an error
        if mismatches: 
            raise ValueError(
                f"Found {len(mismatches)} mismatched patient folder(s) that do not exist in the mapping file. "
                f"See the detailed report at: {host_report_path}") #ok
        else:
            success_creation = (
                "No mismatches found. Patient folder renaming and folder structure creation completed successfully.\n"
                f"- The series_mapping.csv file has been saved at {self.host_images_dir}." #ok
            )

        return success_creation
    

    def check_input_structure(self): 
        """Validates the input directory structure based on the local configuration and protocol."""

        phase_number = 21  
        phase_data = self.mapping_file.get("check_phase", {}).get(str(phase_number), {})
        error_descriptions = phase_data.get("errors", {})
        phase_name = phase_data.get("name", f"Phase {phase_number}")  
        
        # Format check and report names dynamically
        check_name = phase_name.lower().replace(" ", "_") 
        report_name = f"0.021.{check_name}_report" 

        series_group = self.series_group_name
        series_group_mapping = self.local_config["Local config"]["radiological data"]["series_mapping"]
        series_group_subfolder = series_group_mapping.get(series_group) #input name of the series
        image_group_config = self.protocol.get(series_group, {}).get("image", {})
        protocol_group_specific = self.protocol.get(series_group, {})

        check_file_path = os.path.join(self.output_directory_checkfile, "image_data_check_file.json")
        existing_warnings = retrieve_existing_warnings_image_data(check_file_path)

        _, num_slices_group_old, total_series, total_slices = extract_metadata_from_check_file(self.output_directory_checkfile)

        # Image and segmentation format validations
        image_type = image_group_config["type"]["selected"]
        image_format_selected = image_group_config['image file format']['selected']
        if "segmentation" in protocol_group_specific:
            segmentation_format_selected = protocol_group_specific['segmentation']['segmentation file format']['selected']
    
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Initialzations
        report_content = ""
        report_name = f"{report_name}.txt"
        
        # Separate content for image and segmentation errors
        image_errors = ""
        segmentation_errors = ""
        
        has_error = False  # To track if there are any errors
        num_image_errors = 0
        num_segmentation_errors = 0
        num_slices_group = 0
        total_patients = 0
        success_input_structure = ""
        error_counts = {
        "E2101": 0,
        "E2102": 0
        }

        patient_folders = [f for f in os.listdir(self.images_dir) if os.path.isdir(os.path.join(self.images_dir, f)) and f != "Reports"]
        total_patients = len(patient_folders)
        
        # Iterate over patient folders
        for patient_folder in patient_folders:
            patient_path = os.path.join(self.images_dir, patient_folder)

            expected_series_folder = os.path.normpath(series_group_subfolder).split(os.sep)[-1].strip()
            series_path = os.path.join(patient_path, expected_series_folder)

            # Find files that match the selected image format
            image_files = [f for f in os.listdir(series_path) if os.path.isfile(os.path.join(series_path, f)) and (f.endswith(image_format_selected) or f.endswith('.csv') or f.endswith('.txt'))]

            valid_image_files = [f for f in image_files if not (f.endswith('.csv') or f.endswith('.txt'))]

            invalid_files = [f for f in os.listdir(series_path) if os.path.isfile(os.path.join(series_path, f))
                             and not (f.endswith(image_format_selected) or f.endswith('.csv') or f.endswith('.txt'))]
            
            if invalid_files:
                has_error = True
                num_image_errors += 1
                image_errors += f"- Details: Found files with invalid formats in series folder '{expected_series_folder}' (patient '{patient_folder}'): {invalid_files}.\n"

            # Validate the number of image files based on the format selected, excluding CSV files
            if not valid_image_files:
                has_error = True
                num_image_errors += 1
                image_errors += f"- Details: Series folder '{expected_series_folder}' (patient '{patient_folder}') does not contain files in the expected image format {image_format_selected}.\n"
                    
                    
            elif image_format_selected in ['.nii', '.nii.gz', '.nrrd']:
                if len(valid_image_files) != 1:
                    has_error = True
                    num_image_errors += 1
                    image_errors += f"- Details: In series folder '{expected_series_folder}' (patient '{patient_folder}'), expected exactly 1 image file in format '{image_format_selected}', but found {len(valid_image_files)}.\n"
                    
            if valid_image_files:
                if image_type in ["CR", "DX", "RX"]:
                    if len(valid_image_files) != 1:
                        has_error = True
                        num_image_errors += 1
                        image_errors += f"- Details: In series folder '{expected_series_folder}' (patient '{patient_folder}'), expected exactly 1 image file for image type '{image_type}', but found {len(valid_image_files)}.\n"
            
                # For MRI and CT, when using DICOM, we expect multiple slices (not a single file like .nii)
                elif image_type in ["MRI", "CT"] and image_format_selected == ".dcm":
                    if len(valid_image_files) < 2:
                        has_error = True
                        num_image_errors += 1
                        image_errors += f"- Details: In series folder '{expected_series_folder}' (patient '{patient_folder}'), expected at least 2 image files for image type '{image_type}', but found {len(valid_image_files)}.\n"

            # Safe to count slices now
            try:
                num_slices = count_tot_num_slices_per_group(series_path, image_format_selected)
                num_slices_group += num_slices
            except ValueError as e:
                print(f"Failed to count slices in series folder '{expected_series_folder}' (patient '{patient_folder}'): {str(e)}\n")  

            # Check segmentation folder if segmentations are enabled
            if "segmentation" in protocol_group_specific:
                seg_folders = [f for f in os.listdir(series_path) if os.path.isdir(os.path.join(series_path, f))]

                # Validate segmentation files inside segmentation folder
                seg_folder_path = os.path.join(series_path, seg_folders[0])
                
                # Find all files in the segmentation folder
                seg_files = [f for f in os.listdir(seg_folder_path) if os.path.isfile(os.path.join(seg_folder_path, f))]

                if not seg_files:
                    has_error = True
                    num_segmentation_errors += 1
                    segmentation_errors += f"- Details: Segmentation folder '{seg_folders[0]}' (series '{expected_series_folder}', patient '{patient_folder}') is empty.\n"
                    
                else:
                    # Identify invalid segmentation files (i.e., files not matching the selected format)
                    invalid_seg_files = [f for f in seg_files if not f.endswith(segmentation_format_selected)]
                    if invalid_seg_files:
                        has_error = True
                        num_segmentation_errors += 1
                        segmentation_errors += f"- Details: In segmentation folder '{seg_folders[0]}' (series '{expected_series_folder}', patient '{patient_folder}'), found files with invalid formats: {invalid_seg_files}.\n"
                            

        # Combine both image and segmentation error content into the report
        if image_errors:
            #report_content += "Image Validation Errors:\n"
            report_content += f"Error E2101: {error_descriptions.get('E2101', None)}\n"
            report_content += image_errors + "\n"
        
        if segmentation_errors:
            #report_content += "Segmentation Validation Errors:\n"
            report_content += f"Error E2102: {error_descriptions.get('E2102', None)}\n"
            report_content += segmentation_errors + "\n"

        # Update the total number of errors
        num_errors = num_image_errors + num_segmentation_errors

        error_counts = {
        "E2101": num_image_errors if num_image_errors > 0 else None,
        "E2102": num_segmentation_errors if num_segmentation_errors > 0 else None
        }
        
        total_slices = total_slices - num_slices_group_old + num_slices_group
        # Generate the report and check file, regardless of errors
        generate_check_file_image_data(
            check_name=phase_name,
            phase_number=phase_number,
            series_group_name=series_group,
            num_slices_group=num_slices_group,
            num_patients_with_image_data=total_patients,
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

        if has_error:
            report_filename = os.path.join(self.output_directory_report, report_name)
            host_report_filename = os.path.join(self.host_output_directory_report, report_name)
            report_content = f"Report generated on: {current_datetime}\n\n" + f"{phase_name} report:\n\n" + report_content + f"Total Errors: {num_errors}\n\n" 
            with open(report_filename, "w") as report_file:
                report_file.write(report_content)
            raise Exception(
                f"Input structure validation failed. "
                f"See the detailed report at: {host_report_filename}") #ok

        else:
            success_input_structure = "Input structure validation successful. All checks passed."
            return success_input_structure
        

    def check_labels_per_slice(self): 

        phase_number = 22  
        phase_data = self.mapping_file.get("check_phase", {}).get(str(phase_number), {})
        error_descriptions = phase_data.get("errors", {})
        phase_name = phase_data.get("name", f"Phase {phase_number}")  
        
        # Format check and report names dynamically
        check_name = phase_name.lower().replace(" ", "_") 
        report_name = f"0.022.{check_name}_report"

        check_file_path = os.path.join(self.output_directory_checkfile, "image_data_check_file.json")
        existing_warnings = retrieve_existing_warnings_image_data(check_file_path)

        _, num_slices_group_old, total_series, total_slices = extract_metadata_from_check_file(self.output_directory_checkfile)

        series_group = self.series_group_name
        series_group_mapping = self.local_config["Local config"]["radiological data"]["series_mapping"]
        series_group_subfolder = series_group_mapping.get(series_group) #input name of the series (ex. */serie 1)
        image_group_config = self.protocol.get(series_group, {}).get("image", {}) #equal to the original self.protocol["image"]

        # Image and segmentation format validations
        image_format_selected = image_group_config['image file format']['selected']

        # Extract protocol information
        if "slice_labels" not in image_group_config:
            labels_per_slice_success = "Slice labels are not defined in the protocol. No checks required."
            return labels_per_slice_success
            
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        expected_labels = image_group_config["slice_labels"]["labels"]
        report_content = ""
        report_name = f"{report_name}.txt"
        
        has_error = False  # To track if there are any errors
        num_errors = 0
        num_slices_group = 0
        error_counts = {"E2201": None, "E2202": None, "E2203": None, "E2204": None, "E2205": None}
        
        # -------------- Phase 1: Check CSV presence and number--------------
        # Iterate over patient folders
        patient_folders = [f for f in os.listdir(self.images_dir) if os.path.isdir(os.path.join(self.images_dir, f)) and f != "Reports"]
 
        for patient_folder in patient_folders:
            patient_path = os.path.join(self.images_dir, patient_folder)

            series_folder = os.path.normpath(series_group_subfolder).split(os.sep)[-1].strip()
            series_path = os.path.join(patient_path, series_folder)

            num_slices = count_tot_num_slices_per_group(series_path, image_format_selected)
            num_slices_group += num_slices  # Update total slices
                
            # Find CSV files in the series folder
            csv_files = [f for f in os.listdir(series_path) if f.endswith('.csv')]
            
            # Ensure that exactly one CSV file is present
            if len(csv_files) == 0:
                report_content += f"The CSV file with the labels is not present in series folder '{series_folder}' (patient '{patient_folder}').\n"
                error_counts["E2201"] = 1 if error_counts["E2201"] is None else error_counts["E2201"] + 1
                has_error = True
                num_errors += 1 
            elif len(csv_files) > 1:
                report_content += f"Expected exactly 1 CSV file in series folder '{series_folder}' (patient '{patient_folder}'), but found {len(csv_files)}.\n"
                error_counts["E2201"] = 1 if error_counts["E2201"] is None else error_counts["E2201"] + 1
                has_error = True
                num_errors += 1 

        total_slices = total_slices - num_slices_group_old + num_slices_group
        if has_error:
            report_content += "\n"
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
            report_filename = os.path.join(self.output_directory_report, report_name)
            host_report_filename = os.path.join(self.host_output_directory_report, report_name)

            # Construct the specific error details for E2201
            e2201_details = ""
            # Properly format the details section
            formatted_details = "\n".join(
                f"  - {line}" for line in report_content.strip().split("\n") if line
            )
            e2201_details = (
                f"Error E2201: {error_descriptions.get('E2201', None)}\n"
                f"- Details:\n{formatted_details}"
            )
       
            # Combine everything into the report content
            report_content = (
                f"Report generated on: {current_datetime}\n\n"
                f"{phase_name} report:\n\n"
                f"{e2201_details}\n"
                f"\nTotal Errors: {num_errors}\n\n"
            ) 
            with open(report_filename, "w") as report_file:
                report_file.write(report_content)
            raise Exception(
                f"Slice label validation failed: missing or multiple CSV files detected. "
                f"See the detailed report at: {host_report_filename}") #ok
                
        # -------------- Phase 2: Check CSV structure --------------     
        for patient_folder in patient_folders:
            patient_path = os.path.join(self.images_dir, patient_folder)

            series_folder = os.path.normpath(series_group_subfolder).split(os.sep)[-1].strip()
            series_path = os.path.join(patient_path, series_folder)
                
            # Find CSV files in the series folder
            csv_files = [f for f in os.listdir(series_path) if f.endswith('.csv')]       
            csv_file_path = os.path.join(series_path, csv_files[0])

            # Read the CSV file and check its structure
            try:
                with open(csv_file_path, mode='r') as file:
                    sample = file.read(1024)  # Read a small sample of the file to sniff
                    file.seek(0)  # Reset file pointer to the beginning

                    # Handle empty file explicitly before using the Sniffer
                    if not sample.strip():  # Check if the file is empty
                        report_content += (
                            f"CSV file '{csv_files[0]}' in series folder '{series_folder}' "
                            f"(patient '{patient_folder}') is empty.\n"
                        )
                        error_counts["E2202"] = 1 if error_counts["E2202"] is None else error_counts["E2202"] + 1
                        has_error = True
                        num_errors += 1
                        continue
                        
                    # Check for single column case (assume comma as delimiter)
                    rows = sample.splitlines()  # Get rows from the sample
                    try:
                        # Attempt to detect the delimiter with Sniffer
                        dialect = csv.Sniffer().sniff(sample)
                        detected_delimiter = dialect.delimiter
                    except csv.Error:
                        # If Sniffer fails, try common delimiters
                        for delimiter in [',', ';', '\t', '|']:
                            if all(len(row.split(delimiter)) > 1 for row in rows):  # Check if the delimiter works
                                detected_delimiter = delimiter
                                break
                        else:
                            # Default to a comma if no valid delimiter is found
                            detected_delimiter = ','
                    
                    # Now read the file using the detected delimiter
                    reader = csv.reader(file, delimiter=detected_delimiter)
                    rows = list(reader)
                    
                    # Convert the rows into a DataFrame to easily check column count
                    df = pd.DataFrame(rows)
                    
                    # Check the number of columns in the entire CSV file
                    if df.shape[1] == 1:  # Check if the CSV has only one column
                        report_content += (
                            f"CSV file '{csv_files[0]}' in series folder '{series_folder}' "
                            f"(patient '{patient_folder}') must have exactly 2 columns in each row.\n"
                        )
                        error_counts["E2202"] = 1 if error_counts["E2202"] is None else error_counts["E2202"] + 1
                        has_error = True
                        num_errors += 1
                    elif df.shape[1] != 2:  # Check if the CSV file must have exactly 2 columns
                        report_content += (
                            f"CSV file '{csv_files[0]}' in series folder '{series_folder}' "
                            f"(patient '{patient_folder}') must have exactly 2 columns in each row, but it has "
                            f"{df.shape[1]} columns.\n"
                        )
                        error_counts["E2202"] = 1 if error_counts["E2202"] is None else error_counts["E2202"] + 1
                        has_error = True
                        num_errors += 1
            except Exception as e:  # Handle file read errors
                report_content += f"Could not read CSV file '{csv_files[0]}' in series folder '{series_folder}' (patient '{patient_folder}'). Error: {str(e)}.\n"
                error_counts["E2202"] = 1 if error_counts["E2202"] is None else error_counts["E2202"] + 1
                has_error = True
                num_errors += 1 

        if has_error:
            report_content += "\n"
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
            report_filename = os.path.join(self.output_directory_report, report_name)
            host_report_filename = os.path.join(self.host_output_directory_report, report_name)
            
            # Format report details for E2202
            formatted_details = "\n".join(
                f"  - {line}" for line in report_content.strip().split("\n") if line
            )
            e2202_details = (
                f"Error E2202: {error_descriptions.get('E2202', None)}\n"
                f"- Details:\n{formatted_details}"
            )

            # Combine everything into the final report content
            report_content = (
                f"Report generated on: {current_datetime}\n\n"
                f"{phase_name} report:\n\n"
                f"{e2202_details}\n"
                f"\nTotal Errors: {num_errors}\n\n"
            )

            with open(report_filename, "w") as report_file:
                report_file.write(report_content)
        
            raise Exception(
                f"CSV structure validation failed. "
                f"See the detailed report at: {host_report_filename}") #ok

        # -------------- Phase 3: Check CSV details--------------
        # Initialize a dictionary to group error details by code
        error_details_grouped = {
            "E2203": [],
            "E2204": [],
            "E2205": []
        }

        image_file_format = image_group_config['image file format']['selected']
        
        for patient_folder in patient_folders:
            patient_path = os.path.join(self.images_dir, patient_folder)

            series_folder = os.path.normpath(series_group_subfolder).split(os.sep)[-1].strip()
            series_path = os.path.join(patient_path, series_folder)
                
            # Find CSV files in the series folder
            csv_files = [f for f in os.listdir(series_path) if f.endswith('.csv')]       
            csv_file_path = os.path.join(series_path, csv_files[0])

            # Read the CSV file and check its structure
            with open(csv_file_path, mode='r') as file:
                sample = file.read(1024)  
                file.seek(0) 
                dialect = csv.Sniffer().sniff(sample)  
                
                reader = csv.reader(file, dialect)
                rows = list(reader)
                data_rows = [row for row in rows[1:] if row and not all(cell.strip() == "" for cell in row)]  # Excludes the first row and blank lines

                if image_file_format in [".nii", ".nii.gz", ".nrrd", ".dcm", ".png", ".jpg", ".jpeg", ".tiff"]:
                    # Use the helper function to count slices
                    num_slices = count_tot_num_slices_per_group(series_path, image_file_format)
                    #num_slices_group += num_slices 

                    if len(data_rows) != num_slices:
                        error_details_grouped["E2203"].append(
                            f"The number of rows ({len(data_rows)}) in CSV file '{csv_files[0]}' in series folder '{series_folder}' (patient '{patient_folder}') does not match the number of slices ({num_slices}) in the {image_file_format} file."
                        )
                        error_counts["E2203"] = 1 if error_counts["E2203"] is None else error_counts["E2203"] + 1
                        has_error = True
                        num_errors += 1

                # Check each row: the first column should match a file in the series, and the second should be a valid label
                for row in data_rows:
                        
                    slice_filename, label = row
                    if image_file_format == ".dcm":
                        # Check if the slice filename exists in the series folder
                        slice_filepath = os.path.join(series_path, slice_filename)
                        if not os.path.isfile(slice_filepath):
                            error_details_grouped["E2204"].append(
                                f"The slice file '{slice_filename}' in the CSV file '{csv_files[0]}' (series '{series_folder}', patient '{patient_folder}') does not exist in the folder."
                            )
                            error_counts["E2204"] = 1 if error_counts["E2204"] is None else error_counts["E2204"] + 1
                            has_error = True
                            num_errors += 1 

                    # Check if the label is valid
                    if label not in expected_labels:
                        error_details_grouped["E2205"].append(
                            f"The label '{label}' for the slice '{slice_filename}' in the CSV file '{csv_files[0]}' (series '{series_folder}', patient '{patient_folder}') is invalid. Expected one of {expected_labels}."
                        )
                        error_counts["E2205"] = 1 if error_counts["E2205"] is None else error_counts["E2205"] + 1
                        has_error = True
                        num_errors += 1 
                            

        # Return the validation result
        if has_error:
            report_content += "\n"
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

            report_filename = os.path.join(self.output_directory_report, report_name)
            host_report_filename = os.path.join(self.host_output_directory_report, report_name)

            if error_details_grouped["E2203"]:
                report_content += f"Error E2203: {error_descriptions.get('E2203', None)}\n"
                report_content += "- Details:\n"
                report_content += "\n".join(
                    f"  - {detail}" for detail in error_details_grouped["E2203"]
                ) + "\n\n"
        
            if error_details_grouped["E2204"]:
                report_content += f"Error E2204: {error_descriptions.get('E2204', None)}\n"
                report_content += "- Details:\n"
                report_content += "\n".join(
                    f"  - {detail}" for detail in error_details_grouped["E2204"]
                ) + "\n\n"
        
            if error_details_grouped["E2205"]:
                report_content += f"Error E2205: {error_descriptions.get('E2205', None)}\n"
                report_content += "- Details:\n"
                report_content += "\n".join(
                    f"  - {detail}" for detail in error_details_grouped["E2205"]
                ) + "\n\n"

            # Append total errors
            report_content += f"Total Errors: {num_errors}\n"

            # Write the full report to a file
            with open(report_filename, "w") as report_file:
                report_file.write(
                    f"Report generated on: {current_datetime}\n\n"
                    f"{phase_name} report:\n"
                    f"{report_content}\n"
                )
            raise Exception(
                f"CSV data issues found. "
                f"See the detailed report at: {host_report_filename}") #ok
        else:
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
            labels_per_slice_success = "CSV files with slice labels have been successfully validated."
            return labels_per_slice_success
        

    def check_label_per_patient(self): 
        """
        Validates that for each series folder, exactly one TXT file exists containing the patient label,
        and that the label is one of the allowed labels specified in the protocol under "patient_label".
        
        Returns:
            str: Success message if all patient label files are present and valid.
        
        Raises:
            Exception: If any patient folder does not have exactly one TXT file or if the label is invalid.
        """
        phase_number = 23  
        phase_data = self.mapping_file.get("check_phase", {}).get(str(phase_number), {})
        error_descriptions = phase_data.get("errors", {})
        phase_name = phase_data.get("name", f"Phase {phase_number}")  
        
        # Format check and report names dynamically
        check_name = phase_name.lower().replace(" ", "_") 
        report_name = f"0.023.{check_name}_report"

        check_file_path = os.path.join(self.output_directory_checkfile, "image_data_check_file.json")
        existing_warnings = retrieve_existing_warnings_image_data(check_file_path)

        _, num_slices_group_old, total_series, total_slices = extract_metadata_from_check_file(self.output_directory_checkfile)

        series_group = self.series_group_name
        series_group_mapping = self.local_config["Local config"]["radiological data"]["series_mapping"]
        series_group_subfolder = series_group_mapping.get(series_group) #input name of the series
        image_group_config = self.protocol.get(series_group, {}).get("image", {}) #original self.protocol["image"]

        # Image and segmentation format validations
        image_format_selected = image_group_config['image file format']['selected']
        
        # Extract protocol information
        if "patient_label" not in image_group_config:
            label_per_pts_success = "Patient labels are not defined in the protocol. No checks required."
            return label_per_pts_success

        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        expected_pts_labels = image_group_config["patient_label"].get("labels", [])
        if not expected_pts_labels:
            raise ValueError("No expected patient labels specified under 'patient_label' in the protocol.")
            
        report_content = ""
        report_name = f"{report_name}.txt"
        
        has_error = False  # To track if there are any errors
        num_errors = 0
        num_slices_group = 0
        error_counts = {"E2301": None, "E2302": None}
        
        # -------------- Phase 1: Check TXT presence and number --------------
        # Iterate over patient folders
        patient_folders = [f for f in os.listdir(self.images_dir) if os.path.isdir(os.path.join(self.images_dir, f)) and f != "Reports"]
 
        for patient_folder in patient_folders:
            patient_path = os.path.join(self.images_dir, patient_folder)

            series_folder = os.path.normpath(series_group_subfolder).split(os.sep)[-1].strip()
            series_path = os.path.join(patient_path, series_folder)

            num_slices = count_tot_num_slices_per_group(series_path, image_format_selected)
            num_slices_group += num_slices  # Update total slices

            # Find TXT files in the series folder
            txt_files = [f for f in os.listdir(series_path) if f.endswith('.txt')]
            
            # Ensure that exactly 1 TXT file is present
            if len(txt_files) == 0:
                report_content += f"The TXT file with the patient label is not present in series folder '{series_folder}' (patient '{patient_folder}').\n"
                error_counts["E2301"] = 1 if error_counts["E2301"] is None else error_counts["E2301"] + 1
                has_error = True
                num_errors += 1 
            elif len(txt_files) > 1:
                report_content += f"Expected exactly 1 TXT file in series folder '{series_folder}' (patient '{patient_folder}'), but found {len(txt_files)}.\n"
                error_counts["E2301"] = 1 if error_counts["E2301"] is None else error_counts["E2301"] + 1
                has_error = True
                num_errors += 1 

        total_slices = total_slices - num_slices_group_old + num_slices_group
        if has_error:
            report_content += "\n"
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
            report_filename = os.path.join(self.output_directory_report, report_name)
            host_report_filename = os.path.join(self.host_output_directory_report, report_name)

            # Construct the specific error details for E2101
            e2301_details = ""
            # Properly format the details section
            formatted_details = "\n".join(
                f"  - {line}" for line in report_content.strip().split("\n") if line
            )
            e2301_details = (
                f"Error E2301: {error_descriptions.get('E2301', None)}\n"
                f"- Details:\n{formatted_details}"
            )
       
            # Combine everything into the report content
            report_content = (
                f"Report generated on: {current_datetime}\n\n"
                f"{phase_name} report:\n\n"
                f"{e2301_details}\n"
                f"\nTotal Errors: {num_errors}\n\n"
            ) 
            with open(report_filename, "w") as report_file:
                report_file.write(report_content)
            raise Exception(
                f"Patient label validation failed: missing or multiple TXT files detected. "
                f"See the detailed report at: {host_report_filename}") #ok

        # -------------- Phase 2: Check TXT labels -------------- 
        for patient_folder in patient_folders:
            patient_path = os.path.join(self.images_dir, patient_folder)

            series_folder = os.path.normpath(series_group_subfolder).split(os.sep)[-1].strip()
            series_path = os.path.join(patient_path, series_folder)
                
            # Find TXT files in the series folder
            txt_files = [f for f in os.listdir(series_path) if f.endswith('.txt')]       
            txt_file_path = os.path.join(series_path, txt_files[0])
            
            with open(txt_file_path, "r") as f:
                content = f.read().strip()
                # If the content contains newline characters, indent subsequent lines with four spaces
                indented_content = "\n".join("  " + line for line in content.splitlines())
                
            if content not in expected_pts_labels:
                report_content += (f"Patient '{patient_folder}', Series '{series_folder}': Patient label in file '{txt_files[0]}' is invalid.\n"
                   f"  Content:\n"
                   f"  '''\n{indented_content}\n  '''\n\n")  # Preserves newlines properly
                        
                num_errors += 1
                has_error = True  
                error_counts["E2302"] = 1 if error_counts["E2302"] is None else error_counts["E2302"] + 1
             
        if has_error:
            report_content += "\n"
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
            report_filename = os.path.join(self.output_directory_report, report_name)
            host_report_filename = os.path.join(self.host_output_directory_report, report_name)

            # Construct the specific error details for E2101
            e2302_details = ""
            # Properly format the details section
            formatted_details = "\n".join(
                (f"  - {line}" if line.lstrip().startswith("Patient") else f"  {line}") 
                for line in report_content.strip().split("\n") if line
            )
            e2302_details = (
                f"Error E2302: {error_descriptions.get('E2302', None)}\n"
                f"- Details:\n  Expected labels: {expected_pts_labels}\n{formatted_details}"
            )
       
            # Combine everything into the report content
            report_content = (
                f"Report generated on: {current_datetime}\n\n"
                f"{phase_name} report:\n\n"
                f"{e2302_details}\n"
                f"\nTotal Errors: {num_errors}\n\n"
            ) 
            with open(report_filename, "w") as report_file:
                report_file.write(report_content)
            raise Exception(
                f"Invalid patient label(s) found in TXT file(s). "
                f"See the detailed report at: {host_report_filename}") #ok
        else:
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
            label_per_pts_success = "TXT files with patient labels have been successfully validated."
            return label_per_pts_success
        

    def generate_ImageStructureAndLabelValidator_final_report(self, success_creation, success_input_structure, labels_per_slice_success, label_per_pts_success): #ok
        # Get the current datetime
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")

        # Construct the final report
        final_report_lines = [
        f"Report generated on: {formatted_datetime}",
        "",
        "Final report on patient folder creation and input data compliance:",
        "",
        "Patient folder structure creation:",
        f"- {success_creation}",
        "",
        "Input data check:",
        f"- {success_input_structure}",
        "",
        "Labels per slice check:",
        f"- {labels_per_slice_success}",
        "",
        "Labels per patient check:",
        f"- {label_per_pts_success}",
        "",
        "Summary:",
        "All checks completed."
        ]
    
        final_report = "\n".join(final_report_lines)
            
        # Define the report filename
        report_filename = os.path.join(self.output_directory_report, "0.ImageStructureAndLabelValidator_final_report.txt")
    
        # Write the final report to a file
        with open(report_filename, "w") as report_file:
            report_file.write(final_report)
    
        # Print a message indicating that the final report has been generated
        print("ImageStructureAndLabelValidator final report has been generated.")


def run_imagestructure_and_label_validator(protocol, local_config, mapping_file, series_group_name):
    # Define the input directory path with 
    input_dir = os.getenv("INPUT_DIR")

    # Clear log at the start of validation
    clear_log_file(input_dir, LOG_FILENAME)

    # Load the input validation state to extract num_image_patients
    input_state_file = os.path.join(input_dir, "input_validation_state.json")

    try:
        with open(input_state_file, "r") as f:
            state = json.load(f)
        num_image_patients = state.get("num_patients_with_image_data", 0)
    except Exception as e:
        raise RuntimeError(f"Failed to load state file '{input_state_file}': {e}")
    
    # Create an instance of ImageStructureAndLabelValidator
    input_validator = ImageStructureAndLabelValidator(protocol, local_config=local_config, mapping_file=mapping_file, num_image_patients=num_image_patients, series_group_name=series_group_name)

    # Load the state if it exists
    state = load_state(input_validator.state_file)
    series_progress_state =  load_state(input_validator.series_progress_file)

    # Access or create the per-series progress dictionary
    series_state = series_progress_state.setdefault(series_group_name, {})
    last_phase_done = series_state.get("last_successful_phase", 0)
    started = series_state.get("started_validation", False)

    print("Running Count Tot Num Series Per Dataset function")
    try:
        tot_series = count_tot_num_series_per_dataset(input_dir)
        state["tot_series"] = int(tot_series)
    except Exception as e:
        log_error(input_dir, "count_tot_num_series_per_dataset", e, LOG_FILENAME)
        print(f"An unexpected error occurred during count_tot_num_series_per_dataset. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
        raise 
    
    print("Running Count Tot Num Slices Per Dataset function")
    try:
        tot_slices = count_tot_num_slices_per_dataset(input_dir, protocol, local_config)
        state["tot_slices"] = int(tot_slices)
    except Exception as e:
        log_error(input_dir, "count_tot_num_slices_per_dataset", e, LOG_FILENAME)
        print(f"An unexpected error occurred during count_tot_num_slices_per_dataset. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
        raise

    if not started or last_phase_done < 20:
        print("Running Create Patient Folder Structure function") # check phase 20
        try:
            state["success_creation"] = input_validator.create_patient_folder_structure(tot_series, tot_slices)
            save_state(state, input_validator.state_file)

            # Update series progress state
            series_state["started_validation"] = True
            series_state["last_successful_phase"] = 20
            series_progress_state[series_group_name] = series_state
            save_state(series_progress_state, input_validator.series_progress_file)
        except Exception as e:
            log_error(input_dir, "create_patient_folder_structure", e, LOG_FILENAME)
            print(f"An unexpected error occurred during create_patient_folder_structure. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
            raise

    if last_phase_done < 21:
        print("Running Check Input Structure function") # check phase 21
        try:
            state["success_input_structure"] = input_validator.check_input_structure()
            save_state(state, input_validator.state_file)

            series_state["started_validation"] = True
            series_state["last_successful_phase"] = 21
            series_progress_state[series_group_name] = series_state
            save_state(series_progress_state, input_validator.series_progress_file)
        except Exception as e:
            log_error(input_dir, "check_input_structure", e, LOG_FILENAME)
            print(f"An unexpected error occurred during check_input_structure. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
            raise

    if last_phase_done < 22:
        print("Running Check Labels Per Slice function") # check phase 22
        try:
            state["labels_per_slice_success"] = input_validator.check_labels_per_slice()
            save_state(state, input_validator.state_file)

            series_state["last_successful_phase"] = 22
            series_progress_state[series_group_name] = series_state
            save_state(series_progress_state, input_validator.series_progress_file)
        except Exception as e:
            log_error(input_dir, "check_labels_per_slice", e, LOG_FILENAME)
            print(f"An unexpected error occurred during check_labels_per_slice. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
            raise 

    if last_phase_done < 23:
        print("Running Check Label Per Patient function") # check phase 23
        try:
            state["label_per_pts_success"] = input_validator.check_label_per_patient()
            save_state(state, input_validator.state_file)

            series_state["last_successful_phase"] = 23
            series_progress_state[series_group_name] = series_state
            save_state(series_progress_state, input_validator.series_progress_file)
        except Exception as e:
            log_error(input_dir, "check_label_per_patient", e, LOG_FILENAME)
            print(f"An unexpected error occurred during check_label_per_patient. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
            raise

        print("Running generate_ImageStructureAndLabelValidator_final_report function")
        try:
            input_validator.generate_ImageStructureAndLabelValidator_final_report(
                state.get("success_creation"),
                state.get("success_input_structure"),
                state.get("labels_per_slice_success"),
                state.get("label_per_pts_success")
            )
        except Exception as e:
            log_error(input_dir, "generate_ImageStructureAndLabelValidator_final_report", e, LOG_FILENAME)
            print(f"An unexpected error occurred during generate_ImageStructureAndLabelValidator_final_report. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
            raise 
    

