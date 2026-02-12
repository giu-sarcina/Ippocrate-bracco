import numpy as np
import os 
from datetime import datetime
from PIL import Image
import json
from utils import save_state, load_state, clear_log_file, log_error
from ...helpers import (
    generate_check_file_image_data,
    count_tot_num_slices_per_group,
    retrieve_existing_warnings_image_data,
    extract_metadata_from_check_file
)

LOG_FILENAME = "2D_image_validation_error.log"

class ImageValidator:

    def __init__(self, protocol, local_config, mapping_file, num_image_patients, series_group_name):
        self.protocol = protocol
        self.local_config = local_config
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


    def check_uniform_images(self): 
        """
        Checks for completely uniform images (all-black, all-white, or all-gray) in patient series folders.
        Generates a check file and, if any uniform images are found, an error report.
        """
        phase_number = 34  
        phase_data = self.mapping_file.get("check_phase", {}).get(str(phase_number), {})
        error_descriptions = phase_data.get("errors", {})
        phase_name = phase_data.get("name", f"Phase {phase_number}")  
        
        # Format check and report names dynamically
        check_name = phase_name.lower().replace(" ", "_") 
        report_name = f"1.034.{check_name}_report"

        series_group = self.series_group_name
        series_group_mapping = self.local_config["Local config"]["radiological data"]["series_mapping"]
        series_group_subfolder = series_group_mapping.get(series_group) #input name of the series
        image_group_config = self.protocol.get(series_group, {}).get("image", {}) #original self.protocol["image"]
        image_file_format = image_group_config.get("image file format", {}).get("selected", "unknown").lower()
        image_format_selected = image_group_config['image file format']['selected']

        # Retrieve existing warnings
        check_file_path = os.path.join(self.output_directory_checkfile, "image_data_check_file.json")
        existing_warnings = retrieve_existing_warnings_image_data(check_file_path)

        output_directory_checkfile = self.output_directory_checkfile
        _, num_slices_group_old, total_series, total_slices = extract_metadata_from_check_file(output_directory_checkfile)

        error_details = []  # List to store affected files
        total_affected_images = 0
        num_slices = 0
        num_slices_group = 0

        # Get patient folders (excluding non-directories like "Reports")
        patient_folders = [
            f for f in os.listdir(self.images_dir)
            if os.path.isdir(os.path.join(self.images_dir, f)) and f != "Reports"
        ]
        
        # Iterate through patients
        for patient_folder in patient_folders:
            patient_folder_path = os.path.join(self.images_dir, patient_folder)
    
            if os.path.isdir(patient_folder_path):
                # Iterate through each series folder within the patient folder
                series_folder = os.path.normpath(series_group_subfolder).split(os.sep)[-1].strip()
                series_path = os.path.join(patient_folder_path, series_folder)

                num_slices = count_tot_num_slices_per_group(series_path, image_file_format)
                num_slices_group += num_slices  # Update total slices
    
                # Get image files
                image_files = [
                    f for f in os.listdir(series_path)
                    if f.lower().endswith(image_format_selected)
                ]
    
                for image_file in image_files:
                    image_path = os.path.join(series_path, image_file)
    
                    try:
                        # Open image and convert to numpy array
                        image = Image.open(image_path).convert("L")  # Open the image and convert to grayscale
                        image_array = np.array(image)
    
                        # Check if the image has uniform pixel values (all-black, all-white, or all-gray)
                        if np.std(image_array) == 0:
                            error_details.append((patient_folder, series_folder, image_file))
                            total_affected_images += 1
                    except Exception as e:
                        print(f"Error processing {image_path}: {str(e)}")
  
        total_slices = total_slices - num_slices_group_old + num_slices_group
        
        # Store the current datetime for report generation
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        error_counts={"E3401": total_affected_images if total_affected_images > 0 else None}
        # Generate JSON check file
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
        # If no issues found, no need to generate a report
        if not error_details:
            uniform_image_message = "No uniform (completely black, white, or gray) images detected."
            return uniform_image_message
    
        # Generate report file
        report_path = os.path.join(self.output_directory_report, f"{report_name}.txt")
        host_report_path = os.path.join(self.host_output_directory_report, f"{report_name}.txt")
    
        with open(report_path, "w") as report_file:
            report_file.write(f"Report generated on: {current_datetime}\n\n")
            report_file.write(f"{phase_name} report:\n\n")
            report_file.write(f"Error E3401: {error_descriptions.get('E3401', None)}\n")
            report_file.write("- Details:\n")
    
            for patient, series, image in error_details:
                report_file.write(f"  - Patient: {patient}, Series: {series}, Image: {image}\n")
    
            report_file.write(f"\nTotal Errors: {total_affected_images}\n")
    
        print(f"Uniform image report saved at: {report_path}") 
        raise Exception(
            f"Uniform images detected. "
            f"See the detailed report at: {host_report_path}" #!! Path del container, deve essere quello locale.OK
        )


    def check_image_dimensions(self):
        """Check if images are smaller than the target dimensions specified in the protocol."""
        
        phase_number = 35
        phase_data = self.mapping_file.get("check_phase", {}).get(str(phase_number), {})
        error_descriptions = phase_data.get("errors", {})
        phase_name = phase_data.get("name", f"Phase {phase_number}")
    
        check_name = phase_name.lower().replace(" ", "_")
        report_name = f"1.035.{check_name}_report"

        series_group = self.series_group_name
        series_group_mapping = self.local_config["Local config"]["radiological data"]["series_mapping"]
        series_group_subfolder = series_group_mapping.get(series_group)
        image_group_config = self.protocol.get(series_group, {}).get("image", {})
    
        image_format_selected = image_group_config["image file format"]["selected"]
        target_size = tuple(image_group_config["size"])
        target_rows, target_cols = target_size

        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        check_file_path = os.path.join(self.output_directory_checkfile, "image_data_check_file.json")
        existing_warnings = retrieve_existing_warnings_image_data(check_file_path)

        _, num_slices_group_old, total_series, total_slices = extract_metadata_from_check_file(self.output_directory_checkfile)
    
        smaller_slices_report = []
        all_slices_smaller = True
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
    
            if not os.path.exists(series_path):
                continue
    
            for img_file in os.listdir(series_path):
                if not img_file.lower().endswith(image_format_selected):
                    continue
    
                img_path = os.path.join(series_path, img_file)
    
                try:
                    img = Image.open(img_path)
                    width, height = img.size  # PIL uses (width, height)
    
                    if height < target_rows or width < target_cols:
                        msg = (
                            f"Patient: {patient_folder}, Series: {series_folder}, "
                            f"File: {img_file}, Image size: {height}x{width}"
                        )
                        smaller_slices_report.append(msg)
                    else:
                        all_slices_smaller = False
    
                except Exception as e:
                    raise ValueError(f"Error reading image file {img_path}: {e}")
                    
        total_slices = total_slices - num_slices_group_old + num_slices_group
        error_counts = {"E3501": len(smaller_slices_report) if smaller_slices_report else None}
    
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
    
        # Handle report generation
        if smaller_slices_report:
            report_path = os.path.join(self.output_directory_report, f"{report_name}.txt")
            host_report_path = os.path.join(self.host_output_directory_report, f"{report_name}.txt")

            os.makedirs(self.output_directory_report, exist_ok=True)
    
            with open(report_path, "w") as report_file:
                report_file.write(f"Report generated on: {current_datetime}\n\n")
                report_file.write(f"{phase_name} report:\n\n")
                report_file.write(f"Error E3501: {error_descriptions.get('E3501', 'Image dimensions are smaller than expected.')}\n\n")
    
                if all_slices_smaller:
                    report_file.write(f"- Details: All images are smaller than the target size {target_rows}x{target_cols}. Upscaling is not allowed.\n\n")
                else:
                    report_file.write(f"- Details: The following images are smaller than the target size {target_rows}x{target_cols}:\n\n")
    
                for line in smaller_slices_report:
                    report_file.write("  " + line + "\n")
    
                report_file.write(f"\nTotal Errors: {len(smaller_slices_report)}\n")
    
            if all_slices_smaller:
                raise ValueError(f"All images are smaller than the target size. "
                                 f"See the detailed report at: {host_report_path}") #!! Path del container, deve essere quello locale.OK
            else:
                raise ValueError(f"Some images are smaller than the target size. "
                                 f"See the detailed report at: {host_report_path}") #!! Path del container, deve essere quello locale.OK
    
        # No issues
        image_dim_message = "All images meet or exceed the target dimensions. Resizing operations are permitted."

        return image_dim_message
    

    def generate_ImageValidator_final_report(self, uniform_image_message, image_dim_message):
        # Get the current datetime
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")

        # Construct the final report
        final_report_lines = [
        f"Report generated on: {formatted_datetime}",
        "",
        "Image Validation final report:",
        "",
        "Uniform image check:",
        f"- {uniform_image_message}",
        "",
        "Image dimensions check:",
        f"- {image_dim_message}",
        "",
        "Summary:",
        f"All checks completed."
        ]
    
        final_report = "\n".join(final_report_lines)
            
        # Define the report filename
        report_filename = os.path.join(self.output_directory_report, "1.ImageValidation_final_report.txt") 
    
        # Write the final report to a file
        with open(report_filename, "w") as report_file:
            report_file.write(final_report)
    
        # Print a message indicating that the final report has been generated
        print("ImageValidation final report has been generated.")


def run_image_validator(protocol, local_config, mapping_file, series_group_name):
    """
    Runs validation for standard 2D images (e.g., PNG, JPG, TIFF).
    """
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
    
    # Create an instance of ImageValidator
    image_validator = ImageValidator(protocol, local_config, mapping_file, num_image_patients, series_group_name)

    state = load_state(image_validator.state_file)
    series_progress_state =  load_state(image_validator.series_progress_file)

    # Access or create the per-series progress dictionary
    series_state = series_progress_state.setdefault(series_group_name, {})
    last_phase_done = series_state.get("last_successful_phase", 0)

    if last_phase_done < 34:
        print("Running check_uniform_images function...") # check phase 34
        try:
            state["uniform_image_message"] = image_validator.check_uniform_images()
            save_state(state, image_validator.state_file)

            series_state["last_successful_phase"] = 34
            series_progress_state[series_group_name] = series_state
            save_state(series_progress_state, image_validator.series_progress_file)
        except Exception as e:
            log_error(input_dir, "check_uniform_images", e, LOG_FILENAME)
            print(f"An unexpected error occurred during check_uniform_images. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
            raise  

    if last_phase_done < 35:
        print("Running check_image_dimensions function...") # check phase 35
        try:
            state["image_dim_message"] = image_validator.check_image_dimensions()
            save_state(state, image_validator.state_file)

            series_state["last_successful_phase"] = 35
            series_progress_state[series_group_name] = series_state
            save_state(series_progress_state, image_validator.series_progress_file)
        except Exception as e:
            log_error(input_dir, "check_image_dimensions", e, LOG_FILENAME)
            print(f"An unexpected error occurred during check_image_dimensions. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
            raise 

        print("Running generate_ImageValidator_final_report function...")
        try:
            image_validator.generate_ImageValidator_final_report(
                state.get("uniform_image_message"), 
                state.get("image_dim_message")
            )
        except Exception as e:
            log_error(input_dir, "generate_ImageValidator_final_report", e, LOG_FILENAME)
            print(f"An unexpected error occurred during generate_ImageValidator_final_report. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
            raise  

