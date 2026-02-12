import os 
from datetime import datetime
import json
import nrrd
from utils import save_state, load_state, clear_log_file, log_error
from ...helpers import (
    generate_check_file_image_data,
    count_tot_num_slices_per_group,
    retrieve_existing_warnings_image_data,
    extract_metadata_from_check_file
)

LOG_FILENAME = "nrrd_validation_error.log"

class NrrdValidator:

    def __init__(self, protocol, local_config, mapping_file, num_image_patients, series_group_name):
        self.protocol = protocol
        self.local_config = local_config
        self.mapping_file = mapping_file
        self.series_group_name = series_group_name
        self.num_image_patients = num_image_patients
        self.input_dir = os.getenv("INPUT_DIR")
        self.host_input_dir = os.getenv("HOST_BASE_DATA_DIR")
        self.images_dir = os.path.join(self.input_dir, "IMAGES")
        self.state_file = os.path.join(self.input_dir, "image_validation_state.json")
        self.series_progress_file = os.path.join(self.input_dir, "validation_progress_by_series.json")
        self.study_path = os.getenv("ROOT_NAME")
        self.output_directory_report = os.path.join(self.input_dir, "Reports", f"Image Data Reports {self.series_group_name}")
        self.host_output_directory_report = os.path.join(self.host_input_dir, "Reports", f"Image Data Reports {self.series_group_name}")
        self.output_directory_checkfile = os.path.join(self.study_path, "CHECKFILE")
        os.makedirs(self.output_directory_report, exist_ok=True)  
        os.makedirs(self.output_directory_checkfile, exist_ok=True)


    def check_slice_dimensions_nrrd(self):
        """Check if NIfTI slices are smaller than the target dimensions specified in the protocol."""

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
        image_file_format = image_group_config['image file format']['selected'].lower()
    
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
    
            # Only .nrrd files
            nrrd_files = [f for f in os.listdir(series_path) if f.endswith(".nrrd")]
    
            for nrrd_file in nrrd_files:
                nrrd_path = os.path.join(series_path, nrrd_file)
                try:
                    data, header = nrrd.read(nrrd_path)
                    num_slices = data.shape[2] if len(data.shape) >= 3 else 1
    
                    for i in range(num_slices):
                        slice_data = data[:, :, i]
                        rows, cols = slice_data.shape
                        total_slices_group += 1
    
                        if rows < target_rows or cols < target_cols:
                            msg = (
                                f"Patient: {patient_folder}, Series: {series_folder}, "
                                f"File: {nrrd_file}, Slice {i}: {rows}x{cols}"
                            )
                            smaller_slices_report.append(msg)
                        else:
                            all_slices_smaller = False
    
                except Exception as e:
                    raise ValueError(f"Error reading NRRD file {nrrd_path}: {e}") #!! Path del container

        total_slices = total_slices - num_slices_group_old + num_slices_group
        error_counts = {"E3501": len(smaller_slices_report) if smaller_slices_report else None}
    
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
                                 f"See the detailed report at: {host_report_path}") #!! Path del container, deve essere quello locale.ok
            else:
                raise ValueError(f"Some slices are smaller than the target size. "
                                 f"See the detailed report at: {host_report_path}") #!! Path del container, deve essere quello locale.ok
    
        slice_dim_message += "All slices meet or exceed the target dimensions. Resizing operations are permitted."
        print(slice_dim_message)
        return slice_dim_message
    

    def generate_NrrdValidation_final_report(self, slice_dim_message):
        # Get the current datetime
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")

        # Construct the final report
        final_report_lines = [
        f"Report generated on: {formatted_datetime}",
        "",
        "NRRD compliance final report:",
        "",
        "Slice dimensions check:",
        f"- {slice_dim_message}",
        "",
        "Summary:",
        "All checks completed."
        ]

        final_report = "\n".join(final_report_lines)
            
        # Define the report filename
        report_filename = os.path.join(self.output_directory_report, "1.NrrdValidation_final_report.txt") 
    
        # Write the final report to a file
        with open(report_filename, "w") as report_file:
            report_file.write(final_report)
    
        # Print a message indicating that the final report has been generated
        print("NrrdValidation final report has been generated.")


def run_nrrd_validator(protocol, local_config, mapping_file, series_group_name):
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
    
    # Create an instance of NrrdValidator
    nrrd_validator = NrrdValidator(protocol, local_config, mapping_file, num_image_patients, series_group_name)

    state = load_state(nrrd_validator.state_file)
    series_progress_state =  load_state(nrrd_validator.series_progress_file)

    # Access or create the per-series progress dictionary
    series_state = series_progress_state.setdefault(series_group_name, {})
    last_phase_done = series_state.get("last_successful_phase", 0)

    if last_phase_done < 35:
        print("Running check_slice_dimensions_nrrd function...") # check phase 35
        try:
            state["slice_dim_message"] = nrrd_validator.check_slice_dimensions_nrrd()
            save_state(state, nrrd_validator.state_file)  # Save updated state

            series_state["last_successful_phase"] = 35
            series_progress_state[series_group_name] = series_state
            save_state(series_progress_state, nrrd_validator.series_progress_file)
        except Exception as e:
            log_error(input_dir, "check_slice_dimensions_nrrd", e, LOG_FILENAME)
            print(f"An unexpected error occurred during check_slice_dimensions_nrrd. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
            raise 

        print("Running generate_NrrdValidation_final_report function...")
        try:
            nrrd_validator.generate_NrrdValidation_final_report(state.get("slice_dim_message"))
        except Exception as e:
            log_error(input_dir, "generate_NrrdValidation_final_report", e, LOG_FILENAME)
            print(f"An unexpected error occurred during generate_NrrdValidation_final_report. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
            raise 
    

        



