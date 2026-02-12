import os 
import json
from datetime import datetime
from utils import clear_log_file, log_error

LOG_FILENAME = "local_config_validation_error.log"

## CONFIG FILE LOADING AND CHECK
class LocalConfigCheckClass:

        def __init__(self, config_path, config_filename):
            self.config_path = config_path
            self.config_filename = config_filename
            self.config_file_path = os.path.join(self.config_path, self.config_filename)
            self.config_data = self.load_config()  # Load the config during initialization

        def load_config(self):
            def save_error_report(error_message):
                current_datetime = datetime.now()
                formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
                report_filename = f"loading_error_local_config_report.txt"
                report_path = os.path.join(self.config_path, report_filename)
                with open(report_path, 'w') as report_file:
                    report_file.write(f"Report generated on: {formatted_datetime}\n\n")
                    report_file.write(error_message + "\n\n")
    
            try:
                with open(self.config_file_path, 'r') as config_file:
                    config_data = json.load(config_file)
                return config_data
                
            except FileNotFoundError as e:
                error_message = f"Config file '{self.config_file_path}' not found."
                print(error_message)
                save_error_report(error_message)
                raise e
            except json.JSONDecodeError as e:
                error_message = f"Error decoding JSON from config file '{self.config_file_path}': {str(e)}"
                print(error_message)
                save_error_report(error_message)
                raise e
        
        def verify_csv_file(self, config_data):
            file_name = config_data['Local config']['clinical data']['file name']
            if not file_name.lower().endswith(".csv"):
                current_datetime = datetime.now()
                formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
                report_filename = f"local_config_check_report.txt"
                report_path = os.path.join(self.config_path, report_filename)
                
                report_message = f"Error - the file specified in the local config '{self.config_filename}' is not a CSV file."
                print(f"The file specified in the local config '{self.config_filename}' is not a CSV file.")
                with open(report_path, 'w') as report_file:
                    report_file.write(f"Report generated on: {formatted_datetime}\n\n")
                    report_file.write(report_message + "\n\n")

                # Raise an error if the file is not a CSV file
                raise ValueError(report_message)
            else:
                print(f"The file specified in the local config '{self.config_filename}' is a CSV file.")


def run_config_load_and_check():
    config_path = os.getenv("CONFIG_DIR")  #"C:/Users/cristina.iudica/OneDrive - Bracco Imaging SPA/Documenti/ROOT_x-rays_CLIENT_2/LOCALCONFIG"
    config_filename = "Local_config.json"
    local_config_checker = LocalConfigCheckClass(config_path, config_filename)
    local_config = local_config_checker.load_config()

    # Define the input data directory path with 
    input_dir = os.getenv("INPUT_DIR")

    # Clear log at the start of validation
    clear_log_file(input_dir, LOG_FILENAME)

    # Only run verify_csv_file if 'clinical data' key is present
    if "clinical data" in local_config.get("Local config", {}):
        try:
            local_config_checker.verify_csv_file(local_config)
        except Exception as e:
            log_error(input_dir, "verify_csv_file", e, LOG_FILENAME)
            print(f"An unexpected error occurred during verify_csv_file. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
            raise
    else:
        print("Skipping CSV file check: 'clinical data' key not found in local config.")
        
    return local_config
                