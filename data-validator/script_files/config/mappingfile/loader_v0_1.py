import os 
import json
from datetime import datetime

# MAPPING FILE LOADING AND CHECK
class CheckPhaseMappingLoader:
    def __init__(self, input_path, mapping_filename):
        self.input_path = input_path
        self.mapping_filename = mapping_filename
        self.mapping_path = os.path.join(self.input_path, self.mapping_filename)

    def load_mapping(self):
        def save_error_report(error_message):
            # Get the current date and time
            current_datetime = datetime.now()
            formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")

            # Define the report filename
            report_filename = f"loading_error_mappingfile_report.txt"
            report_path = os.path.join(self.input_path, report_filename)

            # Write the error message to the report file
            with open(report_path, 'w') as report_file:
                report_file.write(f"Report generated on: {formatted_datetime}\n\n")
                report_file.write(error_message + "\n\n")

        try:
            with open(self.mapping_path, 'r') as mapping_file:
                mapping = json.load(mapping_file)
            return mapping
        except FileNotFoundError:
            error_message = f"Mapping file '{self.mapping_path}' not found."
            print(error_message)
            save_error_report(error_message)
            raise FileNotFoundError(error_message)
        except json.JSONDecodeError as e:
            error_message = f"Error decoding JSON from mapping file '{self.mapping_path}': {str(e)}."
            print(error_message)
            save_error_report(error_message)
            raise json.JSONDecodeError(msg=error_message, doc=e.doc, pos=e.pos)
        
        
def run_mappingfile_load_and_check():
    mapping_path = os.getenv("MAPPINGFILE_DIR")  #os.getenv("MAPPINGFILE_DIR", "/root/MAPPINGFILE") #"C:/Users/cristina.iudica/OneDrive - Bracco Imaging SPA/Documenti/ROOT_x-rays_CLIENT_2/MAPPINGFILE"
    mapping_filename = "check_phases_mapping_v.0.6.json"
    mapping_loader = CheckPhaseMappingLoader(input_path=mapping_path, mapping_filename=mapping_filename)
    mapping_file = mapping_loader.load_mapping()
    return mapping_file
