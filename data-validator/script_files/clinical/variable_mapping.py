import os
import json
from utils import clear_log_file, log_error

LOG_FILENAME = "variable_mapping_creation_error.log"

class VariableMappingCreation:

    def __init__(self, protocol):
        self.protocol = protocol
        self.clinical_data = self.protocol.get("clinical data", {})
        self.study_path = os.getenv("ROOT_NAME")


    def create_variable_mapping(self):
        mapping = {}

        for variable, details in self.clinical_data.items():
            # Exclude "Patient ID"
            if variable == "Patient ID":
                continue
            
            position = details.get("position")
            expected_values = details.get("expected_values")

            if position is not None:
                variable_code = f"V{position:03d}"
                mapping[variable] = {"variable_code": variable_code}

                if expected_values:
                    levels = {value: f"{variable_code}.{i+1:02d}" for i, value in enumerate(expected_values)}
                    mapping[variable]["levels"] = levels

        # Create folder if it does not exist
        mapping_folder = os.path.join(self.study_path, "MappingFile")
        os.makedirs(mapping_folder, exist_ok=True)

        # Save mapping to JSON file
        mapping_file = os.path.join(mapping_folder, "variables_mapping.json")
        with open(mapping_file, "w") as json_file:
            json.dump(mapping, json_file, indent=4)

        print(f"Mapping file created at: {mapping_file}")


def run_variable_mapping_creation(protocol):

    # Define the input directory path 
    input_dir = os.getenv("INPUT_DIR")
    if not input_dir:
        raise EnvironmentError("INPUT_DIR environment variable not set")

    # Clear log 
    clear_log_file(input_dir, LOG_FILENAME)

    # Get the study path from the local configuration
    study_path = os.getenv("ROOT_NAME")
    if not study_path:
        raise EnvironmentError("ROOT_NAME environment variable not set")

    # Define the expected path for the variable mapping file
    mapping_file_path = os.path.join(study_path, "MappingFile", "variables_mapping.json")

    # Check if the mapping file already exists
    if os.path.exists(mapping_file_path):
        print(f"Mapping file already exists at: {mapping_file_path}. Skipping creation.")
        return # Skip creation if the file is already present
    
    # Create a new variable mapping since the file doesn't exist
    vmc = VariableMappingCreation(protocol)
    print("Running create_variable_mapping function...")
    try:
        vmc.create_variable_mapping()
    except Exception as e:
        log_error(input_dir, "create_variable_mapping", e, LOG_FILENAME)
        print(f"An unexpected error occurred during create_variable_mapping. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
        raise  