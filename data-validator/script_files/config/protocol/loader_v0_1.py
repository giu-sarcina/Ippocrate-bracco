import pandas as pd
import numpy as np
import os 
import json
from datetime import datetime

## PROTOCOL CHECK
class ProtocolCheckClass:

     def __init__(self, input_path, protocol_filename):
         self.input_path = input_path
         self.protocol_filename = protocol_filename
         self.protocol_path = os.path.join(self.input_path, self.protocol_filename)
         self.protocol = self.load_protocol()  # Load the protocol during initialization
         
     def load_protocol(self):
        def save_error_report(error_message):
            # Get the current date and time
            current_datetime = datetime.now()
            formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")

            # Define the report filename
            report_filename = f"error_{self.protocol_filename.split('.')[0]}_report.txt"
            report_path = os.path.join(self.input_path, report_filename)

            # Write the error message to the report file
            with open(report_path, 'w') as report_file:
                report_file.write(f"Report generated on: {formatted_datetime}\n\n")
                report_file.write(error_message + "\n\n")

        try:
            with open(self.protocol_path, 'r') as protocol_file:
                protocol = json.load(protocol_file)
            return protocol
        except FileNotFoundError as e:
            error_message = f"Protocol file '{self.protocol_path}' not found."
            print(error_message)
            save_error_report(error_message)
            # Raise an exception to block the code
            raise e
        except json.JSONDecodeError as e:
            error_message = f"Error decoding JSON from protocol file '{self.protocol_path}': {str(e)}"
            print(error_message)
            save_error_report(error_message)
            raise e

     def protocol_check(self):
         # Get the current date and time
         current_datetime = datetime.now()
         formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")   

         # Initialize the report content
         report_content = f"Report generated on: {formatted_datetime}\n\n"
         report_content += "Protocol check results:\n\n"

         # Initialize protocol correctness
         protocol_correctness = True
         
         # 1. Ensure the clinical data section exists
         clinical_data = self.protocol.get("clinical data", {})
         if not clinical_data:
             report_content += "- 'clinical data' section is missing or empty in the protocol.\n"
             protocol_correctness = False
            
         # 2. Ensure that specific keys are present
         if "Missing Values" not in clinical_data:
             report_content += "- Key 'Missing Values' is missing in the protocol.\n"
             protocol_correctness = False
    
         # 3. Ensure exactly one variable has 'ground truth' subkey
         ground_truth_count = 0
         for key, value in clinical_data.items():
             if isinstance(value, dict) and 'ground-truth' in value:
                 ground_truth_count += 1
    
         if ground_truth_count == 0:
             report_content += "- No variable is defined as 'ground-truth'.\n"
             protocol_correctness = False
         elif ground_truth_count > 1:
             report_content += "- More than one variable contains the 'ground-truth' subkey.\n"
             protocol_correctness = False
         
         # 4. Verify the number of expected values matches the cardinality for categorical variables
         categorical_mismatch = {}
         categorical_variables = [key for key, value in clinical_data.items() if isinstance(value, dict) and 'cardinality' in value]
         for variable in categorical_variables:
             expected_values = clinical_data[variable].get("expected_values")
             cardinality = clinical_data[variable].get("cardinality")
             if expected_values is not None and cardinality is not None:
                 if len(expected_values) != cardinality:
                     categorical_mismatch[variable] = (expected_values, cardinality)
    
         if categorical_mismatch:
             variables_str = ", ".join(f"'{var}'" for var in categorical_mismatch.keys())
             report_content += f"- The number of expected values does not match the cardinality for {variables_str}.\n"
             protocol_correctness = False
    
         # 5. Check positions
         positions = [value.get('position') for key, value in clinical_data.items() if isinstance(value, dict) and value.get('position') is not None]
         if positions:
             #if 1 not in positions:
                #report_content += "- Position 1 is missing in the protocol.\n"
                #protocol_correctness = False
            max_position = max(positions)
            missing_positions = [pos for pos in range(1, max_position + 1) if pos not in positions]
            if missing_positions:
                missing_positions_str = ", ".join(str(pos) for pos in missing_positions)
                report_content += f"- Positions missing in the protocol: {missing_positions_str}.\n"
                protocol_correctness = False           
         
         # If all checks passed, add "Check passed" to the report
         if protocol_correctness:
             report_content += "check passed.\n"
    
         # Save the report content to a text file
         report_filename = "protocol_check_report.txt"
         report_path = os.path.join(os.path.dirname(self.protocol_path), report_filename)
         with open(report_path, "w") as report_file:
             report_file.write(report_content)

         # Raise an error if the protocol is incorrect
         if not protocol_correctness:
             raise ValueError(f"Protocol check failed. See the report for details: {report_path}")
        
         return protocol_correctness, "Protocol check report saved as 'protocol_check_report.txt'."

def run_protocol_check():
    input_path = "C:/Users/cristina.iudica/OneDrive - Bracco Imaging SPA/Documenti/ROOT/Protocol" #os.getenv("PROTOCOL_DIR", "/Root/Protocol")
    protocol_filename = "Protocol_NOseg.json"
    protocol_checker = ProtocolCheckClass(input_path, protocol_filename)
    protocol = protocol_checker.load_protocol()
    protocol_checker.protocol_check()
    return protocol

