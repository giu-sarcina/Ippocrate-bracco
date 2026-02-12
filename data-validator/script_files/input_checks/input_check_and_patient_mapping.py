import os
import pandas as pd
from datetime import datetime
import random
import string
import json
from collections import defaultdict
from collections import OrderedDict
from utils import read_csv_file, read_csv_file_two_layer_label, save_state, load_state, clear_log_file, log_error

LOG_FILENAME = "input_check_and_patient_mapping_error.log"

class InputCheckAndPatientMapping:
    
    def __init__(self, protocol, local_config, mapping_file):
        self.protocol = protocol
        self.local_config = local_config
        self.mapping_file = mapping_file
        self.study_path = os.getenv("ROOT_NAME")
        self.input_dir = os.getenv("INPUT_DIR")
        self.host_input_dir = os.getenv("HOST_BASE_DATA_DIR")
        self.clinical_dir = os.path.join(self.input_dir, "CLINICAL")
        self.images_dir = os.path.join(self.input_dir, "IMAGES")
        self.genomics_dir = os.path.join(self.input_dir, "GENOMICS")
        self.state_file = os.path.join(self.input_dir, "input_validation_state.json")
        self.patients_list_filename = self.local_config["Local config"]["patients list"]
        self.patients_list_path = os.path.join(self.input_dir, self.patients_list_filename)
        self.host_patients_list_path = os.path.join(self.host_input_dir, self.patients_list_filename)
        self.output_directory_report = os.path.join(self.input_dir, "Reports", "Input Check Report")
        self.host_output_directory_report = os.path.join(self.host_input_dir, "Reports", "Input Check Report")
        self.output_directory_checkfile = os.path.join(self.study_path, "CHECKFILE")
        os.makedirs(self.output_directory_report, exist_ok=True)
        os.makedirs(self.output_directory_checkfile, exist_ok=True)
        

    def force_lowercase_extensions(self):  # importante per Windows soprattutto
        """
        Recursively walks through self.input_dir and renames files 
        with uppercase extensions to lowercase. Handles case-only renames on Windows.
        """
        for root, dirs, files in os.walk(self.input_dir):
            for filename in files:
                name, ext = os.path.splitext(filename)
                if ext and ext != ext.lower():
                    old_path = os.path.join(root, filename)
                    new_filename = name + ext.lower()
                    new_path = os.path.join(root, new_filename)
    
                    # Case-insensitive match but not exactly same file
                    if os.path.exists(new_path) and old_path.lower() != new_path.lower():
                        print(f"Skipped (file exists with different name): {new_filename}")
                        continue
    
                    try:
                        if old_path.lower() == new_path.lower():
                            # Windows case-only rename workaround
                            temp_path = os.path.join(root, name + "_temp" + ext.lower())
                            os.rename(old_path, temp_path)
                            os.rename(temp_path, new_path)
                        else:
                            os.rename(old_path, new_path)
    
                        print(f"Renamed: {filename} --> {new_filename}")
                    except Exception as e:
                        print(f"Error renaming {filename}: {e}")
        

    def generate_check_file_input_data(
            self,
            num_patients_with_clinical_data,
            num_patients_with_image_data,
            num_patients_with_genomic_data,
            error_counts,
            warning_counts=None, 
            relevant_error_codes=None
        ):
        """
        Generates a JSON check file for input data validation.
    
        Args:
            num_patients_with_clinical_data (int): Total number of patients with clinical data available.
            num_patients_with_image_data (int): Total number of patients with imaging data available.
            num_patients_with_genomic_data (int): Total number of patients with genomic data available.
            error_counts (dict): Dictionary of error codes and their counts.
            warning_counts (dict, optional): Dictionary of warning codes and their counts.
            relevant_error_codes (list, optional): List of error codes to include (subset of all errors).
            
        Returns:
            None: The function writes the check file to the specified output directory.
            
        """   
        phase_number = 0
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        output_dir = self.output_directory_checkfile
        check_file_path = os.path.join(output_dir, "input_check_file.json")

        # Get phase info from mapping file
        phase_data = self.mapping_file.get("check_phase", {}).get(str(phase_number), {})
        check_name = phase_data.get("name", f"Phase {phase_number}")
    
        # Get error descriptions from mapping
        error_descriptions = phase_data.get("errors", {})
        warning_descriptions = phase_data.get("warnings", {})

        # Errors to be stored as "presence": "yes"/"no"
        binary_errors = {"E002", "E003", "E004", "E006", "E007"}

        # ------------------------------
        # Load existing check file (if any)
        # ------------------------------
        existing_errors_map = {}
        existing_warnings_map = {}
    
        if os.path.exists(check_file_path):
            with open(check_file_path, "r") as f:
                existing_data = json.load(f)
            for err in existing_data.get("errors", []):
                existing_errors_map[err["code"]] = err
            existing_warnings = existing_data.get("warnings")
            if existing_warnings is not None:
                for warn in existing_warnings:
                    existing_warnings_map[warn["code"]] = warn   

        # ------------------------------
        # Determine codes to display for this run
        # ------------------------------
        if relevant_error_codes is None:
            codes_to_display = error_counts.keys()
        else:
            codes_to_display = relevant_error_codes

        # ------------------------------
        # Build NEW error entries, avoid duplicates
        # ------------------------------
        
        # Build NEW error entries, update existing ones if present
        errors_to_add = []
        for code in codes_to_display:
            description = error_descriptions.get(code, "Unknown error code.")
            count = error_counts.get(code, 0)
            
            entry = None
            if code in binary_errors:
                entry = {
                    "code": code,
                    "present": "yes" if count > 0 else "no",
                    "description": description
                }
            else:
                entry = {
                    "code": code,
                    "count": int(count) if count > 0 else None,
                    "description": description
                }
            
            # Update existing or append new
            if code in existing_errors_map:
                existing_errors_map[code] = entry
            else:
                errors_to_add.append(entry)
        
        # Merge old (updated) errors with new
        errors = list(existing_errors_map.values()) + errors_to_add
    
        # ------------------------------
        # Build warnings (same as before)
        # ------------------------------
        warnings = []
        if warning_counts:
            for code, count in warning_counts.items():
                if code in existing_warnings_map:
                    continue  # avoid duplicates
                description = warning_descriptions.get(code, "Unknown warning code.")
                entry = {
                    "code": code,
                    "count": int(count) if count > 0 else None,
                    "description": description
                }
                warnings.append(entry)
            # merge old + new
            warnings = list(existing_warnings_map.values()) + warnings
    
        if not warnings:
            warnings = None
            
        # ------------------------------
        # Build final check file JSON
        # ------------------------------
        check_file_data = {
            "name": check_name,
            "phase_number": phase_number,
            "metadata": {
                "total_patients_with_clinical_data": int(num_patients_with_clinical_data),
                "total_patients_with_image_data": int(num_patients_with_image_data),
                "total_patients_with_genomic_data": int(num_patients_with_genomic_data),
                "timestamp": current_datetime
            },
            "errors": errors,
            "warnings": warnings 
        }
    
        # Write/update the file
        with open(check_file_path, "w") as json_file:
            json.dump(check_file_data, json_file, indent=4)
            
    
    def generate_input_validation_report(self, error_log, num_errors, warning_log=None, num_warnings=0):
        """
        Generates an error and warning report based on the current logs.
        Errors are treated as critical; warnings are informational.
        """       
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")

        # Get phase info from mapping_file
        phase_number=0
        phase_data = self.mapping_file.get("check_phase", {}).get(str(phase_number), {})
        check_name = phase_data.get("name", f"Phase {phase_number}")  # fallback if name is missing
        
        # Format name for title and filename
        check_title = f"{check_name} report"
        filename = check_name.lower().replace(" ", "_")
        report_filename = f"{phase_number}.0.{filename}_report.txt"
        report_content = f"Report generated on: {formatted_datetime}\n\n"
        report_content += f"{check_title}:\n" 

        # Add errors
        if error_log and len(error_log) > 0:
            report_content += "\n".join(error_log) + f"\n\nTotal Errors: {num_errors}\n"

        # Add warnings
        if warning_log and len(warning_log) > 0:
            report_content += "\n".join(warning_log) + f"\nTotal Warnings: {num_warnings}"
  
        # Write report to file
        report_filename = os.path.join(self.output_directory_report, report_filename) ## ADD the LOCAL report_path. This is for the container
        host_report_filename = os.path.join(self.host_output_directory_report, report_filename) # Local path
        with open(report_filename, "w") as report_file:
            report_file.write(report_content)
        return report_filename, host_report_filename

           
    def check_input_data_structure(self): 
        """
        Checks the structure of input data folders (clinical, images, genomics).
        Generates a check file always. If errors exist, generates a report and raises RuntimeError.
        """
        # Initialize error tracking
        error_log = []
        error_counts = defaultdict(int)
    
        phase_number = 0
        phase_data = self.mapping_file.get("check_phase", {}).get(str(phase_number), {})
        error_descriptions = phase_data.get("errors", {})

        expected = {
        "clinical": "clinical data" in self.protocol,
        "images": any(k.startswith("series") for k in self.protocol),
        "genomics": "genomic data" in self.protocol
        }

        # Output counters
        num_clin = 0
        num_img = 0
        num_gen = 0
          
        # ---- CLINICAL ----
        if expected["clinical"]:
            if not os.path.isdir(self.clinical_dir):

                if error_counts["E001"] == 0:
                    error_log.append(f"\nError E001: {error_descriptions.get('E001', None)}")
                    
                error_log.append(f"- Details: Expected folder '{self.clinical_dir}' not found.") # !metto percorso locale, non del container
                error_counts["E001"] += 1
            else:
                clinical_filename = self.local_config["Local config"]["clinical data"]["file name"]
                clinical_path = os.path.join(self.clinical_dir, clinical_filename)   
                
                if not os.path.exists(clinical_path):
                    error_log.append(f"\nError E002: {error_descriptions.get('E002', None)}")
                    error_log.append(f"- Details: Missing clinical file '{clinical_filename}'.")
                    error_counts["E002"] += 1
                else:
                    # Load the CSV
                    clinical_df = read_csv_file_two_layer_label(clinical_path)
                    # Count patients = number of rows
                    num_clin = len(clinical_df) # rough measurement              
                
        # --- IMAGES ---
        if expected["images"]:
            if not os.path.isdir(self.images_dir):

                if error_counts["E001"] == 0:
                    error_log.append(f"\nError E001: {error_descriptions.get('E001', None)}")
                    
                error_log.append(f"- Details: Expected folder '{self.images_dir}' not found.") # !metto percorso locale, non del container
                error_counts["E001"] += 1
            else:
                # Count the number of patient subfolders
                subfolders = [
                    f for f in os.listdir(self.images_dir)
                    if os.path.isdir(os.path.join(self.images_dir, f))
                ]
                num_img = len(subfolders) # rough measurement 

        # --- GENOMICS ---
        if expected["genomics"]:
            if not os.path.isdir(self.genomics_dir):

                if error_counts["E001"] == 0:
                    error_log.append(f"\nError E001: {error_descriptions.get('E001', None)}")

                error_log.append(f"- Details: Expected folder '{self.genomics_dir}' not found.") # !metto percorso locale, non del container
                error_counts["E001"] += 1
            else:
                mapping_name = self.local_config["Local config"]["genomic data"]["patient-sample mapping file"]
                mapping_path = os.path.join(self.genomics_dir, mapping_name)
    
                # Check mapping.csv
                if not os.path.exists(mapping_path):
                    error_log.append(f"\nError E006: {error_descriptions.get('E006', None)}")
                    error_log.append(f"- Details: Missing genomic mapping file '{mapping_name}'.")
                    error_counts["E006"] += 1
                else:
                    mapping_df = pd.read_csv(mapping_path)
                    if "patient_id" in mapping_df.columns:
                        num_gen = mapping_df["patient_id"].astype(str).nunique()
                    else:
                        error_log.append(f"\nError E006: {error_descriptions.get('E006', None)}")
                        error_log.append(f"- Details: Mapping file '{mapping_name}' does not contain required column 'patient_id'.")
                        error_counts["E006"] += 1
                    

                # Check at least one sample*.json
                jsons = [f for f in os.listdir(self.genomics_dir) if f.startswith("sample") and f.endswith(".json")]
                if len(jsons) == 0:
                    if error_counts["E006"] == 0:
                        error_log.append(f"\nError E006: {error_descriptions.get('E006', None)}")
                    error_log.append(f"- Details: No genomic sample_*.json files found.")
                    error_counts["E006"] += 1

        # -----------------------------------------------------------------------
        # ------------------ GENERATE FILES + HANDLE ERRORS ---------------------
        # -----------------------------------------------------------------------
    
        # 1) ALWAYS generate the check file
        self.generate_check_file_input_data(
            num_patients_with_clinical_data=num_clin,
            num_patients_with_image_data=num_img,
            num_patients_with_genomic_data=num_gen,
            error_counts=error_counts,
            warning_counts=None,
            relevant_error_codes=["E001", "E002", "E006"]
        )

        # 2) If any error → create report + raise
        total_errors = sum(error_counts.values())
        if total_errors > 0:
            _, host_report_filename = self.generate_input_validation_report(error_log, total_errors)
    
            # STOP execution
            raise RuntimeError(
                f"Input Data Check FAILED with {total_errors} errors. "
                f"See the detailed report at {host_report_filename}." # change it (local path). ok
            )
    
        # 3) If no errors → success message
        input_data_structure_succ_msg = "Input data structure is valid. No structural errors found."
    
        return num_clin, num_img, num_gen, input_data_structure_succ_msg
    
    
    def validate_clinical_data(self, num_clin, num_img, num_gen): #2) FUNZIONE NUOVA. DA RUNNARE SOLO SE "clinical data" in self.protocol
        """
        Validates clinical data structure and header integrity.
        Always generates the check file. If errors are present, also generates
        an error report and raises RuntimeError.
        Returns a success message if no errors.
        """
        # ---------------------
        # INITIALIZATION
        # ---------------------
        error_log = []
        error_counts = defaultdict(int)
        
        phase_number = 0
        phase_data = self.mapping_file.get("check_phase", {}).get(str(phase_number), {})
        error_descriptions = phase_data.get("errors", {})

        clinical_filename = self.local_config["Local config"]["clinical data"]["file name"]
        clinical_path = os.path.join(self.clinical_dir, clinical_filename)
        clinical_df = read_csv_file_two_layer_label(clinical_path) #!TOLGO IL self.
                
        # ============================================================
        # 1) CHECK MULTI-HEADER STRUCTURE → E003
        # ============================================================
        if isinstance(clinical_df.columns, pd.MultiIndex):
            level0_cols = [c[0] for c in clinical_df.columns]
            level1_cols = [c[1] for c in clinical_df.columns]
    
            # Heuristic: level1 should be either empty/None or string representing units od measure
            # We'll consider an error if any value looks like a numeric data point (int/float)
            if any(isinstance(x, (int, float)) or (isinstance(x, str) and x.strip().replace(".", "", 1).isdigit()) for x in level1_cols):
                error_log.append(f"\nError E003: {error_descriptions.get('E003', None)}")
                error_log.append("- Details: Second level of header should contain units of measures or be empty. Found numeric data instead.")
                error_counts["E003"] += 1
        else:
            # If not a MultiIndex at all, also trigger E003
            error_log.append(f"\nError E003: {error_descriptions.get('E003', None)}")
            error_log.append("- Details: Expected a two-level column header, found single-level.")
            error_counts["E003"] += 1

        # ---------- FINALIZATION (inline) ----------
        total_errors = sum(error_counts.values())
        if total_errors > 0:
            
            relevant_error_codes = ["E003", "E004"]

            # ALWAYS generate check file
            self.generate_check_file_input_data(
                num_patients_with_clinical_data=num_clin,
                num_patients_with_image_data=num_img,
                num_patients_with_genomic_data=num_gen,
                error_counts=error_counts,
                warning_counts=None,
                relevant_error_codes=relevant_error_codes
            )
            _, host_report_filename = self.generate_input_validation_report(error_log, total_errors)
            raise RuntimeError(
                f"Clinical Data Check FAILED with {total_errors} errors. "
                f"See detailed report at {host_report_filename}." #! Adapt to LOCAL path, not container. ok
            )
        # -------------------------------------------
        
        # ============================================================
        # 2) PATIENT ID COLUMN DETECTION → E004
        # ============================================================
        # Normalize DataFrame columns to lowercase (and replace underscores with spaces)
        normalized_columns = {col.strip().lower().replace("_", "").replace(" ", ""): col for col in level0_cols}

        # Case-insensitive ID detection
        possible = ["patientid", "name", "patientname"]

        # Find the first valid patient ID column (case-insensitive)
        patient_id_col = next((normalized_columns[candidate] for candidate in possible if candidate in normalized_columns), None)

        patient_id_added = False  # flag for message

        if patient_id_col is None:
            # No Patient ID found in top-level header
            if ("genomic data" in self.protocol or 
                any(k.startswith("series") for k in self.protocol)):
                error_log.append(f"\nError E004: {error_descriptions.get('E004', None)}")
                error_log.append(
                    "- Details: No valid Patient ID column found in clinical data. "
                    "Expected one of the following (case-insensitive): Patient ID, Name, or Patient Name."
                )
                error_counts["E004"] += 1

                # Mark that no IDs are available
                self.clinical_patient_ids = None
                
            else:
                # Fallback to sequential numbering (purely clinical case)
                self.clinical_patient_ids = list(map(str, range(1, num_clin + 1)))
                
                # Insert the new Patient ID column at the first position
                # First, create the new column as a DataFrame with correct MultiIndex
                patient_id_col = pd.DataFrame(
                    {("PatientID", ""): self.clinical_patient_ids}
                )
                
                # Concatenate the new column to the left of existing DataFrame
                clinical_df = pd.concat([patient_id_col, clinical_df], axis=1)
                
                # Fix second level names: replace any 'Unnamed...' with empty string
                new_tuples = [(lvl0, "" if str(lvl1).startswith("Unnamed") or pd.isna(lvl1) else lvl1)
                              for lvl0, lvl1 in clinical_df.columns]
                
                # Rebuild MultiIndex
                clinical_df.columns = pd.MultiIndex.from_tuples(new_tuples)

                # Overwrite CSV
                clinical_df.to_csv(clinical_path, index=False)
                patient_id_added = True
        else:
            col_data = clinical_df[patient_id_col]

            # If DataFrame → take the first subcolumn
            if isinstance(col_data, pd.DataFrame):
                col_data = col_data.iloc[:, 0]
            
            patient_ids_str = list(map(str, col_data.tolist()))
            # Remove duplicates while preserving order
            self.clinical_patient_ids = list(OrderedDict.fromkeys(patient_ids_str))
            
        # ---- Compute number of patients with clinical data ----
        if self.clinical_patient_ids is None:
            # Use input num_clin directly when E004 occurred
            num_clin = num_clin
        else:
            # Use actual number of unique IDs
            num_clin = len(self.clinical_patient_ids)

        # ============================================================
        # 3) FINALIZATION (INLINE)
        # ============================================================
    
        relevant_error_codes = ["E003", "E004"]
        total_errors = sum(error_counts.values())
    
        # Always generate the check file
        self.generate_check_file_input_data(
            num_patients_with_clinical_data=num_clin,
            num_patients_with_image_data=num_img,
            num_patients_with_genomic_data=num_gen,
            error_counts=error_counts,
            warning_counts=None,
            relevant_error_codes=relevant_error_codes
        )
    
        # If errors → generate report + raise
        if total_errors > 0:
            _, host_report_filename = self.generate_input_validation_report(error_log, total_errors)
            raise RuntimeError(
                f"Clinical Data Check FAILED with {total_errors} errors. "
                f"See detailed report at {host_report_filename}." #! Adapt to LOCAL path, not container.ok
            )

        # If no errors → construct success message
        if patient_id_added:
            clinical_data_validation_succ_msg = (
                "Clinical data validation successful: two-level header format verified. "
                "Patient ID column was missing and has been added with sequential integer IDs."
            )
        else:
            clinical_data_validation_succ_msg = (
                "Clinical data validation successful: two-level header format and Patient ID column verified."
            )

        return num_clin, clinical_data_validation_succ_msg
        

    def validate_series_and_segmentation_folders(self, num_clin, num_img, num_gen):
        """
        Check that each patient folder contains the correct series folders
        based on the 'series_mapping' defined in the local config.
        Additionally, check segmentation folders if required by the protocol.
        Always generates the check file. If errors are present, also generates
        an error report and raises RuntimeError.
        Returns the number of patients with image data and a success message.
        """ 
        # ---------------------
        # INITIALIZATION
        # ---------------------
        error_log = []
        error_counts = defaultdict(int)
        
        phase_number = 0
        phase_data = self.mapping_file.get("check_phase", {}).get(str(phase_number), {})
        error_descriptions = phase_data.get("errors", {})
        
        series_mapping = self.local_config["Local config"]["radiological data"].get("series_mapping", {})
    
        # Expected series folder names (take last part of path in mapping)
        expected_series_names = [v.split('/')[-1] for v in series_mapping.values()]

        patient_folders = [f for f in os.listdir(self.images_dir) if os.path.isdir(os.path.join(self.images_dir, f))]

        # ---- Compute number of patients with image data ----
        num_img = len(patient_folders)

        # Flag to add the main E005 header only once
        e005_header_added = False

        for patient in patient_folders:
            patient_path = os.path.join(self.images_dir, patient)
            series_folders = [f for f in os.listdir(patient_path) if os.path.isdir(os.path.join(patient_path, f))]
    
            # --- Check expected series folders ---
            for series_key, expected_series in zip(series_mapping.keys(), expected_series_names):
                series_path = os.path.join(patient_path, expected_series)

                # Missing series folder
                if expected_series not in series_folders:
                    if not e005_header_added:
                        error_log.append(f"\nError E005: {error_descriptions.get('E005', None)}")
                        e005_header_added = True
                    error_log.append(f"- Details: Patient '{patient}' is missing expected series folder '{expected_series}'.")
                    error_counts["E005"] += 1
                    continue

                # --- Check segmentation folder if required by protocol ---
                protocol_series = self.protocol.get(series_key, {})
                if "segmentation" in protocol_series:
                    # List subfolders inside the series folder
                    subfolders = [f for f in os.listdir(series_path) if os.path.isdir(os.path.join(series_path, f))]
                    if len(subfolders) == 0:
                        if not e005_header_added:
                            error_log.append(f"\nError E005: {error_descriptions.get('E005', None)}")
                            e005_header_added = True
                        error_log.append(
                            f"- Details: Patient '{patient}', series '{expected_series}' "
                            f"is missing its segmentation folder."
                        )
                        error_counts["E005"] += 1
    
                    elif len(subfolders) > 1:
                        if not e005_header_added:
                            error_log.append(f"\nError E005: {error_descriptions.get('E005', None)}")
                            e005_header_added = True
                        error_log.append(
                            f"- Details: Patient '{patient}', series '{expected_series}' "
                            f"contains multiple subfolders; only one segmentation folder is allowed."
                        )
                        error_counts["E005"] += 1

            # --- Check for unexpected extra series folders ---
            for found_series in series_folders:
                if found_series not in expected_series_names:
                    if not e005_header_added:
                        error_log.append(f"\nError E005: {error_descriptions.get('E005', None)}")
                        e005_header_added = True
                    error_log.append(f"- Details: Patient '{patient}' has an unexpected series folder '{found_series}'.")
                    error_counts["E005"] += 1

        # ============================================================
        # FINALIZATION
        # ============================================================
        relevant_error_codes = ["E005"]
        total_errors = sum(error_counts.values())
    
        # Always generate check file
        self.generate_check_file_input_data(
            num_patients_with_clinical_data=num_clin,
            num_patients_with_image_data=num_img,
            num_patients_with_genomic_data=num_gen,
            error_counts=error_counts,
            warning_counts=None,
            relevant_error_codes=relevant_error_codes
        )
    
        # If errors → generate report + raise
        if total_errors > 0:
            _, host_report_filename = self.generate_input_validation_report(error_log, total_errors)
            raise RuntimeError(
                f"Series & Segmentation Check FAILED with {total_errors} errors. "
                f"See detailed report at {host_report_filename}." #! Adapt to LOCAL path, not container.ok
            )
    
        success_msg_img = "Series and segmentation folders validation successful: all expected folders present and correctly structured."

        return num_img, success_msg_img                
                      
    
    def create_patients_list_file(self, num_clin, num_img, num_gen): #4)Funzione nuova
        """
        Creates patients.csv integrating CLINICAL, IMAGES, and GENOMICS data availability.
        Always generates the check file. Generates a report and raises RuntimeError if errors occur.
        Warnings are collected but do not stop execution.
        """ 
            
        # ---------------------
        # INITIALIZATION
        # ---------------------
        error_log = []
        error_counts = defaultdict(int)
        warning_log = []
        warning_counts = defaultdict(int)
        has_warnings = False
        host_report_filename = None
        
        phase_number = 0
        phase_data = self.mapping_file.get("check_phase", {}).get(str(phase_number), {})
        error_descriptions = phase_data.get("errors", {})
        warning_descriptions = phase_data.get("warnings", {})
        
        patients_list_path = os.path.join(self.input_dir, self.patients_list_filename) 

        # Columns of the final table
        columns = [
            "Patient ID",
            "Clinical Data Available",
            "Image Data Available",
            "Genomic Data Available"
        ]

        clinical_enabled = ("clinical data" in self.protocol)
        image_enabled = any(k.startswith("series") for k in self.protocol)
        genomic_enabled = ("genomic data" in self.protocol)
        # ============================================================
        # CASE 1: patients.csv ALREADY EXISTS → LOAD AND VALIDATE
        # ============================================================
        if os.path.exists(patients_list_path):
            patients_df = pd.read_csv(patients_list_path)
    
            # Basic validity checks
            missing_cols = [c for c in columns if c not in patients_df.columns]
            if missing_cols:
                raise RuntimeError(
                    f"Existing patients.csv is invalid. Missing columns: {missing_cols}"
                )
        else:
            # ============================================================
            # CASE 2: patients.csv DOES NOT EXIST → BUILD FROM INPUT DATA
            # ============================================================
            patients_df = pd.DataFrame(columns=columns)
        
            # -----------------------------------------
            # 1) Process CLINICAL DATA
            # -----------------------------------------
            clinical_ids = []
    
            if clinical_enabled and hasattr(self, "clinical_patient_ids"):
                clinical_ids = list(map(str, self.clinical_patient_ids))  # Ensure string IDs
                for pid in clinical_ids:
                    patients_df.loc[len(patients_df)] = [pid, 1, 0, 0]       
            
            # -----------------------------------------
            # 2) Process IMAGE DATA
            # -----------------------------------------
            image_ids = []
        
            if image_enabled:
                patient_folders = [
                    f for f in os.listdir(self.images_dir)
                    if os.path.isdir(os.path.join(self.images_dir, f))
                ]
                image_ids = list(map(str, patient_folders))
        
                for pid in image_ids:
                    if pid in patients_df["Patient ID"].values:
                        # Update existing row
                        patients_df.loc[patients_df["Patient ID"] == pid, "Image Data Available"] = 1
                    else:
                        # New patient only present in images
                        patients_df.loc[len(patients_df)] = [pid, 0, 1, 0]
        
            # -----------------------------------------
            # 3) Process GENOMIC DATA
            # -----------------------------------------
            genomic_ids = []
        
            if genomic_enabled:
                genomics_mapping_filename = self.local_config["Local config"]["genomic data"]["patient-sample mapping file"]
                mapping_file_path = os.path.join(self.genomics_dir, genomics_mapping_filename)
                genomic_map = pd.read_csv(mapping_file_path)
                genomic_ids = list(map(str, genomic_map["patient_id"].astype(str).tolist()))
        
                for pid in genomic_ids:
                    if pid in patients_df["Patient ID"].values:
                        patients_df.loc[patients_df["Patient ID"] == pid, "Genomic Data Available"] = 1
                    else:
                        patients_df.loc[len(patients_df)] = [pid, 0, 0, 1]

        # ============================================================
        # VALIDATION (common to both cases)
        # ============================================================
        # -----------------------------------------
        # 5) Generate warning W001 if mismatched availability
        # -----------------------------------------
    
        # Remove modalities that are entirely absent in protocol
        used_modalities = []
        if clinical_enabled:
            used_modalities.append("Clinical Data Available")
        if image_enabled:
            used_modalities.append("Image Data Available")
        if genomic_enabled:
            used_modalities.append("Genomic Data Available")
    
        # Dictionary to map pattern -> list of patient IDs
        pattern_to_patients = defaultdict(list)   
    
        for _, row in patients_df.iterrows():
            pattern = "".join(str(int(row[col])) for col in used_modalities) #ex. "101", "111", "001"
            pattern_to_patients[pattern].append(row["Patient ID"])
    
        # Only keep patterns that have at least one 0 (missing data)
        mismatch_patterns = {p: ids for p, ids in pattern_to_patients.items() if "0" in p}
    
        if mismatch_patterns:
           warning_msg = f"\nWarning W001: {warning_descriptions.get('W001', None)}\n"
           warning_msg += "- Data availability patterns with missing modalities:\n"
           for pattern, patient_ids in mismatch_patterns.items():
               warning_msg += f"  Pattern {pattern}: {len(patient_ids)} patient(s) -> IDs: {', '.join(patient_ids)}\n"
           warning_log.append(warning_msg)
           warning_counts["W001"] = sum(len(ids) for ids in mismatch_patterns.values())
        else:
           warning_counts["W001"] = 0  # No W001 warning       
    
        # -----------------------------------------
        # 6) Check total number of patients against config
        # -----------------------------------------
        expected_total = self.local_config["Local config"].get("number of total patients", None)
        if expected_total is not None:
            if len(patients_df) != expected_total:
                error_log.append(f"\nError E007: {error_descriptions.get('E007', None)}")
                error_log.append(
                    f"- Details: Expected {expected_total} patients, but found {len(patients_df)} in patients.csv."
                )
                error_counts["E007"] += 1
    
        # -----------------------------------------
        # 7) Save patients.csv if it not exists
        # -----------------------------------------
        
        if not os.path.exists(patients_list_path):
            try:
                patients_df.to_csv(patients_list_path, index=False)
            except Exception as e:
                raise RuntimeError(f"Could not save patients.csv: {e}")

        # ---------------------
        # 8) FINALIZATION
        # ---------------------
        relevant_error_codes = ["E007"]
        total_errors = sum(error_counts.values())
        total_warnings = sum(warning_counts.values())
        
        # Always generate the check file
        self.generate_check_file_input_data(
            num_patients_with_clinical_data=num_clin,
            num_patients_with_image_data=num_img,
            num_patients_with_genomic_data=num_gen,
            error_counts=error_counts,
            warning_counts=warning_counts,
            relevant_error_codes=relevant_error_codes
        )
    
        if total_errors > 0:
            _, host_report_filename = self.generate_input_validation_report(
                error_log, total_errors, warning_log, total_warnings
            )
            raise RuntimeError(
                f"Patients list creation FAILED with {total_errors} errors. "
                f"See detailed report at {host_report_filename}." # adjust the path (local path, not container path).ok
            )
        
        # If only warnings, return success + report
        if total_warnings > 0:
            has_warnings = True
            _, host_report_filename = self.generate_input_validation_report(
                error_log, total_errors, warning_log, total_warnings
            )
    
        success_msg_pts_list = (
            f"Patients list creation successful: patients.csv generated and availability verified.\n"
            f"  File saved at: {self.host_patients_list_path}." # metto quello locale, non del container.ok
        )
        return success_msg_pts_list, has_warnings, host_report_filename
    
       
    @staticmethod    
    def generate_patient_id(index):
        """Generate a new patient ID with a 3-digit number and 5 random letters."""
        patient_num = f"{index:05d}"  # (03d) 3-digit progressive number (001, 002, ...)
        random_letters = ''.join(random.choices(string.ascii_uppercase, k=5))  # 5 random letters
        return f"patient{patient_num}_{random_letters}"


    def patient_mapping(self):
        """Map patient IDs and save to patients.csv (only if not already mapped)."""
        report_pts_mapping = ""
        df = read_csv_file(self.patients_list_path)

        # Check if 'New Patient ID' exists and is complete
        if 'New Patient ID' in df.columns and df['New Patient ID'].notnull().all():
            report_pts_mapping += f"Patient IDs have already been mapped and saved in {self.host_patients_list_path}.\n" # metto path locale.ok
        else:
            # Generate new patient IDs
            df['New Patient ID'] = [
                self.generate_patient_id(i + 1) for i in range(len(df))
            ]
            report_pts_mapping += f"Patient IDs have been mapped and saved in {self.host_patients_list_path}.\n" # metto path locale.ok

            # Overwrite the original CSV file with the updated patient names
            df.to_csv(self.patients_list_path, index=False)

        return report_pts_mapping
    

    def write_final_report(self, input_data_structure_succ_msg, clinical_data_validation_succ_msg, success_msg_img, success_msg_pts_list, has_warnings, warning_report_filename, report_pts_mapping):
        """
        Generate a final summary report combining:
        - Input data structure check
        - Clinical data preliminary validation (if applicable)
        - Image data preliminary validation (if applicable)
        - Patients list creation (with optional warnings)
        - Patient mapping
        """
    
        # Define the report file path
        report_file_name = "0.InputCheckAndPatientMapping_final_report.txt"
        report_file_path = os.path.join(self.output_directory_report, report_file_name)
        host_report_file_path = os.path.join(self.host_output_directory_report, report_file_name)
        
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
        
        # Open the file in write mode
        with open(report_file_path, 'w') as file:
            # Write the first report
            file.write(f"Report generated on: {formatted_datetime}\n\n")
            file.write("Input check and patient mapping final report:\n\n")

            # 1) Input Data Structure
            file.write("Input Data Structure Validation:\n")
            file.write("- " + input_data_structure_succ_msg + "\n\n")  # Add a newline for separation
            
            # 2. Clinical Preliminary Validation (optional)
            if clinical_data_validation_succ_msg is not None:
                file.write("Clinical Data Preliminary Validation:\n")
                file.write("- " + clinical_data_validation_succ_msg + "\n\n")

            # 3. Image Preliminary Validation (optional)
            if success_msg_img is not None:
                file.write("Image Data Preliminary Validation:\n")
                file.write("- " + success_msg_img + "\n\n")

            # 4. Patients List Creation 
            file.write("Patients List Creation:\n")
            file.write("- " + success_msg_pts_list + "\n")

            if has_warnings:
                file.write(
                    "- Warning W001 detected: mismatch in data availability across different modalities.\n"
                    f"  See detailed warning report at: {warning_report_filename}.\n\n" # metto path locale. ok
                )
            else:
                file.write("\n")
            
            # 5. Patient Mapping
            file.write("Patient Mapping:\n")
            file.write("- " + report_pts_mapping + "\n")            
        
        print(f"Final report written to {host_report_file_path}.") #path locale.ok


def run_input_checks_patient_mapping(protocol, local_config, check_phase, mapping_file):

    # Define the input directory path  
    input_dir = os.getenv("INPUT_DIR")

    # Clear log at the start of validation
    clear_log_file(input_dir, LOG_FILENAME)

    # Initialization
    clinical_data_validation_succ_msg = None
    success_msg_img = None

    '''
    if check_phase >= 1:
        print("Phase 0 (Input Checks and Patient Mapping) already completed. Skipping...")
        return  # Skip this phase
    '''
    
    print("Running Phase 0: Input Checks and Patient Mapping...")
    checker_mapper = InputCheckAndPatientMapping(protocol, local_config, mapping_file)

    # Load the state if it exists
    state = load_state(checker_mapper.state_file)

    try:
        checker_mapper.force_lowercase_extensions() # soprattutto per Windows
    except Exception as e:
        log_error(input_dir, "force_lowercase_extensions", e, LOG_FILENAME)
        print(f"An unexpected error occurred during force_lowercase_extensions. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
        raise
    
    print("Running check_input_data_structure function...")
    try:
        num_clin, num_img, num_gen, input_data_structure_succ_msg = checker_mapper.check_input_data_structure()
        state["num_patients_with_clinical_data"] = int(num_clin)
        state["num_patients_with_image_data"] = int(num_img)
        state["num_patients_with_genomic_data"] = int(num_gen)
        save_state(state, checker_mapper.state_file)
    except Exception as e:
        log_error(input_dir, "check_input_data_structure", e, LOG_FILENAME)
        print(f"An unexpected error occurred during check_input_data_structure. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
        raise
    
    # run just if clinical data are present
    if any(k.startswith("clinical data") for k in protocol):
        print("Running validate_clinical_data function...")
        try:
            num_clin, clinical_data_validation_succ_msg = checker_mapper.validate_clinical_data(num_clin, num_img, num_gen)
            state["num_patients_with_clinical_data"] = int(num_clin)
            save_state(state, checker_mapper.state_file)
        except Exception as e:
            log_error(input_dir, "validate_clinical_data", e, LOG_FILENAME)
            print(f"An unexpected error occurred during validate_clinical_data. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
            raise
    else:
        print("No clinical data detected. Skipping preliminary clinical data validation step.")

    # da runnare solo in presenza di immagini
    if any(k.startswith("series") for k in protocol):
        print("Running validate_series_and_segmentation_folders function...")
        try:
            num_img, success_msg_img = checker_mapper.validate_series_and_segmentation_folders(num_clin, num_img, num_gen)
            state["num_patients_with_image_data"] = int(num_img)
            save_state(state, checker_mapper.state_file)
        except Exception as e:
            log_error(input_dir, "validate_series_and_segmentation_folders", e, LOG_FILENAME)
            print(f"An unexpected error occurred during validate_series_and_segmentation_folders. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
            raise
    else:
        print("No imaging data detected. Skipping preliminary imaging data validation step.")

    print("Running create_patients_list_file function...")
    try:
        success_msg_pts_list, has_warnings, warning_report_filename = checker_mapper.create_patients_list_file(num_clin, num_img, num_gen)
    except Exception as e:
        log_error(input_dir, "create_patients_list_file", e, LOG_FILENAME)
        print(f"An unexpected error occurred during create_patients_list_file. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
        raise
    
    print("Running patient_mapping function...")
    try:
        report_pts_mapping = checker_mapper.patient_mapping()
    except Exception as e:
        log_error(input_dir, "patient_mapping", e, LOG_FILENAME)
        print(f"An unexpected error occurred during patient_mapping. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
        raise
    
    print("Running write_final_report function...")
    try:
        checker_mapper.write_final_report(input_data_structure_succ_msg, clinical_data_validation_succ_msg, success_msg_img, success_msg_pts_list, has_warnings, warning_report_filename, report_pts_mapping)
    except Exception as e:
        log_error(input_dir, "write_final_report", e, LOG_FILENAME)
        print(f"An unexpected error occurred during write_final_report. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
        raise

