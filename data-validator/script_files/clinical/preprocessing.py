import pandas as pd
import numpy as np
import os 
from summarytools import dfSummary
import json
from datetime import datetime
import re
from collections import defaultdict
import operator
from utils import read_csv_file, clear_log_file, log_error
from .helpers import (
    generate_check_file_clinical_data,
    retrieve_existing_warnings_clinical_data,
    _strip_patient_suffix
)

OPS = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
}

LOG_FILENAME = "clinical_data_preprocessing_error.log"

# Clinical Data Pre-processing
class ClinicalDataPreProcessing:
    
    def __init__(self, protocol, local_config, mapping_file):
        self.protocol = protocol
        self.local_config = local_config
        self.mapping_file = mapping_file
        self.clinical_data = self.protocol.get("clinical data", {})
        self.study_path = os.getenv("ROOT_NAME")
        self.input_dir = os.getenv("INPUT_DIR")
        self.host_input_dir = os.getenv("HOST_BASE_DATA_DIR")
        self.clinical_dir = os.path.join(self.input_dir, "CLINICAL")
        self.file_name = self.local_config["Local config"]["clinical data"]["file name"]
        self.input_data_path = os.path.join(self.clinical_dir, self.file_name)
        self.output_directory_report = os.path.join(self.input_dir, "Reports", "Clinical Data Reports")
        self.host_output_directory_report = os.path.join(self.host_input_dir, "Reports", "Clinical Data Reports")
        self.output_directory_checkfile = os.path.join(self.study_path, "CHECKFILE")
        self.output_directory_data = os.path.join(self.study_path, "CLINICAL")
        os.makedirs(self.output_directory_report, exist_ok=True)
        os.makedirs(self.output_directory_checkfile, exist_ok=True)
        os.makedirs(self.output_directory_data, exist_ok=True)

        mapping_file_path = os.path.join(self.study_path, "MappingFile", "variables_mapping.json")
        try:
            with open(mapping_file_path, "r") as file:
                self.variables_mapping = json.load(file)
        except FileNotFoundError:
            print(f"Error: {mapping_file_path} not found.")
            self.variables_mapping = {}
        except json.JSONDecodeError:
            print(f"Error: Failed to decode {mapping_file_path}.")
            self.variables_mapping = {}

        mapping_file_path = os.path.join(self.study_path, "MappingFile", "variables_mapping.json")
        try:
            with open(mapping_file_path, "r") as file:
                self.variables_mapping = json.load(file)
        except FileNotFoundError:
            print(f"Error: {mapping_file_path} not found.")
            self.variables_mapping = {}
        except json.JSONDecodeError:
            print(f"Error: Failed to decode {mapping_file_path}.")
            self.variables_mapping = {}


    def substitute_patient_ids(self, data): #MODIFICATA
        """
        Substitute Patient IDs in the data based on the mapping from the patients list.
    
        Args:
        - data (DataFrame): The DataFrame containing the clinical data.
        - warning_counts (dict): Dictionary containing the counts of various warnings.
    
        Returns:
        - DataFrame: The updated DataFrame with substituted Patient IDs.
        - str: Success or error message.
        """
        phase_number = 8  
        phase_data = self.mapping_file.get("check_phase", {}).get(str(phase_number), {})
        error_descriptions = phase_data.get("errors", {})
        phase_name = phase_data.get("name", f"Phase {phase_number}")  
        
        # Format check and report names dynamically
        check_name = phase_name.lower().replace(" ", "_") 
        report_name = f"2.008.{check_name}_report"
        
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        patients_list_filename = self.local_config["Local config"]["patients list"]
        patients_list_path = os.path.join(self.input_dir, patients_list_filename)
        # Read patients_list_filename file
        patients_list_df = read_csv_file(patients_list_path)

        # Check that the patients_list_df has at least 4 columns
        if patients_list_df.shape[1] < 5:
            raise ValueError("patients_list_filename should have at least five columns: ID, Clinical Data (0/1), Image Data (0/1), Genomic Data (0/1), New Patient ID")
            
        # Create a mapping only for patients with clinical data (second column == 1)
        clinical_mapping = patients_list_df[patients_list_df.iloc[:, 1] == 1].set_index(patients_list_df.columns[0])[patients_list_df.columns[4]].to_dict() #!!
        
        # Initializations
        mismatches = []
        success_message = ""
        new_index = []
        warning_counts = {}

        check_file_path = os.path.join(self.output_directory_checkfile, "clinical_data_check_file.json")
        
        # Retrieve existing warnings with descriptions from the check file
        existing_warnings = retrieve_existing_warnings_clinical_data(check_file_path)

        warning_descriptions = {}
        row_counters = defaultdict(int)
        
        # Iterate over the index (Patient IDs) of the data DataFrame
        new_index = []
        for old_id in data.index:
            if old_id in clinical_mapping:
                base_id = clinical_mapping[old_id]
                suffix = row_counters[old_id] + 1
                new_index.append(f"{base_id}_{suffix}")
                row_counters[old_id] += 1
            else:
                # If the old ID is not found in the mapping, add it to mismatches
                mismatches.append(old_id)

        # Initialize error count
        num_errors = 0

        # If there are mismatches, generate a report
        if mismatches:
            report_path = os.path.join(self.output_directory_report, f"{report_name}.txt")
            host_report_path = os.path.join(self.host_output_directory_report, f"{report_name}.txt")
            with open(report_path, "w") as report_file:
                report_file.write(f"Report generated on: {current_datetime}\n\n")
                    
                if mismatches:
                    report_file.write(f"{phase_name} report:\n\n")
                    report_file.write(f"Error E801: {error_descriptions.get('E801', None)}\n")
                    report_file.write("- Details:\n")
                    report_file.write(f"  The following Patient IDs were not found in the mapping file:\n")
                    for mismatch in mismatches:
                        report_file.write(f"  -{mismatch}\n")
                    num_errors += len(mismatches)

                    # Write the total number of mismatches
                    report_file.write(f"\nTotal Errors: {len(mismatches)}\n")
            
            print(f"Mismatch report generated at {host_report_path}")

            # Generate the check file using generate_check_file_clinical_data
            num_patients = len(data)
            error_counts = {"E801": num_errors}  # Error code for mismatched Patient IDs
            generate_check_file_clinical_data(
                check_name=phase_name,
                phase_number=phase_number,
                total_rows=num_patients,
                total_unique_patients=_strip_patient_suffix(
                    pd.Index(new_index)
                ).nunique(),
                timestamp=current_datetime,
                error_counts=error_counts,
                warning_counts=warning_counts,
                error_descriptions=error_descriptions,
                warning_descriptions=warning_descriptions,
                output_dir=self.output_directory_checkfile,
                existing_warnings=existing_warnings
            )

            # Raise an error after generating the report
            raise ValueError(f"Mismatch found in Patient IDs. Check the mismatch report for details at {host_report_path}.")
            
        else:
            # Generate the check file when no mismatches are found
            num_patients = len(data)
            error_counts = {"E801": None}  # No errors if there are no mismatches
            generate_check_file_clinical_data(
                check_name=phase_name,
                phase_number=phase_number,
                total_rows=num_patients,
                total_unique_patients=_strip_patient_suffix(
                    pd.Index(new_index)
                ).nunique(),
                timestamp=current_datetime,
                error_counts=error_counts,
                warning_counts=warning_counts,
                error_descriptions=error_descriptions,
                warning_descriptions=warning_descriptions,
                output_dir=self.output_directory_checkfile,
                existing_warnings=existing_warnings
            )
                
            # Update the index with the new patient IDs (only if no mismatches were found)
            data.index = new_index

            # Ensure the index is renamed to "Patient ID"
            data.index.name = "Patient ID"

            success_message = "No mismatches found. Patient ID substitution completed successfully."
            print(success_message)
    
            return data, success_message 
        
        
    def transformations(self, data, num_dates):
        if num_dates == 0:
             transformations_message = "0 date variables in the protocol."
             return transformations_message, [], data
            
        #extract from the protocol the list of transformed variables
        transformation_details = self.clinical_data.get("Transformations", {})
        transformed_variables_protocol = self.clinical_data.get("Transformations", {}).get("transformed_variables", {})
        transformed_variable_names = set(transformed_variables_protocol.keys())
    
        # Initialize a list to store added variables
        added_variables = []  
    
        # Retrieve the expected NA value from the protocol
        expected_NA_value = self.clinical_data.get("Missing Values", {}).get("expected_value")
        
        # Extract var1 and var2
        var1 = transformation_details.get("var1")
        var2 = transformation_details.get("var2")

        # Check if var1 and var2 are valid keys in the protocol
        if var1 and var2 and var1 in self.clinical_data and var2 in self.clinical_data \
            and "date_format" in self.clinical_data[var1] and "date_format" in self.clinical_data[var2]:
        
            # Convert date variables to datetime objects
            data[var1] = pd.to_datetime(data[var1], format=self.clinical_data[var1]["date_format"], errors="coerce")
            data[var2] = pd.to_datetime(data[var2], format=self.clinical_data[var2]["date_format"], errors="coerce")

            
            # Apply transformations
            for transformed_variable, details in transformed_variables_protocol.items():
                transformation_formula = details.get("formula")
                updated_formula = transformation_formula.replace('var1', f'data["{var1}"]').replace('var2', f'data["{var2}"]')
            
                # Evaluate the updated transformation formula using eval()
                data[transformed_variable] = eval(updated_formula)
                data[transformed_variable] = data[transformed_variable].dt.days

                # Add the transformed variable to the list of added variables
                added_variables.append(transformed_variable)

                # Convert date variables back to strings
                data[var1] = data[var1].dt.strftime(self.clinical_data[var1]["date_format"])
                data[var2] = data[var2].dt.strftime(self.clinical_data[var2]["date_format"])

                data.fillna(expected_NA_value, inplace=True)

                # Convert the transformed variable to integer if it is of float type and all values are integers
                if (data[transformed_variable] % 1 == 0).all() and data[transformed_variable].dtype == float:
                    data[transformed_variable] = data[transformed_variable].astype(int)                 
    
        if transformed_variable_names.issubset(data.keys()):
            transformations_message = "All variables transformed according to the protocol."
        else:
            transformations_message = "Some variables NOT transformed according to the protocol."
    
        return transformations_message, added_variables, data
    

    def handle_ground_truth_variable(self, data):
        # Find the ground truth variable from the protocol
        ground_truth_variable = None
        for var_name, var_details in self.clinical_data.items():
            if isinstance(var_details, dict) and var_details.get("ground-truth") == "yes":
                ground_truth_variable = var_name
                break

        if not ground_truth_variable:
            raise ValueError("Ground-truth variable is not defined in the protocol.")
         
        if data.index.name is None:
            # Reset the index to start from 1 and rename it to "PatientID"
            data.reset_index(drop=True, inplace=True)
            data.index = data.index + 1
            data.index.name = "Patient ID"
        
        ground_truth_data = data[ground_truth_variable]
        raw_expected_NA_value = self.clinical_data.get("Missing Values", {}).get("expected_value")

        # Handle both numerical and string versions of the missing value
        missing_values = [raw_expected_NA_value, str(raw_expected_NA_value)]
    
        # Identify rows with missing values in the ground truth variable
        missing_value_positions = ground_truth_data[ground_truth_data.isin(missing_values)].index.tolist()
    
        if missing_value_positions:
            # Store the positions of the eliminated rows
            eliminated_rows_dict_ground_truth = {ground_truth_variable: missing_value_positions}
            
            # Calculate the number of missing values
            num_missing_values_ground_truth = len(missing_value_positions)
            
            # Eliminate rows with missing values in the ground truth variable
            data = data.drop(index=missing_value_positions)

            # Save the message
            message_gt_NA = f"Observations with missing data on ground-truth variable {ground_truth_variable} have been deleted."
        else:
            eliminated_rows_dict_ground_truth = None
            num_missing_values_ground_truth = 0
            # Save the message
            message_gt_NA = f"Ground-truth variable {ground_truth_variable}: no missing values."
            print(message_gt_NA)
            
        return data, eliminated_rows_dict_ground_truth, num_missing_values_ground_truth, message_gt_NA
    

    def discriminate_variables(self, data):
        continuous_vars = []
        discontinuous_vars = []
    
        transformations = self.clinical_data.get("Transformations", {})
        variables_to_discard = [value for key, value in transformations.items() if key.startswith("var")]
        
        for column in data.columns:
            # Exclude transformed variables
            if column not in variables_to_discard:
                is_continuous = True
                for value in data[column]:
                    if not pd.api.types.is_numeric_dtype(type(value)):
                        is_continuous = False
                        break
                if is_continuous:
                    continuous_vars.append(column)
                else:
                    discontinuous_vars.append(column)
            
        # create separate datasets for continuous and discontinuous variables
        continuous_data = data[continuous_vars]
        discontinuous_data = data[discontinuous_vars]
    
        return continuous_data, discontinuous_data
    

    def handle_missing_continuous_values(self, continuous_data):
        # Check if there are any missing values in continuous_data
        if (continuous_data == self.clinical_data.get("Missing Values", {}).get("expected_value")).sum().sum() == 0:
            continuous_na_message = "0 missing values."
            return continuous_data, continuous_na_message
        
        continuous_data_copy = continuous_data.copy()
        
        missing_counts = {}

        # Retrieve expected NA value from the protocol
        expected_NA_value = self.clinical_data.get("Missing Values", {}).get("expected_value")


        # Funzione per arrotondare il valore float all'intero più vicino
        def round_to_nearest_integer(x):
            if x - int(x) >= 0.5:
                return int(x) + 1
            else:
                return int(x)

        # Handle missing values for continuous variables
        for column in continuous_data_copy.columns:
            missing_counts[column] = (continuous_data_copy[column] == expected_NA_value).sum()
            if missing_counts[column] > 0:
                
                # Get the type of the expected NA value for this column
                expected_NA_type = type(expected_NA_value)
                # Get non-missing values (excluding expected_NA_value)
                non_missing_values = continuous_data_copy[column][continuous_data_copy[column] != expected_NA_value]
                if expected_NA_type == int:
                    mean_value = round_to_nearest_integer(non_missing_values.mean())
                elif expected_NA_type == float:
                    mean_value = non_missing_values.mean()
                
                # Fill missing values with the mean
                continuous_data_copy.loc[continuous_data_copy[column] == expected_NA_value, column] = mean_value

        continuous_na_message = "Done."

        return continuous_data_copy, continuous_na_message
    

    def _analyze_categorical_unbalance(self, categorical_df):
        if categorical_df.empty:
            return {}, {}, 0
    
        one_hot = pd.get_dummies(categorical_df, dummy_na=False).astype(int)
        label_counts = one_hot.sum()
    
        nonsignificant = {}
        nonsignificant_string = {}
        total_count = 0
    
        for column in categorical_df.columns:
            protocol = self.clinical_data.get(column, {})
            condition = protocol.get("condition", {})
            expected_list = condition.get("expected_unbalance", [])
    
            if not expected_list:
                continue
    
            expected_unbalance = float(
                expected_list[0].replace('<', '').replace('%', '').strip()
            )
            cardinality = protocol.get("cardinality", 1)
    
            relevant = [c for c in one_hot.columns if c.startswith(column + "_")]
            if not relevant:
                continue
    
            level_counts = label_counts[relevant]
            nonsignificant_levels = []
    
            for level, count in level_counts.items():
                suffix = level.split("_")[-1]
                x = (1 - ((count / len(categorical_df)) * cardinality)) * 100
                if x >= expected_unbalance:
                    nonsignificant_levels.append(suffix)
    
            if nonsignificant_levels:
                var_map = self.variables_mapping.get(column, {})
                levels_map = var_map.get("levels", {})
                coded = [
                    levels_map.get(l, l).split('.')[-1]
                    for l in nonsignificant_levels
                ]
    
                code = var_map.get("variable_code", column)
                nonsignificant[code] = coded
                nonsignificant_string[column] = nonsignificant_levels
                total_count += len(coded)
    
        return nonsignificant, nonsignificant_string, total_count
    

    def check_non_significant_levels_before(self, discontinuous_data):
        # Replace -999 with NaN
        discontinuous_data = discontinuous_data.replace('-999', pd.NA)

        (
            nonsignificant_counts_before,
            nonsignificant_counts_before_string,
            total_nonsignificant_before
        ) = self._analyze_categorical_unbalance(discontinuous_data)

        # Generate the message based on the results
        if total_nonsignificant_before == 0:
            non_significant_message_before = "Not present."
        else:
            non_significant_message_before = f"Warning - Total number of non significant levels: {total_nonsignificant_before}."

        return nonsignificant_counts_before, nonsignificant_counts_before_string, total_nonsignificant_before, non_significant_message_before
    

    def create_multi_header_df(self, single_header_df): 
        """
        Converts a single-header DataFrame to a multi-header DataFrame by extracting 
        the unit mapping from a saved multi-header CSV file and adding transformed variables.
        
        Parameters:
        - single_header_df (pd.DataFrame): DataFrame with single-level column names.

        Returns:
        - pd.DataFrame: Multi-header DataFrame with units added.
        """
        # Ensure "Patient ID" is not the index
        if single_header_df.index.name == "Patient ID":
            single_header_df = single_header_df.reset_index()  # Convert index to column
                
        # Step 1: Read multi-header CSV file (preserve MultiIndex columns)
        multi_header_data = pd.read_csv(self.input_data_path, sep=";", header=[0, 1])
    
        # Step 2: Create unit mapping from multi-header data
        original_unit_mapping  = dict(zip(
            multi_header_data.columns.get_level_values(0),  # Feature names (first level)
            multi_header_data.columns.get_level_values(1)   # Corresponding units (second level)
        ))
        # Step 3: Extract transformation details
        transformations = self.clinical_data.get("Transformations", {})
        transformed_variables = transformations.get("transformed_variables", {})
    
        # Step 4: Add transformed variables and their units to original_unit_mapping 
        for var_name, details in transformed_variables.items():
            original_unit_mapping [var_name] = details.get("unit_of_measure", "")
            
        # Step 5: Ensure "Patient ID" is included and move it to the first position
        unit_mapping = {"Patient ID": ""}
        unit_mapping.update(original_unit_mapping)  # Preserve order for the rest

        # Step 6: Ensure "Patient ID" exists in single_header_df
        if "Patient ID" not in single_header_df.columns:
            single_header_df.insert(0, "Patient ID", None)  # Insert at first position if missing

        # Step 7: Build multi-index columns for the new DataFrame
        multi_index_columns = pd.MultiIndex.from_tuples(
            [(col, unit_mapping.get(col, "")) for col in single_header_df.columns]
        )

        # Step 7: Convert single-header DataFrame to multi-header DataFrame
        multi_header_df = single_header_df.copy()
        multi_header_df.columns = multi_index_columns  # Assign new multi-index columns

        first_level_columns = multi_header_df.columns.get_level_values(0)
        second_level_columns = multi_header_df.columns.get_level_values(1)

        second_level_columns = ["" if str(col).startswith("Unnamed") else col for col in second_level_columns]
        multi_header_df.columns = pd.MultiIndex.from_tuples(zip(first_level_columns, second_level_columns))           

        return multi_header_df
    

    def _save_discontinuous_and_gt(self, discontinuous_data, continuous_data=None):
        """
        Save ground-truth column separately if discontinuous and save
        preprocessed_discontinuous_data.csv only if there is at least one non-GT categorical variable
        AND there is at least one continuous variable (otherwise it would be redundant with preprocessed_total_data.csv).
        Returns the discontinuous_data without GT.
        """
        ground_truth_variable = None
        for var_name, var_details in self.clinical_data.items():
            if isinstance(var_details, dict) and var_details.get("ground-truth") == "yes":
                ground_truth_variable = var_name
                break
    
        if ground_truth_variable and ground_truth_variable in discontinuous_data.columns:
            ground_truth_details = self.clinical_data.get(ground_truth_variable)
            if ground_truth_details and "cardinality" in ground_truth_details:
                gt_df = discontinuous_data[[ground_truth_variable]].copy()
                gt_df.index = _strip_patient_suffix(gt_df.index)
                gt_path = os.path.join(self.output_directory_data, 'ground_truth.csv')
                self.create_multi_header_df(gt_df).to_csv(gt_path, index=False, sep=";")
                discontinuous_data_without_gt = discontinuous_data.drop(columns=[ground_truth_variable])
            else:
                discontinuous_data_without_gt = discontinuous_data
        else:
            discontinuous_data_without_gt = discontinuous_data

        # Decide whether to save preprocessed_discontinuous_data.csv
        categorical_predictors = set(discontinuous_data_without_gt.columns)

        save_discontinuous = False
        if categorical_predictors:
            if continuous_data is None or continuous_data.empty:
                # Only save if there is at least one continuous column that is not the GT
                save_discontinuous = False
            else:
                cont_columns = [col for col in continuous_data.columns if col != ground_truth_variable]
                if cont_columns:  # There is at least one continuous column that is not GT
                    save_discontinuous = True
    
        if save_discontinuous:
            disc_df_to_save = discontinuous_data_without_gt.copy()
            disc_df_to_save.index = _strip_patient_suffix(disc_df_to_save.index)
            disc_path = os.path.join(self.output_directory_data, 'preprocessed_discontinuous_data.csv')
            self.create_multi_header_df(disc_df_to_save).to_csv(disc_path, index=False, sep=";")
    
        return discontinuous_data_without_gt


    def handle_missing_discontinuous_values(self, discontinuous_data, n, continuous_data=None):
        """
        Handle missing values in discontinuous data, including small cardinality expansion,
        row deletion, and optional saving of preprocessed discontinuous data.
        
        Returns:
        - pd.DataFrame: processed discontinuous data
        - int: number of deleted observations
        - list: deleted row indices
        - str: message
        """
        # Retrieve expected NA value from the protocol
        expected_NA_value = self.clinical_data.get("Missing Values", {}).get("expected_value")
    
        # Handle both numeric and string representations of the missing value
        discontinuous_data = discontinuous_data.replace([expected_NA_value, str(expected_NA_value)], np.nan)
    
        # ------------------------
        # Case 1: no missing values
        # ------------------------
        if discontinuous_data.isnull().sum().sum() == 0:
            discontinuous_na_message = "0 missing values."
                
            _ = self._save_discontinuous_and_gt(discontinuous_data, continuous_data)

            return discontinuous_data, 0, [], discontinuous_na_message  
        
        else:
            # Retrieve expected NA value from the protocol
            expected_NA_value = str(self.clinical_data.get("Missing Values", {}).get("expected_value"))
    
            discontinuous_data = discontinuous_data.copy()
            
            missing_counts = {}
            # Store the original number of unique indices
            original_indices = discontinuous_data.index
            original_unique_indices = discontinuous_data.index.nunique()
            deleted_indices = set()
            
            # Convert expected NA value to NaN
            discontinuous_data.loc[:, :] = discontinuous_data.replace(expected_NA_value, np.nan)
    
            for column in discontinuous_data.columns:
                missing_counts[column] = discontinuous_data[column].isnull().sum()
                if missing_counts[column] > 0:
                    #any_missing_values_handled = True  # Mark that we are handling missing values
                    if discontinuous_data[column].nunique() < n:
                        possible_values = discontinuous_data[column].dropna().unique()
                        
                        for value in possible_values:
                            missing_rows = discontinuous_data[discontinuous_data[column].isnull()].copy()
                            missing_rows[column] = value
                            # Iterate over each row of missing_rows
                            for index, row in missing_rows.iterrows():
                                # Check if the number of NaN values in the row is at most one
                                if row.isnull().sum() < 1:
                                    discontinuous_data = pd.concat([discontinuous_data, row.to_frame().T], ignore_index=False)
                    else:
                        discontinuous_data = discontinuous_data.dropna(subset=[column]).copy()
        
            # Drop rows with more than one missing value
            #discontinuous_data = discontinuous_data[discontinuous_data.isnull().sum(axis=1) <= 1]
            discontinuous_data.dropna(inplace=True)
    
            # Identify the indices of deleted observations
            deleted_indices = set(original_indices) - set(discontinuous_data.index)
            deleted_indices_list = sorted(list(deleted_indices))
            # Calculate the number of deleted observations
            num_deleted_observations = original_unique_indices - discontinuous_data.index.nunique()
    
            # Ensure the index name "Row" is preserved
            discontinuous_data.index.name = "Patient ID"

            discontinuous_data_single_level = discontinuous_data.copy()

            # Save ground-truth and preprocessed discontinuous data if needed
            _ = self._save_discontinuous_and_gt(discontinuous_data, continuous_data)
    
            discontinuous_na_message = "Done."
    
            return discontinuous_data_single_level, num_deleted_observations, deleted_indices_list, discontinuous_na_message


    def _merge_clinical_data(self, categorical_df, continuous_df):
        """
        Merge categorical (possibly duplicated patients) with continuous
        (one row per patient), duplicating continuous values where needed.
        Safely handles empty inputs.
        """

        # -------------------------
        # Handle empty inputs first
        # -------------------------
        if categorical_df.empty:
            # Continuous-only case
            return continuous_df.copy()

        if continuous_df.empty:
            # Categorical-only case
            return categorical_df.copy()
            
        # -------------------------
        # Normal merge case
        # -------------------------   
        # Ensure indexes are named consistently (optional but safer)
        categorical = categorical_df.copy()
        continuous = continuous_df.copy()
    
        categorical["__pid__"] = categorical.index
        continuous["__pid__"] = continuous.index
    
        merged = categorical.merge(
            continuous,
            on="__pid__",
            how="left",           # categorical drives the merge
            validate="many_to_one"
        )
    
        merged = merged.set_index("__pid__")
        merged.index.name = categorical_df.index.name
    
        return merged   


    def _analyze_ground_truth(
        self,
        gt_var,
        merged_continuous_df,
        nonsignificant_after,
        nonsignificant_string,
        warning_descriptions
    ):
        protocol = self.clinical_data[gt_var]
        var_map = self.variables_mapping.get(gt_var, {})
        gt_code = var_map.get("variable_code", gt_var)
    
        # -------------------------
        # CATEGORICAL GT (W902)
        # -------------------------
        if "cardinality" in protocol:
            if gt_code in nonsignificant_after:
                levels = nonsignificant_string.get(gt_var, [])
                message = (
                    f"Warning W902: {warning_descriptions.get('W902', None)}\n"
                    f"- Details: Ground truth variable '{gt_var}' "
                    f"non significant levels: {', '.join(levels)}."
                )
                return "W902", message, len(levels)
    
            return None, (
                f"Ground truth variable '{gt_var}': no non-significant levels."
            ), 0
    
        # -------------------------
        # CONTINUOUS GT (W903)
        # -------------------------
        
        if protocol.get("type") in ("int", "float"):
            gt_data = merged_continuous_df[gt_var]
            condition = protocol.get("condition", {}).get("expected_variability")
    
            if not condition:
                return None, "No variability constraint defined.", 0
    
            variability_condition = condition[0].strip()
            
            m = re.match(r"^\s*(<=|>=|<|>)\s*(\d+(?:\.\d+)?)\s*%\s*$", variability_condition)
            if not m:
                raise ValueError(f"Invalid expected_variability format: '{variability_condition}'")
                
            # Extract operator and threshold
            op_str, value = m.group(1), float(m.group(2))
        
            mean = gt_data.mean()
            std = gt_data.std()
            cv = 0 if mean == 0 else (std / mean) * 100

            if not OPS[op_str](cv, value):
                message = (
                    f"Warning W903: {warning_descriptions.get('W903', None)}\n"
                    f"- Details: The expected variability for ground-truth variable "
                    f"'{gt_var}' is not {variability_condition}. "
                    f"Actual: {cv:.2f}%."
                )
                return "W903", message, 1
    
            return None, (
                f"Ground truth ('{gt_var}') variability is compliant: "
                f"coefficient of variation (CV) = {cv:.2f}%, "
                f"expected {variability_condition} as defined in the protocol."
            ), 0
    
        return None, "Ground truth type not supported.", 0  
    

    def check_non_significant_levels_after_and_ground_truth(              
        self,
        discontinuous_data,
        merged,
        nonsignificant_counts_before=None
    ):
        phase_number = 9
        phase_data = self.mapping_file.get("check_phase", {}).get(str(phase_number), {})
        phase_name = phase_data.get("name", f"Phase {phase_number}")
        warning_desc = phase_data.get("warnings", {})
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        num_warnings_only_inthisfunct = 0

        # check if discontinuous values are present
        has_discontinuous = (
            discontinuous_data is not None and
            not discontinuous_data.empty
        )
        
        gt_var = next(
            k for k, v in self.clinical_data.items()
            if isinstance(v, dict) and v.get("ground-truth") == "yes"
        )

        if has_discontinuous:
            nonsig_after, nonsig_string, w901_count_after = self._analyze_categorical_unbalance(
                discontinuous_data
            )
        else:
            nonsig_after = {}
            nonsig_string = ""
            w901_count_after = 0
    
        gt_warning_code, gt_message, gt_warning_count = self._analyze_ground_truth(
            gt_var, merged, nonsig_after, nonsig_string, warning_desc
        )
    
        warning_counts = {
            "W901": w901_count_after if w901_count_after > 0 else None,
            "W902": gt_warning_count if gt_warning_code == "W902" else None,
            "W903": gt_warning_count if gt_warning_code == "W903" else 0
        }
        
        if gt_warning_code == "W903": # GT is continuous
            num_warnings_only_inthisfunct = w901_count_after + gt_warning_count
        else:
            num_warnings_only_inthisfunct = w901_count_after

        if not has_discontinuous:
            non_significant_message_after = "Not applicable (no categorical variables)."
        elif w901_count_after == 0:
            non_significant_message_after = "Not present."
        else:
            non_significant_message_after = f"Warning W901: {warning_desc.get('W901', None)}\n"
            non_significant_message_after += f"Total Warnings: {w901_count_after}."

        categorical_levels_before = (
            nonsignificant_counts_before if has_discontinuous and nonsignificant_counts_before is not None else {}
        )
        categorical_levels_after = (
            nonsig_after if has_discontinuous else {}
        )
                
        generate_check_file_clinical_data(
            check_name=phase_name,
            phase_number=phase_number,
            total_rows=len(merged), 
            total_unique_patients=_strip_patient_suffix(merged.index).nunique(),
            timestamp=now,
            error_counts={},
            warning_counts=warning_counts,
            error_descriptions={},
            warning_descriptions=warning_desc,
            output_dir=self.output_directory_checkfile,
            categorical_levels_before=categorical_levels_before,
            categorical_levels_after=categorical_levels_after,
            include_categorical_levels=has_discontinuous,
            existing_warnings=retrieve_existing_warnings_clinical_data(
                os.path.join(self.output_directory_checkfile, "clinical_data_check_file.json")
            ),
        )
        # DEVO AGGIUGERE TRA GLI OUTPUT "non_significant_message_after"
        return nonsig_after, nonsig_string, w901_count_after, gt_message, non_significant_message_after, num_warnings_only_inthisfunct, gt_var
    

    def _export_preprocessed_outputs(
        self,
        categorical_df,
        continuous_df,
        merged_df,
        gt_var
    ):
        # -------------------------------
        # Save TOTAL data (without GT)
        # -------------------------------
        total_df = merged_df.drop(columns=[gt_var], errors="ignore").copy()
        # Strip patient suffixes only when saving
        total_df.index = _strip_patient_suffix(total_df.index)
        total_path = os.path.join(
            self.output_directory_data,
            "preprocessed_total_data.csv"
        )
        self.create_multi_header_df(total_df).to_csv(
            total_path, index=False, sep=";"
        )
    
        # -------------------------------
        # Save CONTINUOUS data
            # Derived from TOTAL to preserve
            # row alignment
        # -------------------------------
        categorical_predictors = (
            set(categorical_df.columns) - {gt_var}
            if not categorical_df.empty
            else set()
        )
        
        if not continuous_df.empty and categorical_predictors:
            # Remove GT if present
            cont_columns_to_save = [col for col in continuous_df.columns if col != gt_var]
            
            # Only save if there is at least one continuous column left
            if cont_columns_to_save:
                cont_df = total_df[cont_columns_to_save].copy()
                # Strip patient suffixes only when saving
                cont_df.index = _strip_patient_suffix(cont_df.index)
                cont_path = os.path.join(
                    self.output_directory_data,
                    "preprocessed_continuous_data.csv"
                )
                self.create_multi_header_df(cont_df).to_csv(
                    cont_path, index=False, sep=";"
                )
    
        # -------------------------------
        # Save GROUND TRUTH (if continuous)
        # -------------------------------
        protocol = self.clinical_data.get(gt_var, {})
        if protocol.get("type") in ("int", "float") and gt_var in merged_df.columns:
            gt_df = merged_df[[gt_var]].copy()
            # Strip patient suffixes only when saving
            gt_df.index = _strip_patient_suffix(gt_df.index)
            gt_path = os.path.join(
                self.output_directory_data,
                "ground_truth.csv"
            )
            self.create_multi_header_df(gt_df).to_csv(
                gt_path, index=False, sep=";"
            )   
    

    def encode_discontinuous_data(self, discontinuous_data, continuous_data=None): 
        """
        Encodes categorical variables in discontinuous_data based on specific rules while preserving column order.
        
        Parameters:
        - discontinuous_data (pd.DataFrame): DataFrame containing categorical variables.
        - continuous_data (pd.DataFrame, optional): DataFrame containing continuous variables.
        
        Returns:
        - pd.DataFrame: Encoded DataFrame with preserved column order.
        - str: Message summarizing encoding.
        """
        
        # Find the ground truth variable from the protocol
        ground_truth_variable = None
        for var_name, var_details in self.clinical_data.items():
            if isinstance(var_details, dict) and var_details.get("ground-truth") == "yes":
                ground_truth_variable = var_name
                break

        # Drop ground truth variable if categorical
        if ground_truth_variable:
            variable_details_gt = self.clinical_data.get(ground_truth_variable, {})
            if "cardinality" in variable_details_gt:
                discontinuous_data_without_gt = discontinuous_data.drop(columns=[ground_truth_variable])
            else:
                discontinuous_data_without_gt = discontinuous_data.copy()
        else:
            discontinuous_data_without_gt = discontinuous_data.copy()

        # ---------------------------------------------------
        # Early exit: nothing to encode (only GT was present)
        # ---------------------------------------------------
        if discontinuous_data_without_gt.empty:
            encoding_final_message = (
                "Encoding skipped:\n"
                "- Only the ground-truth variable is categorical; no predictors required encoding.\n"
            )
            return pd.DataFrame(), encoding_final_message
            
        encoded_df = discontinuous_data_without_gt.copy()
        new_columns = []  # Keep track of the new column order
        one_hot_count = 0
        binary_count = 0
        ordinal_count = 0
    
        for column in encoded_df.columns:
            if column in self.clinical_data:
                column_info = self.clinical_data[column]
                order = column_info.get("order", None)
                cardinality = column_info.get("cardinality", None)
                coding = column_info.get("coding", None)
    
                # If order is 0 and cardinality > 2 → One-hot encoding
                if order == 0 and cardinality and cardinality > 2:
                    one_hot = pd.get_dummies(encoded_df[column], prefix=column, dtype=int)
                    encoded_df.drop(columns=[column], inplace=True)  # Remove original column
                    encoded_df = pd.concat([encoded_df, one_hot], axis=1)  # Concatenate encoded columns
                    new_columns.extend(one_hot.columns)  # Keep track of new column names
                    one_hot_count += 1
                else:
                    new_columns.append(column)  # Keep track of existing column names
                
                # If order is 0 and cardinality == 2 → Binary encoding using "coding"
                if order == 0 and cardinality == 2 and coding:
                    encoded_df[column] = encoded_df[column].map(coding)
                    binary_count += 1
    
                # If order is 1 → Ordinal encoding using "coding"
                elif order == 1 and coding:
                    encoded_df[column] = encoded_df[column].map(coding)
                    ordinal_count += 1
                    
        # Initialize message to avoid UnboundLocalError
        encoding_final_message = ""

        # Reorder columns to maintain original sequence
        encoded_df = encoded_df.reindex(columns=new_columns)

        if not encoded_df.empty:
            # Strip patient suffixes before saving
            encoded_df_to_save = encoded_df.copy()
            encoded_df_to_save.index = _strip_patient_suffix(encoded_df_to_save.index)
            # Save encoded_discontinuous_data 
            encoded_discontinuous_data_path = os.path.join(self.output_directory_data, 'encoded_preprocessed_discontinuous_data.csv')
            encoded_data_multi_header = self.create_multi_header_df(encoded_df_to_save)
            encoded_data_multi_header.to_csv(encoded_discontinuous_data_path, index=False, sep=";")

            # Print concise grouped summary
            summary_parts = []
            if one_hot_count > 0:
                summary_parts.append(f"{one_hot_count} variables one-hot encoded")
            if binary_count > 0:
                summary_parts.append(f"{binary_count} variables binary encoded")
            if ordinal_count > 0:
                summary_parts.append(f"{ordinal_count} variables ordinally encoded")
    
            # Build final message
            encoding_final_message = "Encoding of categorical variables completed:\n"
            
            if one_hot_count > 0:
                encoding_final_message += (
                    f"- The categorical variables dataset expanded from {len(discontinuous_data_without_gt.columns)} to {len(encoded_df.columns)} columns due to one-hot encoding.\n"
                )
        
            encoding_final_message += (
                f"- Summary of encoding types: {', '.join(summary_parts)}.\n"
                f"- The encoded dataset was saved as 'encoded_preprocessed_discontinuous_data.csv'.\n"
            )

        # --------------------------------------
        # Merge with continuous data if relevant
        # --------------------------------------
        if continuous_data is not None and not continuous_data.empty:
            # Skip continuous data if the only column is the ground-truth
            continuous_cols_to_merge = [col for col in continuous_data.columns if col != ground_truth_variable]
            if continuous_cols_to_merge and not encoded_df.empty:
                merged_data_encoded = encoded_df.merge(continuous_data[continuous_cols_to_merge],
                                                       left_index=True, right_index=True)
                # Save merged CSV
                # Strip patient suffixes before saving
                merged_data_encoded_to_save = merged_data_encoded.copy()
                merged_data_encoded_to_save.index = _strip_patient_suffix(merged_data_encoded_to_save.index)
                merged_encoded_path = os.path.join(self.output_directory_data, 'encoded_preprocessed_total_data.csv')
                merged_multi_header = self.create_multi_header_df(merged_data_encoded_to_save)
                merged_multi_header.to_csv(merged_encoded_path, index=False, sep=";")
                encoding_final_message += "- The encoded discontinuous dataset was merged with continuous data and saved as 'encoded_preprocessed_total_data.csv'.\n" 
   
        return encoded_df, encoding_final_message
    

    def generate_report_after_preprocessing(
        self,
        success_message,
        transformation_message,
        added_variables=None,
        continuous_na_message=None,
        message_gt_NA=None,
        eliminated_rows_dict_ground_truth=None,
        num_deleted_rows_ground_truth=None,
        discontinuous_na_message=None,
        nonsignificant_counts_before_string=None,
        total_nonsignificant_before=None,
        non_significant_message_before=None,
        nonsignificant_counts_after_string=None,
        total_nonsignificant_after=None,
        non_significant_message_after=None,
        num_deleted_rows_discontinuous_data=None,
        list_deleted_rows_discontinuous=None,
        discontinuous_data=None,
        message_gt=None,
        num_warnings_only_inthisfunct=None,
        encoding_final_message=None
    ):
        
        """
        Generate a comprehensive data preprocessing report.
        Automatically handles missing continuous/discontinuous data sections.
        """
        # -----------------------------
        # Normalize optional inputs
        # -----------------------------
        added_variables = added_variables or []
        continuous_na_message = continuous_na_message or "Skipped: no continuous variables."
        discontinuous_na_message = discontinuous_na_message or "Skipped: no discontinuous variables."
        nonsignificant_counts_before_string = nonsignificant_counts_before_string or {}
        nonsignificant_counts_after_string = nonsignificant_counts_after_string or {}   
        non_significant_message_before = (
            non_significant_message_before or "Skipped: no discontinuous variables."
        )
        non_significant_message_after = (
            non_significant_message_after or "Skipped: no discontinuous variables."
        )
        total_nonsignificant_before = total_nonsignificant_before or 0
        total_nonsignificant_after = total_nonsignificant_after or 0    
        eliminated_rows_dict_ground_truth = eliminated_rows_dict_ground_truth or {}
        list_deleted_rows_discontinuous = list_deleted_rows_discontinuous or []   
        encoding_final_message = encoding_final_message or "Skipped: no discontinuous variables available for encoding."    
        num_deleted_rows_ground_truth = num_deleted_rows_ground_truth or 0
        num_deleted_rows_discontinuous_data = num_deleted_rows_discontinuous_data or 0

        has_discontinuous_data = discontinuous_data is not None and not discontinuous_data.empty
        
        # -----------------------------
        # Report header
        # -----------------------------
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")

        # Add the date of report generation
        report_content  = f"Report generated on: {formatted_datetime}\n\n"

        # Add the title of the report
        report_content  += "Data preprocessing report:\n\n"

        # Add a meaningful title for the success message
        report_content += "1. Patient ID Substitution Status:\n"
        report_content += f"{success_message}\n\n"

        # Add the transformation message
        report_content += "2. Transformations:\n"
        report_content += f"{transformation_message}\n\n"
    
        # Add information about added variables
        if added_variables:
            report_content += f"Added variables: {', '.join(added_variables)}\n\n"

        # Add continuous NA handling message
        report_content += "3. Handling missing data in continuous variables:\n"
        report_content += f"{continuous_na_message}\n\n"

        # Add handling NA in ground truth variable
        report_content += "4. Handling missing data in ground-truth variable:\n"
        if eliminated_rows_dict_ground_truth is None and num_deleted_rows_ground_truth == 0:
            report_content += f"{message_gt_NA}\n\n"
        else:
            # Add the number of missing values in the ground truth
            report_content += f"Number of missing values in ground-truth variable: {num_deleted_rows_ground_truth}\n"
            # Add information about eliminated rows in the ground truth
            if eliminated_rows_dict_ground_truth:
                report_content += "Eliminated rows:\n"
                for key, value in eliminated_rows_dict_ground_truth.items():
                    report_content += f"{key}: {value}\n"
            report_content += "\n"  # Ensure proper spacing
            
        # Add discontinuous NA handling message
        report_content += "5. Handling missing data in discontinuous variables:\n"
        report_content += f"{discontinuous_na_message}\n\n"

        # Add the information BEFORE preprocessing
        report_content += "6. Non significant levels BEFORE discontinuous missing data handling:\n"

        if not has_discontinuous_data:
            report_content += "N/A\n\n"
        elif not nonsignificant_counts_before_string and total_nonsignificant_before == 0:
            report_content += f"{non_significant_message_before}\n\n"
        else:
            for column, levels in nonsignificant_counts_before_string.items():
                if column != 'Total number of non significant levels':  # Exclude 'Total' key
                    report_content += f"\n{column}:\n"
                    for level in levels:
                        report_content += f"- {level}\n"
            report_content += f"\n{non_significant_message_before}\n\n"                   

        # Add the information after preprocessing
        report_content += "7. Non significant levels AFTER discontinuous missing data handling:\n"
        if not has_discontinuous_data:
            report_content += "N/A\n\n"
        elif discontinuous_na_message != "0 missing values.":
            if not nonsignificant_counts_after_string and total_nonsignificant_after == 0:
                report_content += f"{non_significant_message_after}\n\n"
            else:
                for column, levels in nonsignificant_counts_after_string.items():
                    if column != 'Total number of non significant levels':  # Exclude 'Total' key
                        report_content += f"\n{column}:\n"
                        for level in levels:
                            report_content += f"- {level}\n"
                report_content += f"\n{non_significant_message_after}\n\n"

            # Add the information about the number of deleted rows in discontinuous data
            report_content += f"Number of deleted rows in discontinuous data due to missing data handling: {num_deleted_rows_discontinuous_data}\n\n"
        
            # Add the list of deleted rows in discontinuous data
            if list_deleted_rows_discontinuous:
                report_content += "Deleted rows in discontinuous data due to missing data handling:\n"
                report_content += ', '.join(map(str, list_deleted_rows_discontinuous)) + "\n\n"
        else:
            report_content += "Skipped.\n\n"

        # 8. Add final numbers of discontinuous data after missing values handling
        report_content += "8. Final numbers of discontinuous data after missing values handling:\n"

        if not has_discontinuous_data:
            report_content += "N/A\n\n"
        else:
            num_unique_obs = discontinuous_data.index.nunique()
            num_replicated_obs = (
                discontinuous_data.index[discontinuous_data.index.duplicated()]
                .value_counts()
                .count()
            )
            num_deleted_obs_tot = (
                num_deleted_rows_ground_truth + num_deleted_rows_discontinuous_data
            )
        
            report_content += f"- total number of unique observations: {num_unique_obs}\n"
            report_content += f"- total number of replicated observations: {num_replicated_obs}\n"
            report_content += f"- total number of deleted observations: {num_deleted_obs_tot}\n\n"
        
        # Add ground truth information
        report_content += "9. Ground-truth information after preprocessing:\n"
        report_content += f"{message_gt}\n\n"
            
        # 8. Encoding Summary (NEW SECTION)
        report_content += "10. Encoding Summary:\n"
        report_content += f"{encoding_final_message}\n\n"

        # Add warning message based on the count of function-specific warnings
        if num_warnings_only_inthisfunct > 0:
            warning_message = f"Total Warnings detected: {num_warnings_only_inthisfunct}."
        else:
            warning_message = "No issues detected."

        # Append the warning message to the report content
        report_content += f"{warning_message}\n\n"
        report_name = "2.1.data_preprocessing_report.txt"
        # Define the path for the report file
        report_file_path = os.path.join(self.output_directory_report, report_name)
    
        # Save the report content to a file
        with open(report_file_path, "w") as report_file:
            report_file.write(report_content)
    
        return f"The report has been saved as {report_name} in {self.host_output_directory_report} folder."
    

    def exploratory_data_analysis(self, merged_data):
        """
        Perform exploratory data analysis (EDA) on a prepared dataset.
        Generates and saves an HTML summary report.
        """

        # Generate summary
        summary = dfSummary(merged_data)
        
        # Convert the interactive report to HTML
        html_report = summary.to_html()
        
        # Define the path for the output CSV file
        eda_report_path = os.path.join(self.output_directory_report, '2.2.eda_report.html')
        
        # Save the HTML report to the specified file
        with open(eda_report_path, "w") as file:
            file.write(html_report)
        
        return summary, eda_report_path
    

def run_clinical_data_preprocessing(protocol, local_config, simplified_df, num_dates, mapping_file, check_phase):

    # Define the input directory path with 
    input_dir = os.getenv("INPUT_DIR")

    # Clear log at the start of validation
    clear_log_file(input_dir, LOG_FILENAME)

    # Create an instance of ClinicalDataPreProcessing
    preprocessor = ClinicalDataPreProcessing(protocol, local_config, mapping_file)

    # Initialization
    nonsignificant_counts_before=None

    if check_phase < 10:

        print("Running substitute_patient_ids function") # check phase 8
        try:
            data, success_message = preprocessor.substitute_patient_ids(simplified_df)
        except Exception as e:
            log_error(input_dir, "substitute_patient_ids", e, LOG_FILENAME)
            print(f"An unexpected error occurred during substitute_patient_ids. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
            raise

        print("Running transformations function")
        try:
            transformation_message, added_variables, data = preprocessor.transformations(data, num_dates)
        except Exception as e:
            log_error(input_dir, "transformations", e, LOG_FILENAME)
            print(f"An unexpected error occurred during transformations. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
            raise

        print("Running handle_ground_truth_variable function")
        try:
            processed_data, eliminated_rows_dict_ground_truth, num_deleted_rows_ground_truth, message_gt_NA = preprocessor.handle_ground_truth_variable(data)
        except Exception as e:
            log_error(input_dir, "handle_ground_truth_variable", e, LOG_FILENAME)
            print(f"An unexpected error occurred during handle_ground_truth_variable. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
            raise

        print("Running discriminate_variables function")
        try:
            continuous_data, discontinuous_data = preprocessor.discriminate_variables(processed_data)
        except Exception as e:
            log_error(input_dir, "discriminate_variables", e, LOG_FILENAME)
            print(f"An unexpected error occurred during discriminate_variables. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
            raise
        
        if continuous_data.shape[1] > 0: # run just if there are continuous data
            print("Running handle_missing_continuous_values function")
            try:
                continuous_data, continuous_na_message = preprocessor.handle_missing_continuous_values(continuous_data)
            except Exception as e:
                log_error(input_dir, "handle_missing_continuous_values", e, LOG_FILENAME)
                print(f"An unexpected error occurred during handle_missing_continuous_values. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
                raise

        if discontinuous_data.shape[1] > 0: # run just if there are discontinuous data
            print("Running check_non_significant_levels_before function")
            try:
                nonsignificant_counts_before, nonsignificant_counts_before_string, total_nonsignificant_before, non_significant_message_before = preprocessor.check_non_significant_levels_before(discontinuous_data)
            except Exception as e:
                log_error(input_dir, "check_non_significant_levels_before", e, LOG_FILENAME)
                print(f"An unexpected error occurred during check_non_significant_levels_before. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
                raise

            print("Running handle_missing_discontinuous_values function")
            try:
                discontinuous_data, num_deleted_rows_discontinuous_data, deleted_indices_list_discontinuous, discontinuous_na_message = preprocessor.handle_missing_discontinuous_values(discontinuous_data, 4, continuous_data)
            except Exception as e:
                log_error(input_dir, "handle_missing_discontinuous_values", e, LOG_FILENAME)
                print(f"An unexpected error occurred during handle_missing_discontinuous_values. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
                raise    

        print("Running _merge_clinical_data function") 
        try:
            merged_data = preprocessor._merge_clinical_data(discontinuous_data, continuous_data)
        except Exception as e:
            log_error(input_dir, "_merge_clinical_data", e, LOG_FILENAME)
            print(f"An unexpected error occurred during _merge_clinical_data. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
            raise

        print("Running check_non_significant_levels_after_and_ground_truth function") 
        try:
            nonsignificant_counts_after, nonsignificant_counts_after_string, total_nonsignificant_after, mess_gt, non_significant_message_after, num_warnings_only_inthisfunct, gt_var = preprocessor.check_non_significant_levels_after_and_ground_truth(discontinuous_data, merged_data, nonsignificant_counts_before)
        except Exception as e:
            log_error(input_dir, "check_non_significant_levels_after_and_ground_truth", e, LOG_FILENAME)
            print(f"An unexpected error occurred during check_non_significant_levels_after_and_ground_truth. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
            raise

        print("Running _export_preprocessed_outputs function") 
        try:
            preprocessor._export_preprocessed_outputs(categorical_df=discontinuous_data, continuous_df=continuous_data, merged_df=merged_data, gt_var=gt_var)
        except Exception as e:
            log_error(input_dir, "_export_preprocessed_outputs", e, LOG_FILENAME)
            print(f"An unexpected error occurred during _export_preprocessed_outputs. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
            raise

        if discontinuous_data.shape[1] > 0:
            print("Running encode_discontinuous_data function")
            try:
                encoded_discontinuous_data, encoding_final_message = preprocessor.encode_discontinuous_data(discontinuous_data, continuous_data)
            except Exception as e:
                log_error(input_dir, "encode_discontinuous_data", e, LOG_FILENAME)
                print(f"An unexpected error occurred during encode_discontinuous_data. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
                raise

        # Generate reports
        report_kwargs = dict(
            success_message=success_message,
            transformation_message=transformation_message,
            added_variables=added_variables,
            message_gt_NA=message_gt_NA,
            eliminated_rows_dict_ground_truth=eliminated_rows_dict_ground_truth,
            num_deleted_rows_ground_truth=num_deleted_rows_ground_truth,
            message_gt=mess_gt,
            num_warnings_only_inthisfunct=num_warnings_only_inthisfunct
        )

        if continuous_data.shape[1] > 0:
            report_kwargs.update(
                continuous_na_message=continuous_na_message
            )
        
        if discontinuous_data.shape[1] > 0:
            report_kwargs.update(
                discontinuous_na_message=discontinuous_na_message,
                nonsignificant_counts_before_string=nonsignificant_counts_before_string,
                total_nonsignificant_before=total_nonsignificant_before,
                non_significant_message_before=non_significant_message_before,
                nonsignificant_counts_after_string=nonsignificant_counts_after_string,
                total_nonsignificant_after=total_nonsignificant_after,
                non_significant_message_after=non_significant_message_after,
                num_deleted_rows_discontinuous_data=num_deleted_rows_discontinuous_data,
                list_deleted_rows_discontinuous=deleted_indices_list_discontinuous,
                discontinuous_data=discontinuous_data,
                encoding_final_message=encoding_final_message
            )

        print("Running generate_report_after_preprocessing function")
        try:
            preprocessor.generate_report_after_preprocessing(**report_kwargs)
        except Exception as e:
            log_error(input_dir, "generate_report_after_preprocessing", e, LOG_FILENAME)
            print(f"An unexpected error occurred during generate_report_after_preprocessing. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
            raise

        print("Running exploratory_data_analysis function")
        try:
            _, _ = preprocessor.exploratory_data_analysis(merged_data)
        except Exception as e:
            log_error(input_dir, "exploratory_data_analysis", e, LOG_FILENAME)
            print(f"An unexpected error occurred during exploratory_data_analysis. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
            raise
    
    




        