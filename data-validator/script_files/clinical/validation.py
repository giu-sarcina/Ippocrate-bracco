import pandas as pd
import numpy as np
import os 
import csv
from datetime import datetime
import io
import re
from utils import save_state, load_state, clear_log_file, log_error
from .helpers import (
    standardize_decimal_separators,
    generate_check_file_clinical_data,
    retrieve_existing_warnings_clinical_data,
    overwrite_original_with_cleaned_data,
    get_total_warnings_from_check_file
)

LOG_FILENAME = "clinical_data_validation_error.log"

# Clinical data validation
class ClinicalDataValidationCheck:

    def __init__(self, protocol, local_config, mapping_file):
        self.protocol = protocol
        self.local_config = local_config
        self.mapping_file = mapping_file
        self.study_path = os.getenv("ROOT_NAME")
        self.input_dir = os.getenv("INPUT_DIR")
        self.host_input_dir = os.getenv("HOST_BASE_DATA_DIR")
        self.clinical_dir = os.path.join(self.input_dir, "CLINICAL")
        self.host_clinical_dir = os.path.join(self.host_input_dir, "CLINICAL")
        self.clinical_data = self.protocol.get("clinical data", {})
        self.file_name = self.local_config["Local config"]["clinical data"]["file name"]
        self.input_data_path = os.path.join(self.clinical_dir, self.file_name)
        self.state_file = os.path.join(self.input_dir, "clinical_validation_state.json")
        self.output_directory_report = os.path.join(self.input_dir, "Reports", "Clinical Data Reports")
        self.host_output_directory_report = os.path.join(self.host_input_dir, "Reports", "Clinical Data Reports")
        self.output_directory_checkfile = os.path.join(self.study_path, "CHECKFILE")
        os.makedirs(self.output_directory_report, exist_ok=True)
        os.makedirs(self.output_directory_checkfile, exist_ok=True)


    def load_data(self):
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")

        report_filename = f"0.data_loading_check_report.txt"
        # Initialize the report content
        report_content = f"Report generated on: {formatted_datetime}\n\n"
        report_content += "Data loading check report:\n\n"
        report_path = os.path.join(self.output_directory_report, report_filename)
        host_report_path = os.path.join(self.host_output_directory_report, report_filename)

        try:
            # Use Sniffer to detect the dialect of the CSV file
            with open(self.input_data_path, 'r', newline='', encoding='utf-8') as csvfile:
                sniffer = csv.Sniffer()
                sample = csvfile.read(4096).replace('\r', '').replace('\n', '')  # Clean up line endings  # Read a small sample of the file (maybe need to increase it)
                csvfile.seek(0)  # Reset file pointer to the beginning
                sniffer = csv.Sniffer()
                dialect = sniffer.sniff(sample)

                if dialect.delimiter == '\t':
                    sep = '\t'
                elif dialect.delimiter == ';':
                    sep = ';'  
                elif dialect.delimiter == ',':
                    sep = ','
                else:
                    # Handle case when delimiter is unknown
                    error_message = f"Unknown delimiter '{dialect.delimiter}' detected."
                    report_content += error_message
                    print(error_message)
                    raise ValueError(
                        f"{error_message} "
                        f"See the detailed report at: {host_report_path}") 

            # Check if commas are present in the dataset (but not as delimiters)
            with open(self.input_data_path, 'r', encoding='utf-8') as csvfile:
                content = csvfile.read()

            # Determine decimal separator
            if sep != ',' and ',' in content:
                # Standardize decimal separators if commas are present but not used as delimiters
                standardized_data = standardize_decimal_separators(self.input_data_path)
            else:
                # No need to standardize; wrap the content in StringIO
                standardized_data = io.StringIO(content)

            # --- Check if the header is two-layer
            # STEP 1: Read only the first two rows
            header_check_df = pd.read_csv(standardized_data, sep=sep, nrows=2, header=None) 

            # Reset StringIO to read full dataset again
            standardized_data.seek(0)

            # Check for missing cells in the second row
            missing_cells_second_row = header_check_df.iloc[1].isnull().sum()

            if missing_cells_second_row > 0:
                print("Two-level header with missing cells detected – likely correct.")

            else:
                # No missing cells → Proceed to check for duplicates
                second_row_unique = header_check_df.iloc[1].nunique()
                num_columns = len(header_check_df.columns)

                if second_row_unique < num_columns:
                    print("Two-level header with duplicates detected – likely correct.")
                    
                else:
                    # No duplicates → Check for numerical values
                    is_numeric = header_check_df.iloc[1].apply(lambda x: isinstance(x, (int, float)) or str(x).replace('.', '', 1).isdigit()).any()

                    if is_numeric:
                        error_message = "Error: The second header row contains numerical values; it should contain units of measurement."
                        report_content += error_message
                        print(error_message)
                        raise ValueError(
                            f"{error_message} "
                            f"See the detailed report at: {host_report_path}")
                    else:
                        print("Two-level header with unique, non-numeric values detected – likely correct.")
        
            data = pd.read_csv(standardized_data, sep=sep, decimal='.', header=[0, 1]) #decimal=decimal_sep
                
            print(f"Data loaded successfully from '{self.host_clinical_dir}' folder")
            report_content += f"Data loaded successfully from '{self.host_clinical_dir}' folder."
                
            return data
            
        except FileNotFoundError:
            error_message = f"Data file '{self.file_name}' in '{self.host_clinical_dir}' folder not found."
            report_content += error_message
            print(error_message)
            raise Exception(
                f"{error_message} "
                f"See the detailed report at: {host_report_path}")
        except pd.errors.EmptyDataError:
            error_message = f"Data file '{self.file_name}' in '{self.host_clinical_dir}' folder is empty."
            report_content += error_message
            print(error_message)
            raise Exception(
                f"{error_message} "
                f"See the detailed report at: {host_report_path}")
        except pd.errors.ParserError as e:
            error_message = f"Error parsing data file '{self.file_name}' in '{self.host_clinical_dir}' folder: {str(e)}"
            report_content += error_message
            print(error_message)
            raise Exception(
                f"{error_message} "
                f"See the detailed report at: {host_report_path}")
        finally:
            # Write the report whether an exception is raised or not
            with open(report_path, 'w') as report_file:
                report_file.write(report_content)


    def clean_multi_header_and_categorical_columns(self, multi_header_data):
        """
        Function to clean a multi-header DataFrame by stripping leading and trailing spaces
        from both the variable names (1st level) and the units of measure (2nd level) in the header.
        Also cleans the unique classes in categorical variables, and saves the cleaned DataFrame to the given path.
        
        Parameters:
            multi_header_data (pd.DataFrame): The multi-indexed DataFrame to clean.
        
        Returns:
            pd.DataFrame: The cleaned DataFrame with spaces removed from header columns and categorical variables.
        """   
        # Clean the header column levels (Variable Names and Units of Measure)
        column_level_0 = multi_header_data.columns.get_level_values(0).tolist()  # Variable names
        column_level_1 = multi_header_data.columns.get_level_values(1).tolist()  # Units of measure
        
        # Strip spaces from both the variable names and units of measure
        column_level_0 = [col.strip() for col in column_level_0]
        column_level_1 = [unit.strip() for unit in column_level_1]
        column_level_1 = ["" if str(col).startswith("Unnamed") else col for col in column_level_1]
        
        # Reassign the cleaned column names back to the DataFrame
        multi_header_data.columns = pd.MultiIndex.from_tuples(zip(column_level_0, column_level_1))
        
        # Clean categorical variables by stripping spaces in their unique values
        # Iterate over the column indices (multi-indexed columns)
        for i in range(len(multi_header_data.columns)):  # Use column index positions
            
            # Access the column data by its index
            column_data = multi_header_data.iloc[:, i]
            
            # Check if the column is categorical (dtype 'object')
            if column_data.dtype == 'object':
                # Strip spaces from the unique values in the categorical column
                cleaned_classes = column_data.unique()
                cleaned_classes = [str(class_value).strip() for class_value in cleaned_classes]
                
                # Update the column with cleaned values
                multi_header_data.iloc[:, i] = column_data.apply(
                    lambda x: str(x).strip() if isinstance(x, str) else x
                )
                
        # Save the cleaned DataFrame to the specified file path
        multi_header_data.to_csv(self.input_data_path, index=False, sep=";")
        
        return multi_header_data


    def explore_protocol_variable_types(self):
        continuous_variables = []
        discontinuous_variables = []
        date_variables = []

        # Iterate over the protocol
        for variable, details in self.clinical_data.items():
            # Check if the variable has a "type" attribute and if it's "int" or "float"
            if "type" in details and details["type"] in ["int", "float"]:
                continuous_variables.append((variable, details))

            # Check if the variable has a "cardinality" attribute
            elif "cardinality" in details:
                discontinuous_variables.append((variable, details))

            # Check if the variable has a "date_format" attribute
            elif "date_format" in details:
                date_variables.append(variable)

        # Count the number of each variable type
        num_continuous = len(continuous_variables)
        num_discontinuous = len(discontinuous_variables)
        num_dates = len(date_variables)

        print("Number of continuous variables:", num_continuous)
        print("Number of discontinuous variables:", num_discontinuous)
        print("Number of date variables:", num_dates)

        return num_continuous, num_discontinuous, num_dates, continuous_variables, discontinuous_variables, date_variables
    
    
    def process_multi_index_df(self, data):

        # Path to save data flattened single-level header (just column names)
        data_file_path = os.path.join(self.clinical_dir, "single_header_data.csv")
        # Path to save the multi-index header (no data)
        header_file_path = os.path.join(self.clinical_dir, "multi_index_header.csv")
        
        # 1. Create the simplified dataframe with only the first level as header
        first_level_columns = data.columns.get_level_values(0)  # First level (variable names)
        simplified_df = data.copy()
        simplified_df.columns = first_level_columns
        
        # Track occurrences of each column name to handle duplicates
        column_name_counts = {}
        modified_first_level_columns = []

        for col in first_level_columns:
            # Check if the column name has appeared before
            if col in column_name_counts:
                # Increment the count and append the new name with index
                column_name_counts[col] += 1
                modified_first_level_columns.append(f"{col}.{column_name_counts[col]}")
            else:
                # First occurrence, keep the name as is and initialize the count
                column_name_counts[col] = 0
                modified_first_level_columns.append(col)
                
        # Set the modified column names in simplified_df
        simplified_df.columns = modified_first_level_columns

        # ** Check if "Patient ID" exists as a column and set it as the index **
        if "Patient ID" in modified_first_level_columns:
            simplified_df.set_index("Patient ID", inplace=True)  # Set as index
            simplified_df.index.name = "Patient ID"  # Ensure index name is "Patient ID"

        # Save the single-header dataset
        simplified_df.to_csv(data_file_path, index=True, sep=';')
                
        # 2. Create the header dataframe
        second_level_columns = data.columns.get_level_values(1)  # Second level (units or descriptions)
        
        # Replace "Unnamed" values with None
        first_level_columns = [None if str(col).startswith("Unnamed") else col for col in first_level_columns]
        second_level_columns = [None if str(col).startswith("Unnamed") else col for col in second_level_columns]
        
        # Concatenate both levels as two rows
        header_df = pd.DataFrame([second_level_columns], 
                                 index=["Units"])
        
        # Set first_level_columns as the column names for header_df
        header_df.columns = first_level_columns

        # Automatically drop Patient ID-related columns from header_df (case-insensitive)
        # Identify columns to drop by checking if "Patient ID" or related names appear in the first level
        columns_to_drop = [
            col for col in first_level_columns
            if str(col).strip().lower() in ["patient id", "patient_id", "name", "patient name", "patient_name"]
        ]
    
        # Drop the identified columns from header_df
        header_df = header_df.drop(columns=columns_to_drop)

        if header_df is not None and not header_df.empty:
            # Save the multi-index header
            header_df.to_csv(header_file_path, index=True, sep=';')  # No index for header
            print(f"Header Data saved at: {header_file_path}")
        else:
            print("No header data to save.")
    
        return simplified_df, header_df
    

    def check_patient_id_column(self, multi_header_data): 
        """
        This function checks the patient ID column, modifies the index accordingly,
        and saves changes in both single_header_data.csv and the original multi-index file.
        """
        # Retrieve the phase number (in this case, 1 for "Patient ID column check")
        phase_number = 1  
        phase_data = self.mapping_file.get("check_phase", {}).get(str(phase_number), {})
        error_descriptions = phase_data.get("errors", {})
        phase_name = phase_data.get("name", f"Phase {phase_number}")  # Get the phase name for check title
        
        # Format check and report names dynamically
        check_name = phase_name.lower().replace(" ", "_")  # Format the check name (e.g., "Patient ID column check" -> "patient_id_column_check")
        report_name = f"1.001.{check_name}_report"  # Append "_report" for the report name

        simplified_data, _ = self.process_multi_index_df(multi_header_data)
        data_file_path = os.path.join(self.clinical_dir, "single_header_data.csv")

        # List of potential column names to check. You can add more!
        patient_id_columns = ["Name", "Patient Name", "Patient_ID", "Patient_name"]
            
         # Get the current datetime
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Initializations
        error_counts = {"E101": 0}  # Error "101" is for missing Patient ID column
        warning_counts = {}  # No warnings are handled in this function
        messages = ""

        try:
            # Step 1: Check if "Patient ID" is already the index
            if simplified_data.index.name == "Patient ID":
                print("Patient ID is set as the index column.")
                messages += "Patient ID is set as the index column."
            else:
                # Check for the presence of the specified columns
                found_column = None
                # Convert all DataFrame column names to lowercase
                lowercase_columns = [col.strip().lower() if isinstance(col, str) else col for col in simplified_data.columns]
                print(lowercase_columns)

                for column in patient_id_columns:
                    if column.lower() in lowercase_columns:
                        found_column = next((col for col in simplified_data.columns 
                                             if isinstance(col, str) and col.strip().lower() == column.strip().lower()), None)
                        break
                        
                print("found_column", found_column)
                if found_column: #if the there is a column with one of these names: ["Name", "Patient Name", "Patient_ID", "Patient_name"]
                    patient_id_column = simplified_data.loc[:, simplified_data.columns == found_column]
                    simplified_data.index = patient_id_column.iloc[:, 0]
                    # Rename the index to always be "Patient ID"
                    simplified_data.index.name = 'Patient ID'
                    # Drop the 'Patient ID' column since it's now the index
                    simplified_data.drop(columns=patient_id_column.columns, inplace=True)
                    # Search for the index of found_column in the list of top-level column values
                    patient_id_column_idx = multi_header_data.columns.get_level_values(0).tolist().index(found_column)
                    print("patient_id_column_idx", patient_id_column_idx)
                    # the column at the index patient_id_column_idx is renamed to "Patient ID"
                    # Extract both levels of the column index
                    column_level_0 = list(multi_header_data.columns.get_level_values(0))
                    column_level_1 = list(multi_header_data.columns.get_level_values(1))
                    
                    # Rename the specific column at patient_id_column_idx
                    column_level_0[patient_id_column_idx] = "Patient ID"
    
                    # Assign the modified column names back to the DataFrame
                    multi_header_data.columns = pd.MultiIndex.from_tuples(zip(column_level_0, column_level_1))

                    print(f"Column '{found_column}' has been set as the index and renamed to 'Patient ID'.")
                    message = f"Column '{found_column}' has been set as the index and renamed to 'Patient ID'."
                    messages += message
                else: 
                    # Check if "image" is present in the protocol
                    if "image" in self.protocol:
                        error_message = f"Error E101: {error_descriptions.get('E101', None)}"
                        print(error_message)
                        
                        # Get the current datetime
                        formatted_report = f"Report generated on: {current_datetime}\n\n"
                        formatted_report += f"{phase_name} report:\n\n"
                        formatted_report += error_message + "\n\n"  
                        formatted_report += f"Total Errors: 1"
                        error_counts["E101"] += 1  # Increment error count for error "E101"

                        # Save the report to the specified path
                        report_path = os.path.join(self.output_directory_report, f"{report_name}.txt")
                        host_report_path = os.path.join(self.host_output_directory_report, f"{report_name}.txt")
                        with open(report_path, "w") as file:
                            file.write(formatted_report)
                        raise ValueError(
                            f"{error_message} "
                            f"See the detailed report at: {host_report_path}")
                    else:
                        # If no patient ID columns and "image" not in protocol, create default index
                        simplified_data.index = range(1, len(simplified_data) + 1)
                        simplified_data.index.name = "Patient ID"
                        multi_header_data.insert(0, "Patient ID", simplified_data.index)
                        print("No patient ID column found. Default index created and set as 'Patient ID'.")
                        message = "No patient ID column found. Default index created and set as 'Patient ID'."
                        messages += message

            # Save data (clinical data with patient IDs)
            simplified_data.to_csv(data_file_path, index=True, sep=';')
            print(f"Single-header data saved at: {data_file_path}")
            
            second_level_columns = multi_header_data.columns.get_level_values(1)
            second_level_columns = ["" if str(col).startswith("Unnamed") else col for col in second_level_columns]
            
            # Set the updated second-level columns back into the DataFrame
            multi_header_data.columns = [multi_header_data.columns.get_level_values(0), second_level_columns]
            
            # Identify "Patient ID" column where second level is empty
            mask = (multi_header_data.columns.get_level_values(0) == "Patient ID") & (multi_header_data.columns.get_level_values(1) == "")

            # Ensure that at least one column matches the condition
            if any(mask):
                # Get the first matching column name (assuming there's only one "Patient ID" column)
                patient_id_col = multi_header_data.columns[mask][0]
                
                # Get the current position of "Patient ID" column
                current_position = multi_header_data.columns.get_loc(patient_id_col)
            
                # Move "Patient ID" to the first position only if it's not already there
                if current_position != 0:
                    new_order = list(multi_header_data.columns[mask]) + [col for col in multi_header_data.columns if col not in multi_header_data.columns[mask]]
                    multi_header_data = multi_header_data[new_order]
                    print("Moved 'Patient ID' column of multi-header dataframe to the first position.")
                else:
                    print("'Patient ID' column of multi-header dataframe is already in the first position.")

            multi_header_data.to_csv(self.input_data_path, index=False, sep=";")
 
            print(f"Original data updated at: {self.input_data_path}")
  
        finally:
            # Generate the check file using `generate_check_file_clinical_data`
            total_rows = len(simplified_data)
            total_unique_patients = simplified_data.index.nunique()  # Number of unique Patient IDs
            generate_check_file_clinical_data(
                check_name=phase_name,
                phase_number=phase_number,
                total_rows=total_rows,
                total_unique_patients=total_unique_patients,
                timestamp=current_datetime,
                error_counts=error_counts,
                warning_counts=warning_counts, 
                error_descriptions=error_descriptions,
                warning_descriptions={},  # No warnings in this check
                output_dir=self.output_directory_checkfile
            )
        return multi_header_data, messages
    

    def load_single_header_data(self):
        data_file_path = os.path.join(self.clinical_dir, "single_header_data.csv")
        data = pd.read_csv(data_file_path, sep=";", index_col=0)
        return data
    
    
    def adjust_missing_values(self, data):

        phase_number = 2  
        phase_data = self.mapping_file.get("check_phase", {}).get(str(phase_number), {})
        error_descriptions = phase_data.get("errors", {})
        phase_name = phase_data.get("name", f"Phase {phase_number}")  
        
        # Format check and report names dynamically
        check_name = phase_name.lower().replace(" ", "_")  
        report_name = f"1.002.{check_name}_report"  
        
        report_NA = ""
        # Get current date and time
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Handle missing values like NaN, None, or NA
        missing_values_report = ""
        expected_value = self.clinical_data.get("Missing Values", {}).get("expected_value")
        
        # Track Missing Value Positions
        missing_positions = []
    
        # Identify NaN, None, or empty fields and store their positions
        missing_mask = data.isna()
        for column in data.columns:
            missing_indices = data[missing_mask[column]].index.tolist()
            if missing_indices:
                missing_positions.extend([(column, index) for index in missing_indices])  # +1 for human-readable row position
        
        # Count missing values (NaN, None, or NA)
        missing_values_count = data.isna().sum().sum()
        if missing_values_count > 0:
            missing_values_report += f"- Details: Missing values like NaN, None, NA or empty fields found. Replace them with: {expected_value}.\n"
            for column, position in missing_positions:
                missing_values_report += f"  Variable: {column}, Patient ID: {position}\n"

        # Handle values like '.' or '-'
        other_possible_values = self.clinical_data.get("Missing Values", {}).get("other_possible_values", [])
        replace_values_report = ""

        # Track Other Possible Values Positions
        replace_values_positions = []
        for value in other_possible_values:
            mask = data.isin([value])
            for column in data.columns:
                indices = data[mask[column]].index.tolist()
                if indices:
                    replace_values_positions.extend([(column, index) for index in indices])

        other_values_count = sum(data.isin([value]).sum().sum() for value in other_possible_values)        
        if other_values_count > 0:
            replace_values_report += f"\n- Details: Values like {other_possible_values} found. Replace them with: {expected_value}.\n"
            for column, position in replace_values_positions:
                replace_values_report += f"  Variable: {column}, Patient ID: {position}\n"
            
        # Count other possible values
        if missing_values_count == 0 and other_values_count == 0:
            # Check if expected_values are specified in Missing Values of the protocol
            if expected_value is not None and data.isin([expected_value]).any().any():
                report_NA_success = "Missing values correctly identified according to the protocol."
            else:
                report_NA_success = "0 missing data found according to the protocol."
            #return report_NA_success
        
        # Combine reports
        total_missing_data = missing_values_count + other_values_count
        total_wrong_format_message = f"\nTotal Errors: {total_missing_data}\n"
        report_NA += missing_values_report + replace_values_report

        # Save report to a text file if there are missing values or specific values found
        if report_NA:    
            report_NA = f"Report generated on: {current_datetime}\n\n" + f"{phase_name} report:\n\n" + f"Error E201: {error_descriptions.get('E201', None)}\n" + report_NA + total_wrong_format_message
            missing_values_report_path = os.path.join(self.output_directory_report, f"{report_name}.txt")
            host_missing_values_report_path = os.path.join(self.host_output_directory_report, f"{report_name}.txt")
            with open(missing_values_report_path, "w") as report_file:
                report_file.write(report_NA)

        # Prepare error and warning counts for check file
        error_counts = {}
        warning_counts = {}
        if total_missing_data > 0:
            error_counts["E201"] = int(total_missing_data)  # "E201": Invalid missing values detected.
        else:
            error_counts["E201"] = None

        # Generate the check file using `generate_check_file_clinical_data`
        generate_check_file_clinical_data(
            check_name=phase_name,
            phase_number=2,
            total_rows=len(data),
            total_unique_patients=data.index.nunique(),
            timestamp=current_datetime,
            error_counts=error_counts,
            warning_counts=warning_counts,
            error_descriptions=error_descriptions,
            warning_descriptions={},  # No warnings in this check
            output_dir=self.output_directory_checkfile,
        )
    
        # Raise an error if there are invalid missing values
        if total_missing_data > 0:
            raise RuntimeError(
                f"Missing values need to be replaced with {expected_value}. "
                f"See the detailed report at: {host_missing_values_report_path}")
             
        return report_NA_success
    

    def verify_data_vs_protocol_compliance(self, data, header_df):  #MODIFICATA

        # Retrieve the phase number (in this case, 1 for "Patient ID column check")
        phase_number = 3  
        phase_data = self.mapping_file.get("check_phase", {}).get(str(phase_number), {})
        error_descriptions = phase_data.get("errors", {})
        phase_name = phase_data.get("name", f"Phase {phase_number}")  
        
        # Format check and report names dynamically
        check_name = phase_name.lower().replace(" ", "_") 
        report_name = f"1.003.{check_name}_report"  
        
        compliance_report = {}
        formatted_report = ""
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Initialize counters for each type of issue
        missing_columns_count = 0
        extra_columns_count = 0
        column_order_mismatch_count = 0
        data_type_mismatches_count = 0
        warning_counts = {}
        warning_descriptions = {}
  
        # --- Missing and Extra Columns ---
        expected_columns = list(self.clinical_data.keys())
        expected_columns = [col for col in expected_columns if col not in ["Patient ID", "Missing Values", "Transformations"]] # ATTENZIONE QUI SE SI AGGIUNGONO NUOVE CHIAVI NEL PROTOCOLLO
        actual_columns = list(data.columns)

        # Identify missing and extra columns
        missing_columns = [col for col in expected_columns if col not in actual_columns]
        extra_columns = [col for col in actual_columns if col not in expected_columns]
    
        if missing_columns or extra_columns:
            error_details = {}
            if missing_columns:
                error_details['Missing Columns'] = {col: self.clinical_data[col]['position'] for col in missing_columns}
                missing_columns_count = len(missing_columns)
            if extra_columns:
                error_details['Extra Columns'] = {col: actual_columns.index(col) + 1 for col in extra_columns}
                extra_columns_count = len(extra_columns)
    
            compliance_report[f"Error E301: {error_descriptions.get('E301', None)}"] = {'Details': error_details}


        total_issues_part1 = missing_columns_count + extra_columns_count
        
        # Prepare error and warning counts and descriptions
        error_counts = {
            "E301": total_issues_part1 if total_issues_part1 > 0 else None,  # Missing or extra columns
            "E302": None,  
            "E303": None,
            "E304": None,
        }
        
        # Call the generate_check_file_clinical_data function if there are errors
        if total_issues_part1 > 0:
            generate_check_file_clinical_data(
                check_name=phase_name,
                phase_number=3,
                total_rows=len(data),
                total_unique_patients=data.index.nunique(),
                timestamp=current_datetime,
                error_counts=error_counts,
                warning_counts=warning_counts,
                error_descriptions=error_descriptions,
                warning_descriptions=warning_descriptions,
                output_dir=self.output_directory_checkfile
            )
            
            # Prepare the formatted report
            formatted_report += f"Report generated on: {current_datetime}\n\n"
            formatted_report += f"{phase_name} report:\n\n"
            for key, value in compliance_report.items():
                formatted_report += f"{key}\n"
                if isinstance(value, list):
                    for item in value:
                        formatted_report += f"- {item}\n"
                elif isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        formatted_report += f"- {subkey}: {subvalue}\n"
                else:
                    formatted_report += f"- {value}\n"
                formatted_report += "\n"
    
            formatted_report += f"Total Errors: {total_issues_part1}\n\n"
    
            # Write the formatted report to a text file
            protocol_compliance_report_path = os.path.join(self.output_directory_report, f"{report_name}.txt")
            host_protocol_compliance_report_path = os.path.join(self.host_output_directory_report, f"{report_name}.txt")
            with open(protocol_compliance_report_path, "w") as file:
                file.write(formatted_report)
    
            # Raise an error with details for the user
            raise ValueError(f"Errors found! Total number of errors: {total_issues_part1}. Check the report for details at {host_protocol_compliance_report_path} and solve them!")

        # --- Units of measure compliance ---
        unit_of_measure_mismatches = []
        unit_of_measure_mismatches_count = 0

        for column in header_df.columns:
            expected_unit = self.clinical_data.get(column, {}).get('unit_of_measure')
            actual_unit = header_df.loc["Units", column]

            # If the expected unit is None, replace any actual unit with None
            if expected_unit is None or expected_unit == "none":
                expected_unit = None 
                actual_unit = None 

            if expected_unit:
                if actual_unit is None or str(expected_unit).strip() != str(actual_unit).strip():
                    unit_of_measure_mismatches.append({
                        'Column': column,
                        'Expected Unit': expected_unit,
                        'Actual Unit': actual_unit if actual_unit is not None else None
                    })
                    unit_of_measure_mismatches_count += 1
        
        # Add unit mismatches to the compliance report
        if unit_of_measure_mismatches:
            compliance_report[f"Error E302: {error_descriptions.get('E302', None)}"] = {
                'Details': unit_of_measure_mismatches
            }
            
        # Update error counts
        error_counts["E302"] = unit_of_measure_mismatches_count if unit_of_measure_mismatches_count > 0 else None 
        
        # --- Column Order Verification ---
        excluded_keys = ['Patient ID', 'Missing Values', 'Transformations']  # Add more keys if needed
        expected_columns = sorted(
            [col for col in self.clinical_data.keys() if col not in excluded_keys],
            key=lambda col: self.clinical_data[col].get('position', float('inf'))
        )
        #common_variables = set(expected_order).intersection(actual_columns)
        actual_columns = list(col for col in data.columns)  #list(self.remove_parentheses_text(col) for col in data.columns)

        # Initialize list to store column order mismatches
        column_order_mismatches = []

        for index, (expected_column, actual_column) in enumerate(zip(expected_columns, actual_columns)):
            if expected_column != actual_column:
                #print(f"Mismatch at position {index + 1}: '{expected_column }' != '{actual_column}'")

                column_order_mismatches.append({
                    'Column': expected_column,
                    'Expected Position': index + 1,
                    'Actual Position': actual_columns.index(expected_column) + 1
                })
        
        # Add list of column order mismatches to the compliance report
        if column_order_mismatches:
            compliance_report[f"Error E303: {error_descriptions.get('E303', None)}"] = {
                'Details': column_order_mismatches
            }
            column_order_mismatch_count = len(column_order_mismatches)
        else:
            column_order_mismatch_count = 0

        # --- Data Type Verification ---
        expected_NA_value = self.clinical_data.get("Missing Values", {}).get("expected_value")
        for column, data_type in self.clinical_data.items():
            if column in data.columns:
                
                # SKIP categorical variables (handled later in check_categorical_variables)
                is_categorical = (
                    isinstance(data_type, dict)
                    and data_type.get("cardinality") is not None
                )
                if is_categorical:
                    continue

                expected_data_type = data_type.get('type') if isinstance(data_type, dict) else data_type
              
                # Initialize a list to store data type mismatches for each column
                column_data_type_mismatches = []

                # Get actual data types in the column (ignoring expected_NA_value)
                actual_data_types = set(type(value).__name__ for value in data[column] if value != expected_NA_value)

                for index, value in data[column].items():
                    # Check if the value is a boolean, and convert to string if so
                    if isinstance(value, bool):
                        value = str(value)  # Convert to "True" or "False
                        
                # Check for column-wide mismatches if all values are of a single type and differ from expected type
                # Only report if expected data type is 'str' and actual data type is 'int' or 'float'
                if expected_data_type == 'str' and len(actual_data_types) == 1 and next(iter(actual_data_types)) in ['int', 'float']:
                    compliance_report.setdefault(f"Error E304: {error_descriptions.get('E304', None)}", {'Details': {}})
                    compliance_report[f"Error E304: {error_descriptions.get('E304', None)}"]['Details'][column] = [
                        f"The entire {column} variable is composed of {next(iter(actual_data_types))} type elements, instead of {expected_data_type} type elements"
                    ]
                    continue
                    
                # Check if expected data type is int or float, but the column is actually str and does not contain numeric strings
                if expected_data_type in ['int', 'float'] and len(actual_data_types) == 1 and 'str' in actual_data_types:
                    numeric_strings = []  # List to accumulate if a string is numeric (either int or float)
                    
                    for value in data[column]:
                        if isinstance(value, str):
                            # Exclude '-999' from being treated as a valid numeric string
                            if value == '-999':
                                numeric_strings.append(True)  # Treat '-999' as a valid value (so it won't trigger the error)
                                continue
                            
                            if expected_data_type == 'int' and not re.fullmatch(r'^-?\d+$', value):
                                numeric_strings.append(True)
                            elif expected_data_type == 'float' and not re.fullmatch(r'^-?\d+(\.\d+)?$', value):
                                numeric_strings.append(True)
                            else:
                                numeric_strings.append(False)  # This string is not numeric
 
                    if all(value is True for value in numeric_strings):
                        compliance_report.setdefault(f"Error E304: {error_descriptions.get('E304', None)}", {'Details': {}})
                        compliance_report[f"Error E304: {error_descriptions.get('E304', None)}"]['Details'][column] = [
                            f"The entire {column} variable is composed of str type elements, instead of {expected_data_type} type elements"
                        ]
                        continue
                            
                for index, value in data[column].items():  
                    if isinstance(value, bool):
                        value = str(value)
                    if value != expected_NA_value:
                        actual_data_type = type(value).__name__
                        mismatch = False
    
                        if expected_data_type in ['int', 'float']:
                            # If expected data type is int or float, check for non-numeric strings
                            if isinstance(value, str):
                                if expected_data_type == 'int' and not re.fullmatch(r'^-?\d+$', value):
                                    mismatch = True
                                    actual_data_type = 'str'
                                elif expected_data_type == 'float' and not re.fullmatch(r'^-?\d+(\.\d+)?$', value):
                                    mismatch = True
                                    actual_data_type = 'str'
                            elif expected_data_type == 'int' and not isinstance(value, int):
                                mismatch = True
                            elif expected_data_type == 'float' and not isinstance(value, (int, float)):
                                mismatch = True
                        elif expected_data_type == 'str':
                            # If expected data type is str, check for numeric or float representations
                            if isinstance(value, str):
                                # Exclude '-999' from being treated as a numeric value
                                if value == '-999':
                                    continue  # Skip further checks if the value is '-999'
                                    
                                if re.fullmatch(r'^-?\d+$', value):
                                    mismatch = True
                                    actual_data_type = 'int'
                                elif re.fullmatch(r'^-?\d+(\.\d+)?$', value):
                                    mismatch = True
                                    actual_data_type = 'float'
                            else:
                                mismatch = True
                                actual_data_type = type(value).__name__
                        else:
                            mismatch = actual_data_type != expected_data_type
        
                        if mismatch:
                            row_position = index 
                            mismatch_message = {
                                'Expected': expected_data_type,
                                'Actual': actual_data_type,
                                'Row Index': row_position,
                                'Value': value  # Add the actual wrong value here
                            }
                            column_data_type_mismatches.append(mismatch_message)
                    
                    # Add the list of data type mismatches to the compliance report for the column
                    if column_data_type_mismatches:
                        compliance_report.setdefault(f"Error E304: {error_descriptions.get('E304', None)}", {'Details': {}})
                        compliance_report[f"Error E304: {error_descriptions.get('E304', None)}"]['Details'][column] = column_data_type_mismatches
        
        data_type_mismatches_count = len(compliance_report.get(f"Error E304: {error_descriptions.get('E304', None)}", {}).get('Details', {}))        
        if data_type_mismatches_count > 0:
            compliance_report.setdefault(f"Error E304: {error_descriptions.get('E304', None)}", {'Details': {}})
                                    
        total_issues_part2and3 = (
            unit_of_measure_mismatches_count +
            column_order_mismatch_count + 
            data_type_mismatches_count 
        )
            
        # Call generate_check_file_clinical_data for error reporting
        error_counts = {
            "E301": None,  # Missing or extra columns
            "E302": unit_of_measure_mismatches_count if unit_of_measure_mismatches_count > 0 else None,
            "E303": column_order_mismatch_count if column_order_mismatch_count > 0 else None,  # Incorrect variable sequence
            "E304": data_type_mismatches_count if data_type_mismatches_count > 0 else None  # Data type mismatch
        }

        if total_issues_part2and3 > 0:
            # Call the generate_check_file_clinical_data function
            generate_check_file_clinical_data(
                check_name=phase_name,
                phase_number=3,
                total_rows=len(data),
                total_unique_patients=data.index.nunique(),
                timestamp=current_datetime,
                error_counts=error_counts,
                warning_counts=warning_counts,
                error_descriptions=error_descriptions,
                warning_descriptions=warning_descriptions,
                output_dir=self.output_directory_checkfile
            )

            formatted_report += f"Report generated on: {current_datetime}\n\n"
            formatted_report += f"{phase_name} report:\n\n"
            #formatted_report += "Errors:\n"  # Add the "Errors:" heading here
            for key, value in compliance_report.items():
                formatted_report += f"{key}\n"
                if isinstance(value, list):
                    for item in value:
                        formatted_report += f"- {item}\n"
                elif isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        formatted_report += f"- {subkey}: {subvalue}\n"
                else:
                    formatted_report += f"- {value}\n"
                formatted_report += "\n"

            formatted_report += f"Total Errors: {total_issues_part2and3}\n\n"

            # Write the formatted report to a text file
            protocol_compliance_report_path = os.path.join(self.output_directory_report, f"{report_name}.txt")
            host_protocol_compliance_report_path = os.path.join(self.host_output_directory_report, f"{report_name}.txt")
            with open(protocol_compliance_report_path, "w") as file:
                file.write(formatted_report)
                
            raise ValueError(f"Errors found! Total number of errors: {total_issues_part2and3}. Check the report for details at {host_protocol_compliance_report_path} and solve them!")
            
        else:
            compliance_success = "The data adheres to the protocol's variable names, units of measure, sequence, and data types."
            
            # Call generate_check_file_clinical_data for success (no errors)
            error_counts = {
                "E301": None,
                "E302": None,
                "E303": None,
                "E304": None,
            }
    
            # Call the generate_check_file_clinical_data function
            generate_check_file_clinical_data(
                check_name=phase_name,
                phase_number=3,
                total_rows=len(data),
                total_unique_patients=data.index.nunique(),
                timestamp=current_datetime,
                error_counts=error_counts,
                warning_counts=warning_counts,
                error_descriptions=error_descriptions,
                warning_descriptions=warning_descriptions,
                output_dir=self.output_directory_checkfile
            )
        
            return compliance_success
        
    
    def check_numerical_variables(self, data, num_continuous): 
        
        phase_number = 4  
        phase_data = self.mapping_file.get("check_phase", {}).get(str(phase_number), {})
        error_descriptions = phase_data.get("errors", {})
        warning_descriptions = phase_data.get("warnings", {})
        phase_name = phase_data.get("name", f"Phase {phase_number}")  
        
        # Format check and report names dynamically
        check_name = phase_name.lower().replace(" ", "_") 
        report_name = f"1.004.{check_name}_report"  
        #warning_counts = {"W401": 0}
        if num_continuous != 0:
            current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            total_suspected_val = 0
            suspected_values_indices = {}
            continuous_check_message = ""  # Initialize the continuous check message
            total_num_warnings = 0  # Initialize warning counter
            
            # Retrieve expected NA value from the protocol
            expected_NA_value = self.clinical_data.get("Missing Values", {}).get("expected_value")
    
            for key, value in self.clinical_data.items():
                # Skip irrelevant keys
                if key == "Patient ID" or key == "Missing Values" or key == "Transformations": # ATTENZIONE QUI SE SI AGGIUNGONO NUOVE CHIAVI NEL PROTOCOLLO
                    continue
                
                if value["type"] == "int" or value["type"] == "float":
                    
                    if "condition" in value:
                        conditions = value["condition"]
                        column_data = data[key]
                        suspected_values = {}
                        for idx, val in enumerate(column_data, start=1):
                            if val != expected_NA_value and pd.notnull(val):
                                #mismatch = False
                                if "expected_range" in conditions:
                                    range_conditions = conditions["expected_range"]
                                    for condition in range_conditions:
                                        if not eval(f"{val} {condition}"):
                                            #mismatch = True
                                            total_suspected_val += 1
                                            if val not in suspected_values:
                                                suspected_values[val] = []
                                            suspected_values[val].append(idx)  # Add 1 to index to match 1-based indexing in the output
                                            break
                        if suspected_values:
                            suspected_values_indices[key] = suspected_values
    
                        # Check for expected variability
                        if "expected_variability" in conditions:
                            variability_condition = conditions["expected_variability"][0]
                            mean_val = np.mean(column_data)
                            std_dev = np.std(column_data)
                            
                            # Handle case where mean value is zero
                            if mean_val == 0:
                                coefficient_of_variation = 0  # or you could set it to float('nan') or some other value
                            else:
                                coefficient_of_variation = (std_dev / mean_val) * 100
                                
                            # Check if the coefficient of variation meets the specified condition
                            if not eval(f"{coefficient_of_variation} {variability_condition.strip('%')}"):
                                #warnings_check_passed = False
                                total_num_warnings += 1
                                #print("Total warnings:", total_num_warnings)
                                continuous_check_message += f"Warning: The expected variability for variable '{key}' is not {variability_condition}. Actual: {coefficient_of_variation:.2f}%. \n"
          
    
            # Generate errors and warnings independently
            error_counts = {"E401": total_suspected_val if total_suspected_val > 0 else None}
            warning_counts = {"W401": total_num_warnings if total_num_warnings > 0 else None}
    
            # Call the generate_check_file_clinical_data function
            generate_check_file_clinical_data(
                check_name=phase_name,
                phase_number=4,
                total_rows=len(data),
                total_unique_patients=data.index.nunique(),
                timestamp=current_datetime,
                error_counts=error_counts,
                warning_counts=warning_counts,
                error_descriptions=error_descriptions,
                warning_descriptions=warning_descriptions,
                output_dir=self.output_directory_checkfile
            )

            if suspected_values_indices or total_num_warnings > 0:
                # Write suspected values to report file
                report_filename = os.path.join(self.output_directory_report, f"{report_name}.txt")
                host_report_filename = os.path.join(self.host_output_directory_report, f"{report_name}.txt")
                with open(report_filename, "w") as report_file:
                    report_file.write(f"Report generated on: {current_datetime}\n\n")
                    report_file.write(f"{phase_name} report:\n\n")
                    if suspected_values_indices:
                        report_file.write(f"Error E401: {error_descriptions.get('E401', None)}\n")
                        report_file.write(f"- Details: Suspected continuous values: {str(suspected_values_indices)}\n\n")
                        report_file.write(f"Total Errors: {total_suspected_val}\n\n")
                    
                    if total_num_warnings > 0:
                        report_file.write(f"Warning W401: {warning_descriptions.get('W401', None)}\n")
                        report_file.write("- Details:\n")
                        for line in continuous_check_message.splitlines():
                            report_file.write(f"  {line}\n")
                        report_file.write(f"\nTotal Warnings: {total_num_warnings}\n")
    
            # Determine if any errors were found
            if total_suspected_val != 0:
                raise RuntimeError(
                    f"Suspected continuous values have been found. " 
                    f"See the detailed report at: {host_report_filename}")
    
            # Determine the final message for warnings
            if total_num_warnings == 0 and total_suspected_val == 0:
                continuous_check_message = "Continuous variables check passed: conditions met according to the protocol."
            elif total_num_warnings > 0 and total_suspected_val == 0:
                continuous_check_message += f"Number of warnings: {total_num_warnings}"
    
            return continuous_check_message
    
        else:
            continuous_check_message = "No continuous variables present, skipping check."
            return continuous_check_message
        

    def check_categorical_variables(self, data, num_discontinuous, multi_header_data): 

        phase_number = 5  
        phase_data = self.mapping_file.get("check_phase", {}).get(str(phase_number), {})
        error_descriptions = phase_data.get("errors", {})
        warning_descriptions = phase_data.get("warnings", {})
        phase_name = phase_data.get("name", f"Phase {phase_number}")  
        
        # Format check and report names dynamically
        check_name = phase_name.lower().replace(" ", "_") 
        report_name = f"1.005.{check_name}_report" 
        
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        warning_counts = {"W501": 0}

        check_file_path = os.path.join(self.output_directory_checkfile, "clinical_data_check_file.json")
        input_data_path = self.input_data_path
        
        # Retrieve existing warnings with descriptions from the check file
        existing_warnings = retrieve_existing_warnings_clinical_data(check_file_path)
        synonym_substitution_occurred = False
     
        # Identify the ground truth variable
        ground_truth_variable = None
        for var_name, var_details in self.clinical_data.items():
            if isinstance(var_details, dict) and var_details.get("ground-truth") == "yes":
                ground_truth_variable = var_name
                break
                
        if num_discontinuous != 0:
            categorical_report = {}
            warning_report = {}
            non_compliant_var_count = 0  # Initialize issues count
            warning_var_count = 0
    
            expected_NA_value = self.clinical_data.get("Missing Values", {}).get("expected_value")
            # -999 is interpreted as a string in a column of categorical variables
            expected_NA_value = str(expected_NA_value)
    
            # Get the columns that are both in the data and specified in the protocol
            common_columns = set(data.columns).intersection(self.clinical_data.keys())
    
            for column in common_columns:
                specs = self.clinical_data[column]
    
                # Check if the column is categorical 
                if isinstance(specs, dict) and 'cardinality' in specs:
                    expected_values = specs.get('expected_values', [])  # Get the expected unique values from the protocol
                    
                    # Convert column and expected values to string for consistency
                    data[column] = data[column].astype(str)
                    expected_values = [str(value) for value in expected_values]
                    synonymous_values = specs.get('synonymous_values', {})  # Get the synonymous values from the protocol

                    # Store original column to check for changes
                    original_column = data[column].copy()
    
                    # Replace synonymous values with expected values in the data
                    data[column] = data[column].apply(lambda x: next((key for key, value in synonymous_values.items() if x in value), x))

                    # Check if substitution occurred
                    if not data[column].equals(original_column):
                        synonym_substitution_occurred = True
    
                    # Combine the expected unique values with the synonymous values
                    unique_values = expected_values.copy()  # Start with a copy of expected_values
                    for synonym_values in synonymous_values.values():
                        for synonym in synonym_values:
                            if synonym not in unique_values:
                                unique_values.append(synonym)
    
                    report_data = {}
                    # Check if the key 'cardinality' is present in the protocol for the variable
                    if 'cardinality' in specs:
                        # Get the expected cardinality from the protocol
                        expected_cardinality = specs.get('cardinality')
                        
                        # Calculate the actual number of unique values in the data for the column
                        actual_unique_values = [value for value in data[column].dropna().unique() if value != expected_NA_value]
                        actual_cardinality = len(actual_unique_values)
                        
                        # Find the mismatching values
                        mismatching_values_in_protocol = [value for value in actual_unique_values if value not in unique_values] 
                        mismatching_values_in_data = [value for value in expected_values if value not in actual_unique_values]
    
                        # Check if the actual cardinality matches the expected cardinality
                        if expected_cardinality != actual_cardinality or mismatching_values_in_protocol or mismatching_values_in_data:
                            report_data = {
                                'Expected': expected_cardinality,
                                'Actual': actual_cardinality,
                            }
                            
                            # Populate extra and missing labels if mismatches are found
                            if mismatching_values_in_protocol:
                                report_data['Extra labels'] = mismatching_values_in_protocol
                            if mismatching_values_in_data:
                                report_data['Missing labels'] = mismatching_values_in_data
                                
                            # Check if the actual cardinality matches the expected cardinality
                            if report_data['Expected'] != report_data['Actual'] or 'Extra labels' in report_data or 'Missing labels' in report_data:
                                # Handling for ground truth variable
                                if column == ground_truth_variable:
                                    # For ground-truth variable, Extra labels go to Errors, Missing labels go to Warnings
                                    if 'Extra labels' in report_data:
                                        categorical_report[column] = {
                                            'Expected': report_data['Expected'],
                                            'Actual': report_data['Actual'],
                                            'Extra labels': report_data['Extra labels']
                                        }
                                        non_compliant_var_count += 1
        
                                    if 'Missing labels' in report_data:
                                        warning_report[column] = {
                                            'Expected': report_data['Expected'],
                                            'Actual': report_data['Actual'],
                                            'Missing labels': report_data['Missing labels']
                                        }
                                        warning_var_count += 1
        
                                # Handling for non-ground-truth variables
                                else:
                                    # For non-ground-truth variables, both extra and missing labels go to Errors (E501)
                                    if 'Extra labels' in report_data or 'Missing labels' in report_data:
                                        categorical_report[column] = report_data
                                        non_compliant_var_count += 1
    
            # Generate the check file using the helper function
            error_counts = {"E501": non_compliant_var_count if non_compliant_var_count > 0 else None}
            warning_counts = {"W501": warning_var_count if warning_var_count > 0 else None}
            total_rows = len(data)
            total_unique_patients = len(data.index.unique())
            
            generate_check_file_clinical_data(
                check_name=phase_name,
                phase_number=phase_number,
                total_rows=total_rows,
                total_unique_patients=total_unique_patients,
                timestamp=current_datetime,
                error_counts=error_counts,
                warning_counts=warning_counts,
                error_descriptions=error_descriptions,
                warning_descriptions=warning_descriptions,
                output_dir=self.output_directory_checkfile,
                existing_warnings=existing_warnings
            )
            
            
            if non_compliant_var_count == 0 and warning_var_count == 0:
                if synonym_substitution_occurred:
                   overwrite_original_with_cleaned_data(data, multi_header_data, input_data_path) 
                categorical_check_message = "Discontinuous variables check passed: discontinuous variables meet the cardinality and levels indicated in the protocol."
                return categorical_check_message
            
            else:
                report_filename = os.path.join(self.output_directory_report, f"{report_name}.txt")
                host_report_filename = os.path.join(self.host_output_directory_report, f"{report_name}.txt")
                with open(report_filename, "w") as report_file:
                    report_file.write(f"Report generated on: {current_datetime}\n\n")
                    report_file.write(f"{phase_name} report:\n\n")
                    
                    if categorical_report:
                        report_file.write(f"Error E501: {error_descriptions.get('E501', None)}\n")
                        report_file.write("- Details:\n")
                        report_file.write(f"  {str(categorical_report)}")
                        report_file.write("\n\nTotal Errors: ")
                        report_file.write(str(non_compliant_var_count))
                        report_file.write("\n\n")
                
                    if warning_report:
                        report_file.write(f"Warning W501: {warning_descriptions.get('W501', None)}\n")
                        report_file.write("- Details:\n")
                        report_file.write(f"  {{'{ground_truth_variable}' (ground-truth): {str(warning_report[ground_truth_variable])}}}\n")
                        report_file.write("\nTotal Warnings: ")
                        report_file.write(str(warning_var_count))
    
                if non_compliant_var_count > 0:
                    raise RuntimeError(
                        f"Suspected discontinuous values have been found. "
                        f"See the detailed report at: {host_report_filename}")
                    
                if warning_var_count > 0:
                    categorical_check_message = f"Warning: Missing one or more classes in the ground-truth variable. Number of missing classes: {warning_var_count}"
                    
                return categorical_check_message

        else:
            categorical_check_message = "No discontinuous variables present, skipping categorical variable check."
            return categorical_check_message
        

    def check_date_variables(self, data, num_dates): 
        """
        Check if dates in a DataFrame are valid according to the protocol.

        Args:
        - data (DataFrame): The DataFrame containing the date columns.

        Returns:
        - list: List of messages indicating the consistency of dates.
        """
        
        phase_number = 6  
        phase_data = self.mapping_file.get("check_phase", {}).get(str(phase_number), {})
        error_descriptions = phase_data.get("errors", {})
        phase_name = phase_data.get("name", f"Phase {phase_number}")  
        
        # Format check and report names dynamically
        check_name = phase_name.lower().replace(" ", "_") 
        report_name = f"1.006.{check_name}_report"  
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        check_file_path = os.path.join(self.output_directory_checkfile, "clinical_data_check_file.json")
        
        # Retrieve existing warnings with descriptions from the check file
        existing_warnings = retrieve_existing_warnings_clinical_data(check_file_path)
        
        if num_dates != 0:
        # Initialize a list to store messages
            messages = []
            # Initialize counter for dates in wrong format
            incorrect_format_count = 0
            columns_with_errors = set()
            expected_NA_value = self.clinical_data.get("Missing Values", {}).get("expected_value")

            # -999 is interpreted as a string in a column of categorical variables
            expected_NA_value = str(expected_NA_value)
            
            # Iterate over items in the protocol
            for column_name, column_data in self.clinical_data.items():
                # Check if the item has the key 'date_format'
                if 'date_format' in column_data:
                    # Get the date format from the protocol
                    date_format = column_data['date_format']
                    
                    # Check if the column is defined in the DataFrame
                    if column_name in data.columns:
                        date_column = data[column_name]
    
                        # Get the integer location of the column
                        column_index = data.columns.get_loc(column_name)
                        
                        # Iterate over rows in the current column
                        for idx, date_str in date_column.items():
                            if pd.isnull(date_str) or date_str == expected_NA_value: # Skip NaN values
                                continue
                            try:
                                datetime.strptime(date_str, date_format)
                            except ValueError:
                                # Compute the 1-based row number
                                row_number = data.index.get_loc(idx) + 1
                                # Append the error to the messages list
                                messages.append(
                                    f"Invalid date format at row {row_number}, column '{column_name}' ({column_index + 1}): "
                                    f"'{date_str}'"
                                )
                                incorrect_format_count += 1
                                # Track the column with an error
                                columns_with_errors.add(column_name)
                            
            # Calculate the number of unique variables with at least one mismatch
            unique_error_columns_count = len(columns_with_errors)

            # Generate the check file using the helper function 
            error_counts = {"E601": unique_error_columns_count if unique_error_columns_count > 0 else None} 
            warning_counts = {}
            total_rows = len(data)
            total_unique_patients = len(data.index.unique())
            warning_descriptions={}

            generate_check_file_clinical_data(
                check_name=f"{phase_name}",
                phase_number=phase_number,
                total_rows=total_rows,
                total_unique_patients=total_unique_patients,
                timestamp=current_datetime,
                error_counts=error_counts,
                warning_counts=warning_counts,
                error_descriptions=error_descriptions,
                warning_descriptions=warning_descriptions,
                output_dir=self.output_directory_checkfile,
                existing_warnings=existing_warnings
            )
            
            # If no inconsistencies were found, add a message indicating that all dates are in the correct format
            if incorrect_format_count == 0:
                date_check_message = "Date type variables check passed: all dates are in the correct format according to the protocol."
                return date_check_message
            else:                    
                report_filename = os.path.join(self.output_directory_report, f"{report_name}.txt")
                host_report_filename = os.path.join(self.host_output_directory_report, f"{report_name}.txt")
                with open(report_filename, "w") as report_file:
                    report_file.write(f"Report generated on: {current_datetime}\n\n")
                    report_file.write(f"{phase_name} report:\n\n")
                    report_file.write(f"Error E601: {error_descriptions.get('E601', None)}\n")
                    report_file.write(f"- Details: Expected format '{date_format}'.\n")
                    for message in messages:
                        report_file.write(f"  {message}\n")
                    report_file.write(f"\nNumber of unique variables (columns) with mismatched date formats: {unique_error_columns_count}\n\n")
                    report_file.write(f"Total Errors: {incorrect_format_count}\n\n")
                
                raise RuntimeError(
                    f"Date format inconsistencies found. "
                    f"See the detailed report at: {host_report_filename}")
        else:
            date_check_message = "No date variables present, skipping date format validation."
            return date_check_message
        

    def check_replicates(self, data, multi_header_data): 
        """
        Checks for replicates (duplicate rows) in the dataset and handles them.
        
        Args:
            data (DataFrame): The clinical dataset to check for replicates.
        
        Returns:
            str: A message indicating whether replicates were found and handled.
        """
        phase_number = 7  
        phase_data = self.mapping_file.get("check_phase", {}).get(str(phase_number), {})
        warning_descriptions = phase_data.get("warnings", {})
        phase_name = phase_data.get("name", f"Phase {phase_number}")  
        
        # Format check and report names dynamically
        check_name = phase_name.lower().replace(" ", "_") 
        report_name = f"1.007.{check_name}_report" 

        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        warning_counts = {"W701": 0}

        check_file_path = os.path.join(self.output_directory_checkfile, "clinical_data_check_file.json")
        input_data_path = self.input_data_path
        
        # Retrieve existing warnings with descriptions from the check file
        existing_warnings = retrieve_existing_warnings_clinical_data(check_file_path)
        
        # Check for replicates (duplicate rows)
        duplicate_data = data[data.duplicated()]
        total_rows = data.shape[0]
        total_unique_patients = data.index.nunique()  # Assuming there is a "PatientID" column

        if not duplicate_data.empty:
            # Replicates found: Generate report and remove duplicates
            report_filename = f"{report_name}.txt"
            report_filepath = os.path.join(self.output_directory_report, report_filename)
    
            report_content = []
            report_content.append(f"Report generated on: {current_datetime}\n")
            report_content.append(f"{phase_name} report:\n")
            report_content.append(f"Warning W701: {warning_descriptions.get('W701', None)}")

            # Get boolean mask of duplicated rows
            duplicated_mask = data.duplicated(keep='first')
            
            # Build list of entries like: "ID10001 (row 12)"
            duplicate_details = [
                f"{data.index[i]} (row {i + 1})"
                for i in range(len(data))
                if duplicated_mask.iloc[i]
            ]

            # Add to report
            report_content.append(f"- Details: Replicated and removed rows: {', '.join(duplicate_details)}\n")
            
            # Add total warning count
            report_content.append(f"Total Warnings: {duplicate_data.shape[0]}")
    
            # Write the report content to the file
            with open(report_filepath, 'w') as report_file:
                report_file.write("\n".join(report_content))
    
            print(f"Replicates check report saved to {report_filepath}")
            
            # Remove duplicates from data
            data = data.drop_duplicates()
            total_rows = data.shape[0]
            total_unique_patients = data.index.nunique()  # Assuming there is a "PatientID" column
    
            # NB: Save the modified dataset to overwrite the single_header_data.csv file
            single_header_data_path = os.path.join(self.clinical_dir, "single_header_data.csv")
            data.to_csv(single_header_data_path, index=True, sep=";")
            print(f"Modified dataset saved to {single_header_data_path}")
    
            num_replicates = duplicate_data.shape[0]
            warning_counts["W701"] = num_replicates  # Define the warning code for replicates

             # Load the original multi-header dataset
            #multi_header_data = pd.read_csv(self.input_data_path, sep=";", header=[0, 1])

            multi_header_data = overwrite_original_with_cleaned_data(data, multi_header_data, input_data_path)
    
            # Generate the check file using the helper function
            generate_check_file_clinical_data(
                check_name=phase_name,
                phase_number=7,
                total_rows=total_rows,
                total_unique_patients=total_unique_patients,
                timestamp=current_datetime,
                error_counts={},
                warning_counts=warning_counts,
                error_descriptions={},
                warning_descriptions=warning_descriptions,
                output_dir=self.output_directory_checkfile,
                existing_warnings=existing_warnings
            )

            # Return the message indicating replicates were found and removed
            replicate_message = f"Warning: Replicates found and removed. Number of removed replicates: {num_replicates}"
            return data, multi_header_data, replicate_message
        
        else:
            print("No replicates found.")
    
            # No replicates
            warning_counts["W701"] = None  # Define the warning code for replicates
            
            # Generate the check file using the helper function
            generate_check_file_clinical_data(
                check_name=phase_name,
                phase_number=7,
                total_rows=total_rows,
                total_unique_patients=total_unique_patients,
                timestamp=current_datetime,
                error_counts={},
                warning_counts=warning_counts,
                error_descriptions={}, 
                warning_descriptions=warning_descriptions,  
                output_dir=self.output_directory_checkfile,
                existing_warnings=existing_warnings
            )
            
            replicate_message = "No replicates found."
            return data, multi_header_data, replicate_message


    def generate_final_check_report(self, messages, report_NA_success, compliance_success, continuous_check_message, categorical_check_message, date_check_message, no_replicate_message):
        """
        Generates the final data quality check report and saves it to a file.
    
        Args:
            messages (str): Message about Patient ID column check.
            report_NA_success (str): Message about missing data check.
            compliance_success (str): Message about protocol compliance check.
            continuous_check_message (str): Message about continuous variables check.
            categorical_check_message (str): Message about categorical variables check.
            date_check_message (str): Message about date variables check.
            no_replicate_message (str): Message about replicates check.
        """  
        # Get the current datetime
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
        
        # Read the check file to get the total_num_warnings (last but one digit)
        check_file_path = os.path.join(self.output_directory_checkfile, "clinical_data_check_file.json")
        total_num_warnings = get_total_warnings_from_check_file(check_file_path)

        # Construct the final report
        final_report_lines = [
            f"Report generated on: {formatted_datetime}",
            "",
            "Final report about data protocol compliance and quality check:",
            "",
            "Patient ID column check:",
            f"- {messages}",
            "",
            "Missing data check:",
            f"- {report_NA_success}",
            "",
            "Protocol compliance check:",
            f"- {compliance_success}",
            "",
            "Continuous variables check:"
        ]
        
        # Handle continuous_check_message with or without '\n'
        if '\n' in continuous_check_message:
            lines = continuous_check_message.split('\n')
            for line in lines:
                if line.strip():  # Check if the line is not empty after stripping
                    final_report_lines.append(f"- {line.strip()}")
        else:
            if continuous_check_message.strip():  # Check if the message is not empty after stripping
                final_report_lines.append(f"- {continuous_check_message}")
        
        final_report_lines.extend([
            "",
            "Discontinuous variables check:",
            f"- {categorical_check_message}",
            "",
            "Date variables check:",
            f"- {date_check_message}",
            "",
            "Replicates check:",
            f"- {no_replicate_message}",
            ""
        ])

        # Conditional message based on total_num_warnings
        if total_num_warnings > 0:
            warning_message = f"Total Warnings detected: {total_num_warnings}."
            final_report_lines.append(warning_message)
        else:
            final_report_lines.append("No issues detected.")
        
        final_report = "\n".join(final_report_lines)
        
        # Define the report filename
        report_filename = os.path.join(self.output_directory_report, "1.final_data_quality_check_report.txt")
        
        # Write the final report to a file
        with open(report_filename, "w") as report_file:
            report_file.write(final_report)
        
        # Print a message indicating that the final report has been generated
        print("Final check report has been generated.")


def run_clinical_data_check(protocol, local_config, mapping_file, check_phase):

    # Define the input directory path  
    input_dir = os.getenv("INPUT_DIR")

    # Clear log at the start of validation
    clear_log_file(input_dir, LOG_FILENAME)

    # Create an instance of ClinicalDataValidationCheck
    validation_check = ClinicalDataValidationCheck(protocol, local_config, mapping_file)

    # Load the state if it exists
    state = load_state(validation_check.state_file)

    try:
        print("Running load_data function...")
        data = validation_check.load_data()
    except Exception as e:
        log_error(input_dir, "load_data", e, LOG_FILENAME)
        print(f"An unexpected error occurred during load_data. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
        raise
    
    try:
        print("Running clean_multi_header_and_categorical_columns function...")
        data = validation_check.clean_multi_header_and_categorical_columns(data)
    except Exception as e:
        log_error(input_dir, "clean_multi_header_and_categorical_columns", e, LOG_FILENAME)
        print(f"An unexpected error occurred during clean_multi_header_and_categorical_columns. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
        raise
    
    try:
        print("Running process_multi_index_df function...")
        simplified_df, header_df = validation_check.process_multi_index_df(data)
    except Exception as e:
        log_error(input_dir, "process_multi_index_df", e, LOG_FILENAME)
        print(f"An unexpected error occurred during process_multi_index_df. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
        raise

    try:
        print("Running explore_protocol_variable_types function...")
        num_continuous, num_discontinuous, num_dates, _, _, _ = validation_check.explore_protocol_variable_types()
    except Exception as e:
        log_error(input_dir, "explore_protocol_variable_types", e, LOG_FILENAME)
        print(f"An unexpected error occurred during explore_protocol_variable_types. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
        raise

    # Check Patient ID column 
    if check_phase < 2:  
        print("Running Check Patient ID Column function...") # check phase 1
        try:
            data, state["patientID_message"] = validation_check.check_patient_id_column(data)
            save_state(state, validation_check.state_file)  # Save updated state
        except Exception as e:
            log_error(input_dir, "check_patient_id_column", e, LOG_FILENAME)
            print(f"An unexpected error occurred during check_patient_id_column. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
            raise  

    if check_phase < 3:
        print("Running Adjust Missing Values function") # check phase 2
        try:
            state["missing_value_message"] = validation_check.adjust_missing_values(simplified_df)
            save_state(state, validation_check.state_file)
        except Exception as e:
            log_error(input_dir, "adjust_missing_values", e, LOG_FILENAME)
            print(f"An unexpected error occurred during adjust_missing_values. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
            raise 

    # Verify Protocol Compliance
    if check_phase < 4:
        print("Running Verify Data vs Protocol Compliance function") # check phase 3
        try:
            state["compliance_check"] = validation_check.verify_data_vs_protocol_compliance(simplified_df, header_df)
            save_state(state, validation_check.state_file)
        except Exception as e:
            log_error(input_dir, "verify_data_vs_protocol_compliance", e, LOG_FILENAME)
            print(f"An unexpected error occurred during verify_data_vs_protocol_compliance. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
            raise 
    
    # Check numerical variables   
    if check_phase < 5:
        print("Running Check Numerical Variables function") # check phase 4
        try:
            continuous_check_message = validation_check.check_numerical_variables(simplified_df, num_continuous)
            state["continuous_check_message"] = continuous_check_message
            save_state(state, validation_check.state_file)
        except Exception as e:
            log_error(input_dir, "check_numerical_variables", e, LOG_FILENAME)
            print(f"An unexpected error occurred during check_numerical_variables. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
            raise 

    # Check categorical variables
    if check_phase < 6:
        print("Running Check Categorical Variables function") # check phase 5
        try:
            state["discontinuous_check_message"] = validation_check.check_categorical_variables(simplified_df, num_discontinuous, data)
            save_state(state, validation_check.state_file)
        except Exception as e:
            log_error(input_dir, "check_categorical_variables", e, LOG_FILENAME)
            print(f"An unexpected error occurred during check_categorical_variables. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
            raise 

    # Check date variables
    if check_phase < 7:
        print("Running Check Date Variables function") # check phase 6
        try:
            state["date_format_check_message"] = validation_check.check_date_variables(simplified_df, num_dates)
            save_state(state, validation_check.state_file)
        except Exception as e:
            log_error(input_dir, "check_date_variables", e, LOG_FILENAME)
            print(f"An unexpected error occurred during check_date_variables. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
            raise 

    if check_phase < 8:
        print("Running Check Replicates function") # check phase 7
        try:
            simplified_df, data, state["replicate_message"] = validation_check.check_replicates(simplified_df, data)
            save_state(state, validation_check.state_file)
        except Exception as e:
            log_error(input_dir, "check_replicates", e, LOG_FILENAME)
            print(f"An unexpected error occurred during check_replicates. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
            raise 

        print("Running generate_final_check_report function")
        try:
            validation_check.generate_final_check_report(
                state.get("patientID_message"),
                state.get("missing_value_message"),
                state.get("compliance_check"),
                state.get("continuous_check_message"),
                state.get("discontinuous_check_message"),
                state.get("date_format_check_message"),
                state.get("replicate_message")
            )
        except Exception as e:
            log_error(input_dir, "generate_final_check_report", e, LOG_FILENAME)
            print(f"An unexpected error occurred during generate_final_check_report. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
            raise 

    return simplified_df, data, num_dates


    


    
