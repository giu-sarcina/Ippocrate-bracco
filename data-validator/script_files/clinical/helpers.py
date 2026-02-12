import re
import io
import os
import json
import pandas as pd

def standardize_decimal_separators(file_path): 
    """Standardize all decimal separators to periods in a CSV file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Replace commas used as decimal separators with periods
    # Match numbers like 123,45 and replace the comma with a period
    standardized_content = re.sub(r'(\d+),(\d+)', r'\1.\2', content)
    
    # Use StringIO to simulate a file object from the standardized content
    standardized_file = io.StringIO(standardized_content)

    return standardized_file


def generate_check_file_clinical_data(
            check_name,
            phase_number,
            total_rows,
            total_unique_patients,
            timestamp,
            error_counts,
            warning_counts,   
            error_descriptions,
            warning_descriptions,
            output_dir,
            categorical_levels_before=None,
            categorical_levels_after=None,
            include_categorical_levels=False,
            existing_warnings=None
        ):
        """
        Helper function to generate a JSON check file for clinical data.
    
        Args:
            total_rows (int): Total number of rows in the dataset.
            total_unique_patients (int): Total number of unique patients.
            error_counts (dict): Dictionary of error codes and their counts.
            warning_counts (dict): Dictionary of warning codes and their counts.
            phase_number (int): Phase number of the current check.
            output_dir (str): Directory to save the JSON check file.
            check_name (str): Name of the check.
            error_descriptions (dict, optional): Dictionary of error descriptions.
            warning_descriptions (dict, optional): Dictionary of warning descriptions.
            categorical_levels_before (dict, optional): Significant levels of categorical variables before missing data handling.
            categorical_levels_after (dict, optional): Significant levels of categorical variables after missing data handling.
            include_categorical_levels (bool, optional): Whether to include categorical levels in the output.
            existing_warnings (list, optional): List of existing warnings to include in the check file.
        """
        
        # Construct error list
        errors = []
        for code, count in error_counts.items():
            if code == "E101":
                error_entry = {
                    "code": code,
                    "present": "yes" if count > 0 else "no",
                    "description": error_descriptions.get(code, "Unknown error code.")
                }
            else:
                error_entry = {
                    "code": code,
                    "count": count,
                    "description": error_descriptions.get(code, "Unknown error code.")
                }
            errors.append(error_entry)
    
        # Construct warning list
        new_warning_entries = []
        for code, count in warning_counts.items():
            if code == "W903":
                warning_entry = {
                    "code": code,
                    "present": "yes" if count > 0 else "no",
                    "description": warning_descriptions.get(code, "Unknown warning code.")
                }
            else:
                warning_entry = {
                    "code": code,
                    "count": count,
                    "description": warning_descriptions.get(code, "Unknown warning code.")
                }
            new_warning_entries.append(warning_entry)
    
        # Deduplicate: merge with existing warnings, avoid duplicates by code+description
        combined_warnings = existing_warnings.copy() if existing_warnings else []
    
        for new_entry in new_warning_entries:
            duplicate_found = any(
                new_entry["code"] == existing["code"] and
                new_entry["description"] == existing["description"]
                for existing in combined_warnings
            )
            if not duplicate_found:
                combined_warnings.append(new_entry)
    
        # Construct the final JSON structure
        check_file_data = {
            "name": check_name,
            "phase_number": phase_number,
            "metadata": {
                "total_rows": total_rows,
                "total_unique_patients": total_unique_patients,
                "timestamp": timestamp
            },
            "errors": errors if errors else None,
            "warnings": combined_warnings if combined_warnings else None
        }
    
        # Include categorical variable levels if requested
        if include_categorical_levels:
            check_file_data["categorical_variable_non_significant_levels"] = {
                "before_missing_data_handling": (
                    None if not categorical_levels_before else categorical_levels_before
                ),
                "after_missing_data_handling": (
                    None if not categorical_levels_after else categorical_levels_after
                )
            }
    
        # Write to file
        check_file_path = os.path.join(output_dir, "clinical_data_check_file.json")
        with open(check_file_path, "w") as json_file:
            json.dump(check_file_data, json_file, indent=4)


def retrieve_existing_warnings_clinical_data(check_file_path):
    """
    Retrieve the list of warnings from a clinical check file.

    Args:
        check_file_path (str): Full path to the clinical check JSON file.

    Returns:
        list: A list of dictionaries, each containing a warning (or empty list if not found).
    """
    
    try:
        with open(check_file_path, "r") as file:
            check_data = json.load(file)
            
            # Extract warnings
            warnings = check_data.get("warnings", [])
    
            return warnings
    except (FileNotFoundError, json.JSONDecodeError):
        # If the file doesn't exist or is invalid, return an empty dictionary
        return []
    
    
def overwrite_original_with_cleaned_data(single_header_data, multi_header_data, input_data_path): 
    """
    Overwrite the original dataset file with a cleaned version that uses a multi-level header 
    combining variable names and units. This ensures format consistency after cleaning operations.

    Args:
        single_header_data (pd.DataFrame): Cleaned data with a flat (single-level) header.
        multi_header_data (pd.DataFrame): Original data with a multi-level header (used to restore units).
        input_data_path (str): Path where the updated DataFrame should be saved.

    Returns:
        pd.DataFrame: DataFrame with multi-indexed columns that was saved.
    """
    
    try:
        # Extract the first and second level of the multi-header
        unit_mapping = dict(zip(
            multi_header_data.columns.get_level_values(0),  # Feature names
            multi_header_data.columns.get_level_values(1)   # Corresponding units
        ))

        # Ensure "Patient ID" is not the index
        if single_header_data.index.name == "Patient ID":
            single_header_data = single_header_data.reset_index()  # Convert index to column
            
        # Check if the number of columns matches between both DataFrames
        if single_header_data.shape[1] != multi_header_data.shape[1]:
            raise ValueError("Mismatch in number of columns between original dataset and cleaned data.")

        # **Create MultiIndex from column names and corresponding units**
        multi_index_columns = pd.MultiIndex.from_tuples(
            [(col, unit_mapping.get(col, "")) for col in single_header_data.columns]  # No names assigned
        )
    
        # **Apply MultiIndex to DataFrame**
        multi_index_df = pd.DataFrame(single_header_data.values, columns=multi_index_columns)

        # Save the updated DataFrame back to the original file path
        multi_index_df.to_csv(input_data_path, index=False, sep=";")
        print(f"Original dataset successfully overwritten at: {input_data_path}")

        return multi_index_df

    except Exception as e:
        print(f"Error during overwriting process: {e}")
        return None
    

def get_total_warnings_from_check_file(check_file_path): 
    """
    Reads the check file and calculates the total number of warnings.

    Args:
        check_file_path (str): Path to the JSON check file.

    Returns:
        int: Total number of warnings.
    """
    try:
        # Read and parse the JSON check file
        with open(check_file_path, "r") as file:
            check_file_data = json.load(file)
        
        # Extract the warnings list
        warnings = check_file_data.get("warnings", [])
        
        # Calculate the total number of warnings
        total_warnings = sum(warning.get("count", 0) if warning.get("count") is not None else 0 for warning in warnings)
        
        return total_warnings
    
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error reading or parsing the check file: {e}")
        return 0  # Return 0 warnings if the file can't be read or is invalid


def _strip_patient_suffix(index):
    """
    Removes visit / acquisition suffixes from patient IDs.
    Example: patient00001_TVGHQ_1 → patient00001_TVGHQ
    """
    return index.str.replace(r"_\d+$", "", regex=True)