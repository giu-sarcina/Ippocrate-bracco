# ─────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────
import os
import json
import nibabel as nib
import nrrd
import pandas as pd
import numpy as np
import csv
import cv2
import re
import torchio as tio
from pydicom.tag import Tag
from pydicom.sequence import Sequence
from pydicom.dataset import Dataset

# ─────────────────────────────────────────────────────────
# Image Utilities
# ─────────────────────────────────────────────────────────      

def generate_check_file_image_data(  
        check_name,
        phase_number,
        series_group_name,
        num_slices_group,
        num_patients_with_image_data,
        num_series,
        num_tot_slices,
        timestamp,
        error_counts,
        warning_counts,   
        error_descriptions,
        warning_descriptions,
        output_dir,
        existing_warnings=None
    ):
    """
    Generates a JSON check file for input data validation.

    This function creates a standardized check file named 'image_data_check_file.json' in the specified output 
    directory. It includes validation results (errors, warnings), metadata for a specific image series group, and 
    global metadata for all processed series. If a file with the same name already exists, it will be overwritten.

    Args:
        check_name (str): Name of the check process or file.
        phase_number (int): Phase number associated with the image data validation.
        series_group_name (str): Name or identifier of the specific group of image series being validated.
        num_slices_group (int): Number of image slices in this specific group.
        num_patients_with_image_data (int): Total number of patients with imaging data available (global).
        num_series (int): Total number of image series processed (global).
        num_tot_slices (int): Total number of image slices processed (global).
        timestamp (str): Timestamp of the check, typically in the format "YYYY-MM-DD HH:MM:SS".
        error_counts (dict): Dictionary of error codes and their corresponding counts.
        warning_counts (dict): Dictionary of warning codes and their corresponding counts.
        error_descriptions (dict): Mapping of error codes to their descriptions.
        warning_descriptions (dict): Mapping of warning codes to their descriptions.
        output_dir (str): Directory where the check file will be saved.
        existing_warnings (list, optional): List of existing warnings to include in the file.

    Returns:
        None: The function writes the generated JSON file to the specified output directory as 
        'image_data_check_file.json', overwriting any existing file with that name.
        
    """
    # Construct error list
    errors = [
        {
            "code": code,
            "count": count,
            "description": error_descriptions.get(code, "Unknown error code.")
        }
        for code, count in error_counts.items()
    ] if error_counts else None

    # Construct new warning list for this specific series_group
    new_warning_entries = [
        {
            "code": code,
            "count": int(count) if count is not None else None,
            "description": warning_descriptions.get(code, "Unknown warning code.")
        }
        for code, count in warning_counts.items()
    ] if warning_counts else []  #if any(warning_counts.values()) else []

    # Combine with existing warnings (grouped by series group)
    warnings = existing_warnings.copy() if existing_warnings else {}

    if new_warning_entries:
        if series_group_name not in warnings:
            warnings[series_group_name] = []
    
        existing_entries = warnings[series_group_name]
    
        # Add only unique warnings
        for new_entry in new_warning_entries:
            duplicate_found = any(
                new_entry["code"] == existing["code"] and
                new_entry["description"] == existing["description"]
                for existing in existing_entries
            )
            if not duplicate_found:
                existing_entries.append(new_entry)

    # Construct the final JSON structure
    check_file_data = {
        "name": check_name,
        "phase_number": phase_number,
        "series_group": series_group_name,
        "num_slices_group": int(num_slices_group),
        "global metadata": {
            "total_patients_with_image_data": int(num_patients_with_image_data),
            "total_series": int(num_series),
            "total_slices": int(num_tot_slices),
            "timestamp": timestamp
        },
        "errors": errors,
        "warnings": warnings if warnings else None
    }

    # Write to file
    check_file_path = os.path.join(output_dir, "image_data_check_file.json")
    with open(check_file_path, "w") as json_file:
        json.dump(check_file_data, json_file, indent=4)


def retrieve_existing_warnings_image_data(check_file_path): 
    """
    Retrieve the warnings section from a check file.
    :return: A dictionary of warnings grouped by series group, or None if originally null.
    """
    try:
        with open(check_file_path, "r") as file:
            check_data = json.load(file)
            
            warnings = check_data.get("warnings", None)

            if warnings is None:
                return None  # Preserve null if that's what was originally written

            if isinstance(warnings, dict):
                return warnings

            # If it's a list (old format), wrap in a default key
            return {"legacy_warnings": warnings}

    except (FileNotFoundError, json.JSONDecodeError):
        return None  # If file is missing or corrupted, treat it as no warnings
    

def count_tot_num_slices_per_group(series_path, image_file_format): 
    """Counts the number of slices in the given series folder based on the image file format."""
    try:
        if image_file_format in [".nii", ".nii.gz"]:
            nifti_file = [f for f in os.listdir(series_path) if f.endswith('.nii')][0]
            nifti_path = os.path.join(series_path, nifti_file)
            img = nib.load(nifti_path)
            num_slices = img.shape[-1]
            return num_slices  # Number of slices is the last dimension
        elif image_file_format == ".nrrd":
            nrrd_file = [f for f in os.listdir(series_path) if f.endswith('.nrrd')][0]
            nrrd_path = os.path.join(series_path, nrrd_file)
            header = nrrd.read_header(nrrd_path)
            num_slices = header.get('sizes', [])[2]
            return num_slices  # Assuming the 3rd dimension represents slices
        elif image_file_format in [".dcm", ".png", ".jpg", ".jpeg", ".tiff"]:
            # Count the number of .dcm files in the series folder
            image_files = [f for f in os.listdir(series_path) if f.endswith(tuple([image_file_format]))]
            num_slices = len(image_files)
            return num_slices
    except Exception as e:
        raise ValueError(f"Error while counting slices in series folder '{series_path}': {str(e)}")
    

def extract_metadata_from_check_file(output_directory_checkfile):
    """
    Extracts metadata values from the image data check file.

    This function reads the 'image_data_check_file.json' located in the given output directory,
    and extracts specific metadata values including the name of the series group, the number of slices per group,
    the total number of series, and the total number of slices.

    Args:
        output_directory_checkfile (str): Path to the directory containing the check file.

    Returns:
        tuple: A tuple containing:
            - series_group (str): Name of the series group
            - num_slices_group (int or None): Number of slices per group, if available.
            - total_series (int or None): Total number of image series, if available.
            - total_slices (int or None): Total number of image slices, if available.

    Raises:
        FileNotFoundError: If the check file does not exist in the specified directory.
        ValueError: If the check file is not a valid JSON file.
        KeyError: If an expected key is missing in the JSON structure.
    """ 
    check_file_path = os.path.join(output_directory_checkfile, "image_data_check_file.json")
        
    try:
        # Open and load the JSON check file
        with open(check_file_path, 'r') as f:
            check_data = json.load(f)
        
        # Extract the relevant metadata values
        series_group = check_data.get("series_group", None)
        num_slices_group = check_data.get("num_slices_group", None)
        total_series = check_data.get("global metadata", {}).get("total_series", None)
        total_slices = check_data.get("global metadata", {}).get("total_slices", None)
        
        # Return the extracted values
        return series_group, num_slices_group, total_series, total_slices
    
    except FileNotFoundError:
        raise FileNotFoundError(f"The check file at {check_file_path} was not found.")
    except json.JSONDecodeError:
        raise ValueError(f"The check file at {check_file_path} is not a valid JSON file.")
    except KeyError as e:
        raise KeyError(f"Missing expected key in the check file: {e}")
    

def extract_tag_names_and_values(ds, tag_list):
    """
    Extracts values for tags listed in tag_list.
    If the tag is a sequence, recursively extracts all nested content.
    """
    tag_names_and_values = {}

    # Convert tag_list to Tag objects
    tag_list_converted = [Tag(int(a.strip(), 16), int(b.strip(), 16))
                            for a, b in (tag.strip("()").split(",") for tag in tag_list)]

    def extract_from_dataset(dataset, parent_path=""):
        result = {}
        for elem in dataset:
            key = f"{parent_path} --> {elem.name}" if parent_path else elem.name
            if isinstance(elem.value, Sequence):
                for idx, item in enumerate(elem.value):
                    if isinstance(item, Dataset):
                        item_path = f"{key} [Item {idx + 1}]"
                        result.update(extract_from_dataset(item, parent_path=item_path))
            else:
                result[key] = elem.value
        return result

    # Only process the tags in the tag_list
    for tag in tag_list_converted:
        if tag in ds:
            element = ds[tag]
            if isinstance(element.value, Sequence):
                for idx, item in enumerate(element.value):
                    item_path = f"{element.name} [Item {idx + 1}]"
                    tag_names_and_values.update(extract_from_dataset(item, parent_path=item_path))
            else:
                tag_names_and_values[element.name] = element.value

    return tag_names_and_values


def classify_slice_orientation(image_orientation): #based on the normal vector
    row_cosines = image_orientation[:3]
    col_cosines = image_orientation[3:]
    
    # Calculate the normal vector by taking the cross product of row and column vectors
    normal_vector = [
        row_cosines[1]*col_cosines[2] - row_cosines[2]*col_cosines[1],
        row_cosines[2]*col_cosines[0] - row_cosines[0]*col_cosines[2],
        row_cosines[0]*col_cosines[1] - row_cosines[1]*col_cosines[0]
    ]
    
    # Determine the plane based on the dominant direction of the normal vector
    if abs(normal_vector[2]) > abs(normal_vector[0]) and abs(normal_vector[2]) > abs(normal_vector[1]):
        return "Axial"
    elif abs(normal_vector[0]) > abs(normal_vector[1]) and abs(normal_vector[0]) > abs(normal_vector[2]):
        return "Sagittal"
    elif abs(normal_vector[1]) > abs(normal_vector[0]) and abs(normal_vector[1]) > abs(normal_vector[2]):
        return "Coronal"
    else:
        return "Unknown"
    
    
def read_csv_file(file_path):
    """
    Reads a CSV file and detects its delimiter.
    
    Args:
        file_path (str): The path to the CSV file.
    
    Returns:
        tuple: A tuple containing the DataFrame and the detected delimiter.
    
    Raises:
        ValueError: If the file does not exist or the delimiter cannot be detected.
    """
    if not os.path.exists(file_path):
        raise ValueError(f"File '{file_path}' not found.")

    with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
        sniffer = csv.Sniffer()
        sample = csvfile.read(1024)  # Read a small sample of the file
        csvfile.seek(0)  # Reset file pointer to the beginning
        dialect = sniffer.sniff(sample)

        # Determine the delimiter
        if dialect.delimiter == '\t':
            sep = '\t'
        elif dialect.delimiter == ';':
            sep = ';'
        elif dialect.delimiter == ',':
            sep = ','
        else:
            raise ValueError("Unsupported delimiter detected in the file.")
        
        # Read the clinical data file
        data_df = pd.read_csv(file_path, sep=sep)
    
    return data_df


def get_orientation(image_orientation_patient):
    # Determines the L/R and P/A orientation of a DICOM slice based on the orientation values
    x_cosines = [float(value) for value in image_orientation_patient[:3]]
    y_cosines = [float(value) for value in image_orientation_patient[3:]]

    # Determine the dominant X orientation based on its maximum absolute value
    x_orientation = 'L' if x_cosines[np.argmax(np.abs(x_cosines))] >= 0 else 'R'

    # Determine the dominant Y orientation based on its maximum absolute value
    y_orientation = 'P' if y_cosines[np.argmax(np.abs(y_cosines))] >= 0 else 'A'

    return x_orientation, y_orientation


def load_patient_series_data(series_group, output_directory_data): 
    """
    Yields only the first matching series folder (per patient) that starts with series_group.

    Yields:
        Tuple[str, str, str, str, dict, np.ndarray]: 
            patient_id, series_id, metadata_file, tensor_file, metadata, tensor_array
    """
    
    for patient_id in os.listdir(output_directory_data):
        patient_path = os.path.join(output_directory_data, patient_id)
        if not os.path.isdir(patient_path):
            continue

        # Find the first matching series_id that starts with series_group
        series_id = next(
            (sid for sid in os.listdir(patient_path) 
                if sid.startswith(series_group) and os.path.isdir(os.path.join(patient_path, sid))),
            None
        )
        
        if not series_id:
            continue  # No matching series for this patient
            
        series_path = os.path.join(patient_path, series_id)
        
        metadata_file = None
        tensor_file = None
        for file in os.listdir(series_path):
            if file.startswith("metadata") and file.endswith(".json"):
                metadata_file = os.path.join(series_path, file)
            if file.startswith("tensor") and file.endswith(".npy"):
                tensor_file = os.path.join(series_path, file)

        if not tensor_file:
            continue

        metadata = None
        if metadata_file:
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            except Exception as e:
                print(f"Warning: Failed to read metadata for {patient_id}/{series_id}: {e}")
            
        tensor = np.load(tensor_file)

        yield patient_id, series_id, metadata_file, tensor_file, metadata, tensor


def load_patient_feature_vectors(series_group, output_directory_data, reversed_patient_id_map, reversed_series_id_map):
    """
    Loads all feature vectors (files starting with 'features' and ending with '.npy') 
    from the output directory for the specified series group.

    Returns:
        Tuple[List[np.ndarray], List[Dict[str, str]]]:
            - List of feature vectors
            - List of dictionaries with original and standardized patient/series IDs
    """
    features = []
    info = []

    for patient_id in os.listdir(output_directory_data):
        patient_path = os.path.join(output_directory_data, patient_id)
        if not os.path.isdir(patient_path):
            continue

        # Look for the matching series
        series_id = next(
            (sid for sid in os.listdir(patient_path)
                if sid.startswith(series_group) and os.path.isdir(os.path.join(patient_path, sid))),
            None
        )
        if not series_id:
            continue

        series_path = os.path.join(patient_path, series_id)

        for file in os.listdir(series_path):
            if file.startswith("features") and file.endswith(".npy"):
                feat_path = os.path.join(series_path, file)
                feat_vector = np.load(feat_path)

                std_patient_id = patient_id  # standardized
                std_series_id = series_id    # standardized
                
                orig_patient_id = reversed_patient_id_map.get(patient_id, patient_id) # original
                orig_series_id = reversed_series_id_map.get(series_id, series_id) # original

                features.append(feat_vector)
                info.append({
                    "orig_patient_id": orig_patient_id,
                    "orig_series_id": orig_series_id,
                    "std_patient_id": std_patient_id,
                    "std_series_id": std_series_id
                })

    return features, info


def load_patient_mappings(local_config, input_dir):
    """
    Loads patient ID mappings from patients.csv.
    Maps internal patient folder names to original patient names.
    """
    # Get filename and full path
    patients_list_filename = local_config["Local config"]["patients list"]
    patients_list_path = os.path.join(input_dir, patients_list_filename)
    # Read patients_list_filename file
    patients_list_df = read_csv_file(patients_list_path)
    # Create a mapping only for patients with image data (second column == 1)
    patient_id_map  = patients_list_df[patients_list_df.iloc[:, 2] == 1].set_index(patients_list_df.columns[0])[patients_list_df.columns[4]].to_dict()

    return patient_id_map


def load_series_mappings(input_dir, series_group):
    """
    Loads series ID mappings from series_mapping.csv.
    Maps internal series folder names to original series names.
    """    
    input_dir = os.path.normpath(input_dir)

    # Load the series mapping CSV file
    series_mapping_filename = "series_mapping.csv"  # Adjust this path if necessary
    series_mapping_path = os.path.join(input_dir, series_mapping_filename)

    if not os.path.isfile(series_mapping_path):
        raise FileNotFoundError(
            f"series_mapping.csv not found at: {series_mapping_path}"
        )
    
    series_mapping_df = pd.read_csv(series_mapping_path)

    # Build full mapping: {"patient X\\serie Y": "series001_ct_XXXXX", ...}
    full_series_id_map = dict(zip(series_mapping_df['Original Series Name'],
                                    series_mapping_df['New Series Name']))

    # Filter to include only mappings starting with the selected series group
    filtered_series_id_map = {
        orig: new for orig, new in full_series_id_map.items()
        if new.startswith(series_group)
    }

    return filtered_series_id_map


def ct_convert_to_hu_and_clip(image, rescale_slope, rescale_intercept, image_group_config):
    """
    Converts a CT image from raw values to Hounsfield Units (HU) and applies clipping 
    based on the protocol-defined interval for the current series group.
    
    Args:
        image (np.ndarray): The raw image pixel data.
        rescale_slope (float): DICOM Rescale Slope.
        rescale_intercept (float): DICOM Rescale Intercept.
        image_group_config (dict): The 'image' section of the protocol for the given series group.
        
    Returns:
        np.ndarray: Clipped HU image.
    """
    
    # Retrieve clipping interval from protocol
    clipping_interval = image_group_config['clipping']['clipping interval']
    a_min = clipping_interval[0]
    a_max = clipping_interval[1]
    
    hu_array = image * rescale_slope + rescale_intercept
    hu_array_clipped = np.clip(hu_array, a_min=a_min, a_max=a_max)
    return hu_array_clipped


def image_wise_clipping(image, image_group_config):
    
    # Extract modality type from the protocol
    modality = image_group_config.get("type", {}).get("selected", None)
    
    image_float = image.astype(float)
    pixels = image_float.flatten()
    
    # Define clipping thresholds based on modality
    if modality in ["CR", "DX", "RX"]:
        p_min, p_max = np.percentile(pixels, (1, 99))
    elif modality == "MRI":
        p_min, p_max = np.percentile(pixels, (0.1, 99.9))
    else:
        # Raise an error if modality is not CR, DX, or MRI
        raise ValueError(f"Unsupported modality: {modality}. Only 'CR', 'DX', or 'MRI' are allowed.")
    
    # Apply clipping
    image_clipped = np.clip(image_float, a_min=p_min, a_max=p_max)
    
    return image_clipped


def mr_db_wise_clipping(image, p_knee_val):
    image_float = image.astype(float)
    # Apply clipping to the superior part of the distribution
    image_clipped = np.clip(image_float, a_min=-np.inf, a_max=p_knee_val) # no lower bound for clipping 
    return image_clipped


def intensity_scaling(image_clipped, image_group_config):
    
    # Add an extra dimension to make it compatible with TorchIO
    image_3d = image_clipped[np.newaxis, np.newaxis, :, :]
    # Create TorchIO subject with the image
    subject = tio.Subject(
        image=tio.ScalarImage(tensor=image_3d)
    )
    
    # Read out_min_max values from the protocol based on the selected scaling option
    out_min_max = image_group_config["pixel scaling"]["selected"]
    
    # Define the transformation pipeline
    transform = tio.Compose([
        tio.transforms.ToCanonical(),
        tio.transforms.Resample("image"),
        tio.transforms.RescaleIntensity(out_min_max=out_min_max) #in_min_max=[-1000, 1000]
    ])
    ## [-1000, +1000] captures the range from air to dense bone
    # Apply the transformation to the single subject
    transformed_subject = transform(subject)
    transformed_image_array = transformed_subject['image'].numpy()[0, 0, :, :]  # Convert to NumPy array
    return transformed_image_array


def standardization_mean0_std1(image_clipped):
    # Add an extra dimension to make it compatible with TorchIO
    image_3d = image_clipped[np.newaxis, np.newaxis, :, :]

    # Check if the image is all black
    if np.std(image_3d) == 0:
        return None  # Return None if the image is all black
    
    # Create TorchIO subject with the image
    subject_standardization = tio.Subject(
        image=tio.ScalarImage(tensor=image_3d)
    )
    # Define transformation pipeline
    transform_standardization = tio.Compose([
        tio.transforms.ToCanonical(),
        tio.transforms.Resample("image"),
        tio.transforms.ZNormalization()  # Standardization by mean and standard deviation
    ])
    # Apply transformation to the subject
    transformed_subject_standardization = transform_standardization(subject_standardization)
    # Visualize the transformed image
    transformed_image_array_standardization = transformed_subject_standardization['image'].numpy()[0, 0, :, :]  # Convert to NumPy array
    return transformed_image_array_standardization


def apply_clahe(image):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to an image.

    :param image: Input image as a NumPy array (grayscale).
    :return: CLAHE-enhanced image as a NumPy array.
    """
    # Ensure image is in uint8 format (required for CLAHE)
    if image.dtype != np.uint8:
        image = (image - image.min()) / (image.max() - image.min()) * 255
        image = image.astype(np.uint8)

    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_image = clahe.apply(image)

    return clahe_image


def count_series_group_slices_from_npy(series_group, output_directory_data):
    """
    Count the total number of slices across all tensors for the current series group,
    based on the loaded tensor data from self.load_patient_series_data().

    Returns:
        int: Total number of slices across all tensors.
    """
    
    total_slices = 0
    
    for _, _, _, _, _, tensor in load_patient_series_data(series_group, output_directory_data):
        if tensor.ndim == 3:  # 3D tensor: (slices, height, width)
            total_slices += tensor.shape[0]
        elif tensor.ndim == 2:  # 2D tensor: count as 1 slice
            total_slices += 1
        else:
            raise ValueError(f"Unexpected tensor shape: {tensor.shape}")
    return total_slices


def compute_ncc(fixed_array, moving_array):
        fixed_mean = fixed_array.mean()
        moving_mean = moving_array.mean()
        
        numerator = ((fixed_array - fixed_mean) * (moving_array - moving_mean)).sum()
        denominator = ((fixed_array - fixed_mean)**2).sum() * ((moving_array - moving_mean)**2).sum()
        denominator = denominator**0.5
        
        ncc_value = numerator / denominator
        return ncc_value

    
def compute_adjacent_slice_ncc_scores(image_slices):
    num_slices = len(image_slices)
    middle_index = num_slices // 2
    ncc_scores = []
    
    for i in range(middle_index, 0, -1):
        ncc_score = compute_ncc(image_slices[i], image_slices[i-1])
        ncc_scores.append((i, i-1, ncc_score))
    
    for i in range(middle_index, num_slices - 1):
        ncc_score = compute_ncc(image_slices[i], image_slices[i+1])
        ncc_scores.append((i, i+1, ncc_score))
    
    return ncc_scores


def load_patient_series_seg_data(series_group, output_directory_data):
    """
    Generator that yields segmentation data (tensor) from saved .npy and .json files.
    It only yields data from the first matching series per patient, where the series folder 
    name starts with `self.series_group_name`.

    Yields:
        Tuple[str, str, str, np.ndarray, Union[int, str, None]]: 
            patient_id, series_id, tensor_file_path, tensor_array, segment_identifier

            - segment_identifier: either an int (e.g., 1), str label (e.g., "Edema"), or None if not found
    """ 
    for patient_id in os.listdir(output_directory_data):
        patient_path = os.path.join(output_directory_data, patient_id)
        if not os.path.isdir(patient_path):
            continue

        # Find the first matching series_id that starts with series_group
        series_id = next(
            (sid for sid in os.listdir(patient_path) 
                if sid.startswith(series_group) and os.path.isdir(os.path.join(patient_path, sid))),
            None
        )
        
        if not series_id:
            continue  # No matching series for this patient
            
        series_path = os.path.join(patient_path, series_id)

        # Find the segmentation folder (e.g., "seg_ct_...") within the series folder
        segmentation_folder = next(
            (sf for sf in os.listdir(series_path) 
                if sf.startswith("seg") and os.path.isdir(os.path.join(series_path, sf))),
            None
        )

        if not segmentation_folder:
            continue  # No segmentation folder found for this series
        
        subfolder_path = os.path.join(series_path, segmentation_folder)

        npy_files = [f for f in os.listdir(subfolder_path) if f.endswith('.npy')]

        for npy_file in npy_files:
            npy_path = os.path.join(subfolder_path, npy_file)

            segment_identifier = None
            # Extract segment number from filename using regex
            match = re.search(r"segment([^.]+)", npy_file)
            if match:
                label = match.group(1)
                segment_identifier = int(label) if label.isdigit() else label

            try:
                tensor = np.load(npy_path)

                yield patient_id, series_id, npy_path, tensor, segment_identifier

            except Exception as e:
                print(f"Failed to load file for {patient_id}/{series_id}: {e}")


def count_tot_num_series_per_dataset(input_dir): #conteggio metadato globale
    """
    Count the total number of series folders in the dataset.
    This assumes that all subfolders inside each patient folder are series folders.
    """
    total_series = 0
    
    patient_folders = [f for f in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, f)) and f != "Reports"]
    
    # Iterate over patient folders
    for patient_folder in patient_folders:
        patient_path = os.path.join(input_dir, patient_folder)

        # Check for series folders within each patient folder
        series_folders = [f for f in os.listdir(patient_path) if os.path.isdir(os.path.join(patient_path, f))]

        total_series += len(series_folders)
            
    return total_series


def count_tot_num_slices_per_dataset(input_dir, protocol, local_config): #conteggio metadato globale
    """
    Count the total number of image slices across all patient series folders.
    """

    total_slices = 0    
    patient_folders = [f for f in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, f)) and f != "Reports"]  

    # Iterate over patient folders
    for patient_folder in patient_folders:
        patient_path = os.path.join(input_dir, patient_folder)

        # Check for series folders within each patient folder
        series_folders = [f for f in os.listdir(patient_path) if os.path.isdir(os.path.join(patient_path, f))]

        # Iterate over each series folder
        for series_folder in series_folders:
            series_path = os.path.join(patient_path, series_folder)

            # Extract the series name from the protocol and map it to the folder using split
            for series_key, series_value in local_config["Local config"]["radiological data"]["series_mapping"].items():
                # get the last part of the folder path
                if series_folder == os.path.normpath(series_value).split(os.sep)[-1].strip():  # Compare with the folder name #series_value.split('/')[-1].strip()
                    # Now we have matched the folder to the protocol series
                    
                    # Extract the image file format from the protocol based on series_key
                    image_format_selected = protocol[series_key]["image"]["image file format"].get("selected", "unknown").lower()

                    # Ensure the image format is a valid format
                    if image_format_selected == "unknown":
                        raise ValueError(f"Unknown image format for series: {series_key}. Please check the configuration.")

                    try:
                        num_slices = count_tot_num_slices_per_group(series_path, image_format_selected)
                        total_slices += num_slices
                    except ValueError as e:
                        print(f"Warning: Skipping series '{series_folder}' in patient '{patient_folder}' due to error: {e}")
                        continue

    return total_slices