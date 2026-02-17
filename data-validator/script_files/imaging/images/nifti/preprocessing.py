import pandas as pd
import numpy as np
import os 
import csv
from datetime import datetime
from PIL import Image
import shutil
import json
from kneed import KneeLocator
from skimage.transform import resize
import nibabel as nib
from skimage.metrics import structural_similarity as ssim
import stat
import torch
import torchvision.transforms as transforms
from torchvision.models import inception_v3, Inception_V3_Weights
from torch import nn
from scipy.spatial.distance import pdist, squareform
from utils import save_state, load_state, clear_log_file, log_error
from ...helpers import (
    read_csv_file,
    generate_check_file_image_data,
    retrieve_existing_warnings_image_data,
    extract_metadata_from_check_file,
    load_patient_series_data,
    load_patient_feature_vectors,
    load_patient_mappings,
    load_series_mappings,
    ct_convert_to_hu_and_clip,
    image_wise_clipping,
    mr_db_wise_clipping,
    intensity_scaling,
    standardization_mean0_std1,
    count_series_group_slices_from_npy,
    count_tot_num_series_per_dataset,
    count_tot_num_slices_per_dataset,
    compute_adjacent_slice_ncc_scores
)

LOG_FILENAME = "nifti_preprocessing_error.log"

class NiftiReorientPreprocessor:

    def __init__(self, protocol, local_config, mapping_file, num_image_patients, series_group_name):
        self.protocol = protocol
        self.local_config = local_config
        self.mapping_file = mapping_file
        self.num_image_patients = num_image_patients
        self.series_group_name = series_group_name
        self.input_dir = os.getenv("INPUT_DIR")
        self.host_input_dir = os.getenv("HOST_BASE_DATA_DIR")
        self.images_dir = os.path.join(self.input_dir, "IMAGES")
        self.host_images_dir = os.path.join(self.host_input_dir, "IMAGES")
        self.state_file = os.path.join(self.input_dir, "image_validation_state.json")
        self.series_progress_file = os.path.join(self.input_dir, "validation_progress_by_series.json")
        self.study_path = os.getenv("ROOT_NAME")
        self.host_study_path = os.getenv("HOST_ROOT_DIR")
        self.output_directory_report = os.path.join(self.input_dir, "Reports", f"Image Data Reports {self.series_group_name}")
        self.host_output_directory_report = os.path.join(self.host_input_dir, "Reports", f"Image Data Reports {self.series_group_name}")
        self.output_directory_checkfile = os.path.join(self.study_path, "CHECKFILE")
        self.output_directory_data = os.path.join(self.study_path, "IMAGES")
        self.host_output_directory_data = os.path.join(self.host_study_path, "IMAGES")
        os.makedirs(self.output_directory_report, exist_ok=True)
        os.makedirs(self.output_directory_checkfile, exist_ok=True)
        os.makedirs(self.output_directory_data, exist_ok=True)


    def orientation_to_string(self, affine):
        """
        Convert affine matrix orientation to a string (e.g., 'LPS').
        
        Parameters:
        affine: numpy.ndarray
            The affine matrix to be converted.

        Returns:
        string: 'LPS', 'RAS', etc.
        """
        orientation = nib.aff2axcodes(affine)
        orientation_string = ''.join(orientation)
        return orientation_string
        

    def convert_to_lps(self, data, affine):
        """
        Convert the NIfTI image and affine matrix to LPS orientation.
        
        Parameters:
        data: numpy.ndarray
            The voxel data to be reoriented.
        affine: numpy.ndarray
            The affine matrix of the image.

        Returns:
        tuple: (reoriented_data, new_affine, message)
            Reoriented voxel data, the updated affine matrix, and a message.
        """
        reoriented_occurred = False
        
        # Define the LPS orientation
        lps_orient = nib.orientations.axcodes2ornt(('L', 'P', 'S'))
        
        # Determine the current orientation
        current_orient = nib.orientations.io_orientation(affine)
        
        # Check if the orientation is already in LPS
        if self.orientation_to_string(affine) == "LPS": 
            return data, affine, reoriented_occurred

        # Compute the orientation transformation matrix needed to go from the current orientation to LPS
        reorient = nib.orientations.ornt_transform(current_orient, lps_orient)
        
        # Apply the transformation to reorient the data
        reoriented_data_lps = nib.orientations.apply_orientation(data, reorient)

        # Compute the affine transformation needed to adjust the affine matrix
        reoriented_affine_transform = nib.orientations.inv_ornt_aff(reorient, data.shape)
        
        # Calculate the new affine matrix
        new_affine = np.dot(affine, reoriented_affine_transform)

        reoriented_occurred = True
    
        return reoriented_data_lps, new_affine, reoriented_occurred
        
    
    def check_and_reorient_niftivolumes(self): #ok #il tensore salvato qui ha il numero di slice in terza posizione (240, 240, 155)
        """
        Iterates over all volume NIfTI files in the series folders, checking if they are in LPS orientation,
        reorienting them if needed, and saving:
        - tensor_*.npy for image data
        - metadata_*.json for affine and I-S orientation flag
        """
        
        local_config = self.local_config
        series_group = self.series_group_name
        
        patient_id_map = load_patient_mappings(local_config, self.input_dir)
        series_id_map = load_series_mappings(self.images_dir, series_group)
        
        reoriented_occurred = False  # Track if any volume was reoriented
        
        series_group_mapping = self.local_config["Local config"]["radiological data"]["series_mapping"]
        series_group_subfolder = series_group_mapping.get(series_group) #input name of the series
        
        # Get list of patient folders
        patient_folders = [f for f in os.listdir(self.images_dir) if os.path.isdir(os.path.join(self.images_dir, f)) and f != "Reports"]

        # Iterate through all patient folders
        for orig_patient_id in patient_folders:
            patient_folder_path = os.path.join(self.images_dir, orig_patient_id)
            if os.path.isdir(patient_folder_path):

                if orig_patient_id.isdigit(): 
                    lookup_key = int(orig_patient_id)  # "020" -> 20
                else:
                    lookup_key = orig_patient_id

                new_patient_id = patient_id_map.get(lookup_key)

                if not new_patient_id:
                    print(f"Warning: No mapping for patient ID {orig_patient_id}")
                    continue
                
                series_folder = os.path.normpath(series_group_subfolder).split(os.sep)[-1].strip()
                series_folder_path = os.path.join(patient_folder_path, series_folder)
                
                if os.path.isdir(series_folder_path):
                    full_series_key = os.path.join(orig_patient_id, series_folder)
                    # If mappings use Windows-style separators, normalize only for lookup
                    if "\\" in list(series_id_map.keys())[0]:  # Detect if mapping keys use '\'
                        full_series_key = full_series_key.replace("/", "\\")

                    new_series_name = series_id_map.get(full_series_key)

                    if new_series_name and "_" in new_series_name:
                        parts = new_series_name.split("_")
                        if len(parts) >= 3:
                            # Reorder: ["series001", "ct", "YWPXH"] → "ct_YWPXH_series001"
                            filename_base = f"{parts[1]}_{parts[2]}_{parts[0]}"
                        else:
                            filename_base = new_series_name
                    else:
                        filename_base = f"{series_folder}_{orig_patient_id}"
                    
                    # Locate NIfTI files directly in the series folder
                    nifti_files = [f for f in os.listdir(series_folder_path) if f.lower().endswith(('.nii', '.nii.gz'))]
                    
                    for nifti_file in nifti_files:
                        nifti_file_path = os.path.join(series_folder_path, nifti_file)

                        try:
                            nifti_img = nib.load(nifti_file_path)
                            nifti_data = nifti_img.get_fdata()
                            affine = nifti_img.affine
        
                            # Check the current orientation
                            orientation_string = self.orientation_to_string(affine)
                            print(f"Current orientation for {nifti_file}: {orientation_string}")
        
                            # Check the Superior-Inferior (S/I) direction flag
                            si_flag = orientation_string[2] == 'I'  # True if the third axis is 'I', indicating reorientation needed
        
                            if orientation_string != "LPS":
                               # Convert the data to LPS orientation
                               nifti_data, affine, reoriented_occurred_here = self.convert_to_lps(nifti_data, affine)
                                   
                               if reoriented_occurred_here:
                                   reoriented_occurred = True
    
                            # Prepare output paths (anonymized)
                            patient_out_path = os.path.join(self.output_directory_data, new_patient_id)
                            series_out_path = os.path.join(patient_out_path, new_series_name)
                            os.makedirs(series_out_path, exist_ok=True)
                            
                            tensor_path = os.path.join(series_out_path, f"tensor_{filename_base}.npy")
                            np.save(tensor_path, nifti_data)
    
                            # Save metadata as .json
                            metadata = {
                                "patient_folder": new_patient_id,
                                "series_folder": new_series_name,
                                "affine": affine.tolist(),
                                "reordering_flag": si_flag
                            }
                            metadata_path = os.path.join(series_out_path, f"metadata_{filename_base}.json")
                            with open(metadata_path, 'w') as meta_file:
                                json.dump(metadata, meta_file, indent=4)

                        except Exception as e:
                            print(f"Error processing {nifti_file}: {e}") #!! Path del container

        return reoriented_occurred
    

    def process_array(self): #ok (num_slices in the third dimension) 
        """
        Loads tensors and metadata, applies clipping and normalization, and overwrites them.
        Returns:
            final_processing_summary (str): Summary of the processing steps.
        """
        local_config = self.local_config
        output_directory_data = self.output_directory_data
        series_group = self.series_group_name
        
        patient_id_map = load_patient_mappings(local_config, self.input_dir)
        series_id_map = load_series_mappings(self.images_dir, series_group)

        # Reverse maps to retrieve original keys from standardized IDs
        reversed_patient_id_map = {v: k for k, v in patient_id_map.items()}
        reversed_series_id_map = {v: k for k, v in series_id_map.items()} 
        
        image_group_config = self.protocol.get(series_group, {}).get("image", {})
        
        # extract information from the protocol
        image_type = image_group_config["type"]["selected"]
        clipping_option = image_group_config["clipping"]["selected"]
        normalization_option = image_group_config["normalization"]["selected"]

        processing_summary = []

        # Track whether clipping or normalization was applied
        clipping_applied = False
        normalization_applied = False

        # List to store discarded slices info for CSV
        discarded_slices = []

        for patient_id, series_id, metadata_file, tensor_file, metadata, tensor in load_patient_series_data(series_group, output_directory_data):

            orig_patient_id = reversed_patient_id_map.get(patient_id, patient_id)
            orig_series_id = reversed_series_id_map.get(series_id, series_id)
            orig_series_id = os.path.basename(orig_series_id)
            
            updated_tensor = []
            
            # Use default values for NIfTI files
            rescale_slope = 1
            rescale_intercept = -1024
    
            # Prepare lists to store all pixels for global MRI normalization, if needed
            all_pixels = []
            p_knee_val = None
    
            # If db-wise normalization is selected for MRI
            if image_type == "MRI" and clipping_option == "db-wise (only MRI)":
                for slice_idx in range(tensor.shape[2]):
                    slice_data = np.squeeze(tensor[:, :, slice_idx]) # slices in z position
                    all_pixels.extend(slice_data.flatten())
    
                all_pixels = np.array(all_pixels).astype(float)
                percentiles = np.arange(95, 100, 0.01)
                percentile_values = np.percentile(all_pixels, percentiles)
                percentiles_desc_order = percentiles[::-1]
                percentile_values_desc_order = percentile_values[::-1]
                kl = KneeLocator(percentiles_desc_order, percentile_values_desc_order, curve="convex")
                knee_value = kl.knee
                p_knee_val = np.percentile(all_pixels, knee_value)
                print("X-axis value corresponding to the knee point:", knee_value)
    
            # Process each slice
            for slice_idx in range(tensor.shape[2]):
                image_array = np.squeeze(tensor[:, :, slice_idx])
    
                # Initialize variables to track transformations
                normalized_image = None
                clipped_image = image_array
    
                # Clipping based on image type and protocol options
                if image_type == "CT":
                    if clipping_option == "image-wise":
                        clipped_image = ct_convert_to_hu_and_clip(image_array, rescale_slope, rescale_intercept, image_group_config)
                        clipping_applied = True
                    elif clipping_option == "db-wise (only MRI)":
                        raise ValueError("Selected clipping method is not valid for CT images!")
                    
                elif image_type == "MRI":
                    if clipping_option == "image-wise":
                        #print(f"Before image-wise clipping (slice {slice_idx}): Min: {np.min(clipped_image)}, Max: {np.max(clipped_image)}")
                        clipped_image = image_wise_clipping(image_array, image_group_config)
                        clipping_applied = True
                        #print(f"After image-wise clipping (slice {slice_idx}): Min: {np.min(clipped_image)}, Max: {np.max(clipped_image)}")
                    elif clipping_option == "db-wise (only MRI)" and p_knee_val is not None:
                        clipped_image = mr_db_wise_clipping(image_array, p_knee_val)
                        clipping_applied = True
    
                # Normalization based on protocol options
                if normalization_option == "intensity_scaling":
                    normalized_image = intensity_scaling(clipped_image, image_group_config)
                    normalization_applied = True
                elif normalization_option == "standardization_mean0_std1":
                    normalized_image = standardization_mean0_std1(clipped_image)
                    normalization_applied = True
                elif normalization_option == "none":
                    normalized_image = clipped_image  # No normalization, use the clipped image
    
                # Convert the processed image to float64 for consistency
                if normalized_image is not None:
                    normalized_image = normalized_image.astype(np.float64)
                
                # Skip slices that are entirely black (zero standard deviation)
                if normalized_image is None or (np.std(normalized_image) == 0):
                    print(f"Skipping all-black image at slice index {slice_idx + 1} for patient {patient_id}, series {series_id}")
                    # Record discarded slice information
                    discarded_slices.append([orig_patient_id, orig_series_id, slice_idx + 1])
                    continue

                # Append the processed image to the list of slices for this patient
                updated_tensor.append(normalized_image.astype(np.float32))
                
            if not updated_tensor:
                print(f"All slices were discarded for patient {patient_id}, series {series_id}. Skipping save.")
                continue

            # Save updated tensor
            tensor_3d = np.stack(updated_tensor, axis=0)
            tensor_3d = np.transpose(tensor_3d, (0, 2, 1))
            np.save(tensor_file, tensor_3d)
    
            print(f"Processed patient: {patient_id}, series: {series_id}")
    
        # Create a summary of the protocol options applied
        if clipping_applied:
            processing_summary.append(f"- Clipping applied according to protocol: {clipping_option}.")
        else:
            processing_summary.append("- Clipping not required by protocol.")
        
        if normalization_applied:
            processing_summary.append(f"- Normalization applied according to protocol: {normalization_option}.")
        else:
            processing_summary.append("- Normalization not required by protocol.")
            
        if discarded_slices:
            #print("discarded_slices:", discarded_slices)
            csv_path = os.path.join(self.images_dir, "discarded_slices.csv")
            
            # Create a set to store existing discarded slices
            existing_rows = set()
        
            # If the CSV already exists, read its existing rows
            if os.path.isfile(csv_path):
                with open(csv_path, mode='r', newline='') as csv_file:
                    reader = csv.reader(csv_file)
                    next(reader, None)  # Skip header
                    for row in reader:
                        existing_rows.add(tuple(row))  # Store as tuple for fast lookup
        
            # Prepare only the new discarded slices (avoid duplicates)
            new_rows = []
            for row in discarded_slices:
                if (row[0], row[1], str(row[2])) not in existing_rows:    
                    new_rows.append(row)
        
            if new_rows:
                with open(csv_path, mode='a', newline='') as csv_file:
                    writer = csv.writer(csv_file)
                    # If file didn't exist before, write header
                    if not existing_rows:
                        writer.writerow(["Patient Name", "Series Folder Name", "Discarded Slice Index"])
                    writer.writerows(new_rows)
                print(f"{len(new_rows)} new discarded slices written to {csv_path}") #!! Path del container
            else:
                print("No new discarded slices to add; CSV remains unchanged.")
        else:
            print("No discarded slices; CSV file will not be created/updated.")
            
        # Add number of discarded slices to the processing summary
        num_discarded_slices = len(discarded_slices)
        processing_summary.append(f"- Number of discarded slices (all black): {num_discarded_slices}.")

        # If any slices were discarded, add a message about the CSV file
        if num_discarded_slices > 0:
            processing_summary.append(f"- discarded_slices.csv file has been created/updated in the directory: {self.host_images_dir}") #!! Path del container, deve essere quello locale.ok

        # Join summary statements and return them along with processed slices
        final_processing_summary = "\n".join(processing_summary)
        
        return final_processing_summary
    

    def resize_array(self): #ok
        """
        Loads .npy tensors, resizes image slices to match the target size in the protocol,
        and overwrites the original .npy tensor file if resizing is needed.
        """
        output_directory_data = self.output_directory_data
        series_group = self.series_group_name
        image_group_config = self.protocol.get(series_group, {}).get("image", {}) 
        
        # Extract size from the protocol
        target_size = tuple(image_group_config["size"])

        any_resized = False
        final_message_resize = ""

        for patient_id, series_id, _, tensor_file, _, tensor in load_patient_series_data(series_group, output_directory_data):
            resized_tensor = []
            was_resized = False  # Track for this specific tensor

            for image_array in tensor:
                image_array = np.squeeze(image_array)  # Remove any singleton dimensions
                if image_array.shape != target_size:
                    resized_array = resize(image_array, target_size, anti_aliasing=True)
                    was_resized = True
                    resized_tensor.append(resized_array.astype(np.float64))
                else:
                    resized_tensor.append(image_array.astype(np.float64))
                    
            if was_resized: 
                np.save(tensor_file, np.stack(resized_tensor, axis=0))
                print(f"Resized and saved tensor for: {patient_id}/{series_id}")
                any_resized = True
            else:
                print(f"No resize needed for: {patient_id}/{series_id}")
              
        # Construct the final message based on whether any slices were resized
        if any_resized:    
            final_message_resize += "Resizing completed."
        else:
            final_message_resize += "Resizing not needed: slices are already of the correct size."
        
        return final_message_resize


    def save_labels(self):  
        """
        Save patient TXT labels and 3D slice CSV labels into series folders,
        applying reordering and discarding slices if needed. Returns a summary
        message.
        """
        local_config = self.local_config
        output_directory_data = self.output_directory_data

        series_group = self.series_group_name
        image_group_config = self.protocol.get(series_group, {}).get("image", {}) 
        image_type = image_group_config["type"]["selected"]

        patient_id_map = load_patient_mappings(local_config, self.input_dir)
        series_mapping = load_series_mappings(self.images_dir, series_group) 

        # normalize mapping keys 
        series_mapping = {os.path.normpath(k): v for k, v in series_mapping.items()} 

        normalized_series_mapping = {}

        for k, v in series_mapping.items():
            patient_part, series_part = os.path.normpath(k).split(os.sep)

            # normalize patient id numerically
            if patient_part.isdigit():
                patient_part = str(int(patient_part))

            new_key = os.path.join(patient_part, series_part)
            normalized_series_mapping[new_key] = v

        series_mapping = normalized_series_mapping

        # Reverse maps for original IDs
        reversed_patient_id_map = {v: k for k, v in patient_id_map.items()}
        reversed_series_id_map = {v: k for k, v in series_mapping.items()}

        # Check if discarded_slices.csv exists
        discarded_slices_path = os.path.join(self.images_dir, "discarded_slices.csv")
        discarded_slices_df = None
        if os.path.exists(discarded_slices_path):
            discarded_slices_df = pd.read_csv(discarded_slices_path, header=None, names=["PatientID", "SeriesID", "Index"])
           
        final_message_npy_and_labels = ""
        saved_patient_label_count = 0
        saved_slice_label_count = 0

        # Check if slice labels are specified in the protocol
        slice_labels = image_group_config.get("slice_labels", None)
        
        # Load patient series data and iterate over them
        for patient_id, series_id, _, _, metadata_json, tensor in load_patient_series_data(series_group, output_directory_data):
            # Map back to original IDs (folder names)
            orig_patient_id = reversed_patient_id_map.get(patient_id)
            if orig_patient_id is None:
                    raise KeyError(
                        f"Patient ID {patient_id} not found in reversed patient ID mapping. "
                        "Processing stopped."
                    )

            orig_series_id = reversed_series_id_map.get(series_id)
            if orig_series_id is None:
                raise KeyError(
                    f"Series ID {series_id} not found in reversed series mapping."
                )
            orig_series_id = os.path.basename(orig_series_id)
  
            # Construct the full key with both patient ID and series name (like "ID14\\1.000000-3DAXT1postcontrast-17667")
            full_series_name = os.path.normpath(
                os.path.join(str(orig_patient_id), str(orig_series_id))
            )

            if full_series_name not in series_mapping:
                raise KeyError(
                    f"Series mapping missing for '{full_series_name}'."
                )

            # Extract reordering_flag from metadata_json (assuming it's a dict)
            reordering_flag = metadata_json.get("reordering_flag", None) if image_type in ["MRI", "CT"] else None

            new_patient_id = patient_id_map[orig_patient_id]
            patient_folder_path = os.path.join(self.output_directory_data, new_patient_id)

            # Use full_series_name to get the correct new series folder
            new_series_folder_name = series_mapping[full_series_name]
            new_series_folder_path = os.path.join(patient_folder_path, new_series_folder_name)

            # Ensure the new series folder exists
            if not os.path.exists(new_series_folder_path):
                raise FileNotFoundError(f"Series folder {new_series_folder_name} does not exist for patient {new_patient_id}, stopping process.")

            # Save the 3D tensor
            modality = new_series_folder_name.split('_')[1]
            five_letters = new_series_folder_name.split('_')[-1]
            series_number = new_series_folder_name.split('_')[0][-3:]

            # Check if patient_label is defined in the protocol
            if "patient_label" in image_group_config:
                # Read original patient_label.txt from the old series folder
                original_series_folder = os.path.join(self.images_dir, orig_patient_id, orig_series_id)
                if not os.path.exists(original_series_folder):
                    raise FileNotFoundError(
                        f"Original series folder does not exist: {original_series_folder}"
                    )
                # Find the unique CSV file in the series folder
                pts_label_txt_file = next((f for f in os.listdir(original_series_folder) if f.endswith(".txt")), None)

                if pts_label_txt_file is None:
                    raise FileNotFoundError(
                        f"No TXT label file found in {original_series_folder} "
                        "but patient_label is required by protocol."
                    )
                
                original_pts_label_path = os.path.join(original_series_folder, pts_label_txt_file)

                if os.path.exists(original_pts_label_path):
                    # Construct new .txt filename
                    new_txt_filename = f"patient_label_{modality}_{five_letters}_series{series_number}.txt"
                    new_txt_path = os.path.join(new_series_folder_path, new_txt_filename)

                    # Copy the content from the original .txt file to the new one
                    with open(original_pts_label_path, "r", encoding="utf-8") as old_file:
                        content = old_file.read()

                    with open(new_txt_path, "w", encoding="utf-8") as new_file:
                        new_file.write(content)
                    
                    saved_patient_label_count += 1

                    print(f"Saved patient label file as {new_txt_filename} in {new_series_folder_path}")

            # If slice_labels are defined, process and save labels
            if slice_labels: # only in the 3D scenario
                # Read original labels.csv from the old series folder
                original_series_folder = os.path.join(self.images_dir, orig_patient_id, orig_series_id)
                # Find the unique CSV file in the series folder
                label_csv_file = next((f for f in os.listdir(original_series_folder) if f.endswith(".csv")), None)
                original_labels_path = os.path.join(original_series_folder, label_csv_file)
                original_labels_df = read_csv_file(original_labels_path) #!
                # access the second column (labels column)
                original_labels = original_labels_df.iloc[:, 1].tolist()  

                # Apply reordering if necessary (reverse labels order)
                if reordering_flag:
                    original_labels.reverse()

                # Discard slices based on discarded_slices.csv (only filter labels)
                valid_labels = original_labels  # Start with all labels
                valid_indices = list(range(len(original_labels)))
    
                if discarded_slices_df is not None:
                    series_discard_indices = discarded_slices_df[
                        (discarded_slices_df["PatientID"] == orig_patient_id) &
                        (discarded_slices_df["SeriesID"] == orig_series_id)
                    ]["Index"].tolist()
                    
                    # Adjust the discarded indices for 0-based indexing
                    series_discard_indices = [int(i) - 1 for i in series_discard_indices]
                    
                    valid_indices = [i for i in valid_indices if i not in series_discard_indices]
                    valid_labels = [original_labels[i] for i in valid_indices]

                if tensor.shape[0] == 0:
                    print(f"All slices discarded for series {orig_series_id}, skipping...")
                    continue
                    
                # Save labels.csv
                labels_filename = f"labels_{modality}_{five_letters}_series{series_number}.csv"
                labels_filepath = os.path.join(new_series_folder_path, labels_filename)
                labels_data = [
                    [f"slice{i+1:03d}_{modality}_{five_letters}_series{series_number}", label]
                    for i, label in enumerate(valid_labels)
                ]
                labels_df = pd.DataFrame(labels_data, columns=["Slice Name", "Label"])
                labels_df.to_csv(labels_filepath, index=False)
                saved_slice_label_count += 1
                print(f"Saved labels CSV {labels_filename} in {new_series_folder_path}") 

        final_message_npy_and_labels += f"Tensors saved in .npy format into series folders within patient folders in {self.host_output_directory_data}." #!!

        if "patient_label" in image_group_config:
            if saved_patient_label_count > 0:
                final_message_npy_and_labels += "\n- Patient labels were saved as TXT files in the respective series folders."
            else:
                final_message_npy_and_labels += (
                    "\n- WARNING: Patient labels were expected by the protocol, "
                    "but no TXT label files were saved."
                )
            
        # Include a message about labels processing if slice_labels are defined
        if slice_labels:
            if saved_slice_label_count > 0:
                final_message_npy_and_labels += "\n- Slice labels were processed and saved as CSV files in the respective series folders."
            else:
                final_message_npy_and_labels += "\n- WARNING: Slice labels were expected by the protocol, but no CSV label files were saved."
            
        return final_message_npy_and_labels
    

    def convert_npy_to_image(self):
        """Convert npy files to images."""

        series_group = self.series_group_name
        image_group_config = self.protocol.get(series_group, {}).get("image", {}) #original self.protocol["image"]
        
        # Extract image format from the protocol
        output_format = image_group_config["output format"]["selected"]
        normalization_option = image_group_config["normalization"]["selected"]
        scaling_option = image_group_config["pixel scaling"]["selected"]
        
        # Check if scaling is needed based on the normalization and scaling options
        should_scale = not (normalization_option == "intensity_scaling" and scaling_option == [0, 255])

        message_from_npy_to_image = ""
        
        # Iterate through patient folders
        for patient_folder in os.listdir(self.output_directory_data):
            patient_folder_path = os.path.join(self.output_directory_data, patient_folder)
            
            if os.path.isdir(patient_folder_path):
                # Find the series folder that starts with series_group
                series_folder = next(
                    (folder for folder in os.listdir(patient_folder_path)
                     if folder.startswith(series_group) and os.path.isdir(os.path.join(patient_folder_path, folder))),
                    None
                )
                    
                if series_folder:
                    series_folder_path = os.path.join(patient_folder_path, series_folder)
                    
                    # Extract modality, five random letters, and series number from the series folder name
                    try:
                        modality = series_folder.split('_')[1]          # Extracts 'mri' or 'ct'
                        five_letters = series_folder.split('_')[-1]    # Extracts 'HEEQN' or other letters
                        series_number = series_folder.split('_')[0][-3:]  # Extracts '001'
                    except IndexError:
                        raise ValueError(f"Series folder {series_folder} does not follow the expected naming convention.")

                    # Iterate through npy files in the series folder
                    for npy_file in os.listdir(series_folder_path):
                        if npy_file.startswith("tensor") and npy_file.endswith(".npy"):
                            npy_file_path = os.path.join(series_folder_path, npy_file)
                            
                            # Load the 3D numpy array from the .npy file
                            array_3d = np.load(npy_file_path)
                            
                            # Create the image folder inside the series folder
                            image_folder_name = f"image_{modality}_{five_letters}_series{series_number}"
                            image_folder_path = os.path.join(series_folder_path, image_folder_name)
                            os.makedirs(image_folder_path, exist_ok=True)
                            
                            # Iterate through each slice in the 3D array
                            for i, slice_2d in enumerate(array_3d):
                                # Scale the array values to the range between 0 and 255 if needed
                                if should_scale:
                                    scaled_array = ((slice_2d - slice_2d.min()) / (slice_2d.max() - slice_2d.min())) * 255
                                    final_image_array = np.uint8(scaled_array)
                                else:
                                    final_image_array = np.uint8(slice_2d)
                                
                                # Convert numpy array to image
                                final_image = Image.fromarray(final_image_array)
                                
                                # Determine the output file name and path
                                output_file_name = f"image{i+1:03d}_{modality}_{five_letters}_series{series_number}{output_format}"
                                output_file_path = os.path.join(image_folder_path, output_file_name)
                                
                                # Save the image in the desired format
                                final_image.save(output_file_path)
                                print(f"Saved {output_file_name} in {image_folder_path}") #!! Path del container, si può non cambiare
                                    
        message_from_npy_to_image += f"Images in {output_format} format have been generated from npy tensors and saved into series folders within patient folders in {self.host_output_directory_data}." #!! Path del container, deve essere quello locale.ok               
        return message_from_npy_to_image
    
    
    # Helper function for intra-replicates check
    def check_intraseries_replicates(self, output_directory_data, slice_threshold_std=0.005):
        """
        Flags series where all adjacent slices are nearly identical by checking
        the standard deviation of SSIM values across the volume.
    
        Args:
            slice_threshold_std (float): Threshold below which std deviation of SSIM is suspiciously low.
    
        Returns:
            dict: {
                "patient_id/series_id": {
                    "std_dev": float,
                    "similarities": [float, float, ...]
                },
                ...
            }
        """
        flagged_series = {}
    
        for patient_id, series_id, _, _, _, tensor in load_patient_series_data(self.series_group_name, output_directory_data):
            if tensor.shape[0] < 2:
                continue  # Cannot compare if only one slice
    
            ssim_values = []
            for i in range(len(tensor) - 1):
                slice1 = tensor[i].squeeze()
                slice2 = tensor[i + 1].squeeze()
                data_range = max(slice1.max() - slice1.min(), slice2.max() - slice2.min())
                if data_range == 0:
                    continue  # Skip constant slices
    
                sim_index, _ = ssim(slice1, slice2, full=True, data_range=data_range)
                ssim_values.append(sim_index)
    
            if len(ssim_values) < 2:
                continue  # Not enough data to compute std
    
            std_ssim = np.std(ssim_values)
    
            if std_ssim < slice_threshold_std:
                label = f"{patient_id}/{series_id}"
                flagged_series[label] = {
                    "std_dev": std_ssim,
                    "similarities": ssim_values
                }
    
        return flagged_series
    
    
    def check_interseries_replicates_by_features(self, series_group, output_directory_data, reversed_patient_id_map, reversed_series_id_map, threshold=1e-3):
        """
        Identifies inter-series replicates using Euclidean distance between feature vectors.
    
        Args:
            series_group (str): Name of the image group.
            output_directory_data (str): Path to the processed data directory.
            reversed_patient_id_map (dict): Map of standardized -> original patient IDs.
            reversed_series_id_map (dict): Map of standardized -> original series IDs.
            threshold (float): Distance threshold below which two feature vectors are considered duplicates.
    
        Returns:
            Set[str]: Standardized patient IDs whose series are flagged as inter-replicates.
        """
        grouped_series = []
        series_pairs_checked = set()
        
        features, info = load_patient_feature_vectors(
            series_group=series_group,
            output_directory_data=output_directory_data,
            reversed_patient_id_map=reversed_patient_id_map,
            reversed_series_id_map=reversed_series_id_map
        )
    
        if len(features) < 2:
            return set()
    
        features = np.array(features)
        distances = squareform(pdist(features, metric='euclidean'))
    
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                pair = tuple(sorted((i, j)))
                if pair in series_pairs_checked:
                    continue
                series_pairs_checked.add(pair)
        
                if distances[i, j] < threshold:
                    label1 = f"{info[i]['std_patient_id']}/{info[i]['std_series_id']}"
                    label2 = f"{info[j]['std_patient_id']}/{info[j]['std_series_id']}"
        
                    # Try to find existing group and merge
                    group_found = False
                    for group in grouped_series:
                        if label1 in group or label2 in group:
                            group.add(label1)
                            group.add(label2)
                            group_found = True
                            break
        
                    if not group_found:
                        grouped_series.append(set([label1, label2]))

        inter_replicate_series = set()
        for group in grouped_series:
            sorted_group = sorted(group)
            to_remove = sorted_group[1:]  # keep the first one
            inter_replicate_series.update(to_remove)

        return grouped_series, inter_replicate_series
    

    def check_image_replicates_after_processing(self): 
        """
        Detects inter-series and intra-series replicates based on feature vectors and SSIM variation, respectively.
        Removes affected patient folders and generates a report.
    
        Returns:
            tuple: (
                str: Warning message summary,
                int: Number of slices in series group after processing,
                int: Total number of series,
                int: Total number of slices,
                int: Total number of removed patient folders
            )
        """
        
        def on_rm_error(func, path):
            os.chmod(path, stat.S_IWRITE)  # remove read-only restriction
            func(path)  # retry the operation
            
        local_config = self.local_config
        series_group = self.series_group_name
        
        patient_id_map = load_patient_mappings(local_config, self.input_dir)
        series_id_map = load_series_mappings(self.images_dir, series_group)

        # Reverse maps to retrieve original keys from standardized IDs
        reversed_patient_id_map = {v: k for k, v in patient_id_map.items()}
        reversed_series_id_map = {v: k for k, v in series_id_map.items()}
        
        phase_number = 36  
        phase_data = self.mapping_file.get("check_phase", {}).get(str(phase_number), {})
        warning_descriptions = phase_data.get("warnings", {})
        phase_name = phase_data.get("name", f"Phase {phase_number}")  
        
        # Format check and report names dynamically
        check_name = phase_name.lower().replace(" ", "_") 
        report_name = f"2.036.{check_name}_report"

        output_directory_data = self.output_directory_data
        series_group = self.series_group_name
        image_group_config = self.protocol.get(series_group, {}).get("image", {}) 

        image_type = image_group_config["type"]["selected"]
        
        # Generate current date and time
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        warning_counts = {"W3601": None,
                        "W3602": None}
        
        inter_replicate_patients = set()
        inter_replicate_patients_output_names = set()
        intra_replicate_patients = set()
        
        output_directory_checkfile = self.output_directory_checkfile
        _, _, total_series, total_slices = extract_metadata_from_check_file(output_directory_checkfile)

        check_file_path = os.path.join(self.output_directory_checkfile, "image_data_check_file.json")
        # Retrieve existing warnings
        existing_warnings = retrieve_existing_warnings_image_data(check_file_path)

        # === INTER-SERIES REPLICATE CHECK VIA FEATURES ===
        grouped_series, inter_replicate_series = self.check_interseries_replicates_by_features(
            series_group=series_group,
            output_directory_data=output_directory_data,
            reversed_patient_id_map=reversed_patient_id_map,
            reversed_series_id_map=reversed_series_id_map,
            threshold=1e-3  # Adjust as needed
        )

        for label in inter_replicate_series:
            std_pid = label.split("/")[0]
            orig_pid = reversed_patient_id_map.get(std_pid, std_pid)

            inter_replicate_patients.add(orig_pid)              # original name for reporting
            inter_replicate_patients_output_names.add(std_pid)  # standardized name for deletion

        # === INTRA-SERIES REPLICATES CHECK (only for 3D) === 
        flagged_series_std = {}
        if image_type not in ["CR", "DX", "RX"]:
            flagged_series_std = self.check_intraseries_replicates(output_directory_data)
            for label in flagged_series_std:
                std_pid = label.split("/")[0]  # already standardized
                intra_replicate_patients.add(std_pid) 
  
        # Remove affected patient folders
        removed_patient_folders = inter_replicate_patients_output_names.union(intra_replicate_patients)
        for patient_id in removed_patient_folders:
            folder_path = os.path.join(output_directory_data, patient_id)
            if os.path.exists(folder_path):
                shutil.rmtree(folder_path, onerror=on_rm_error)
                
            # Delete from input_dir (original name)
            orig_patient_id = reversed_patient_id_map.get(patient_id, patient_id)
            orig_patient_id_str = str(orig_patient_id)

            # Try direct match first
            input_folder_path = os.path.join(self.images_dir, orig_patient_id_str)

            if not os.path.exists(input_folder_path):
                # Try to find matching folder ignoring leading zeros
                for folder_name in os.listdir(self.images_dir):
                    if folder_name.isdigit() and int(folder_name) == int(orig_patient_id):
                        input_folder_path = os.path.join(self.images_dir, folder_name)
                        break

            if os.path.exists(input_folder_path):
                shutil.rmtree(input_folder_path, onerror=on_rm_error)

        warning_counts["W3601"] = len(inter_replicate_patients) if inter_replicate_patients else None
        warning_counts["W3602"] = len(intra_replicate_patients) if intra_replicate_patients else None

        # recount slices in the series group after potential deletions
        num_slices_group_new = count_series_group_slices_from_npy(series_group, output_directory_data)
        # recompute global metadata only if deletions occurred
        if removed_patient_folders:
            total_series = count_tot_num_series_per_dataset(self.images_dir) 
            total_slices = count_tot_num_slices_per_dataset(self.images_dir, self.protocol, local_config) 
            
        num_patients_with_image_data = len([
            f for f in os.listdir(self.images_dir)
            if os.path.isdir(os.path.join(self.images_dir, f)) and f != "Reports"
        ])
        
        # Generate the check file
        generate_check_file_image_data(
            check_name=phase_name,
            phase_number=phase_number,
            series_group_name=series_group,
            num_slices_group=num_slices_group_new,
            num_patients_with_image_data=num_patients_with_image_data,
            num_series=total_series,
            num_tot_slices=total_slices,
            timestamp=current_datetime,
            error_counts={},
            warning_counts=warning_counts,
            error_descriptions={},
            warning_descriptions=warning_descriptions,
            output_dir=self.output_directory_checkfile,
            existing_warnings=existing_warnings
        )
    
        if grouped_series or flagged_series_std:
            os.makedirs(self.output_directory_report, exist_ok=True)
            report_path = os.path.join(self.output_directory_report, f"{report_name}.txt")
            host_report_path = os.path.join(self.host_output_directory_report, f"{report_name}.txt")
            
            with open(report_path, "w") as report_file:
                
                # Add the report generation date and time to the first line
                report_file.write(f"Report generated on: {current_datetime}\n\n")
                report_file.write(f"{phase_name} report:\n\n")

                if grouped_series:
                    report_file.write(f"Warning W3601: {warning_descriptions.get('W3601', None)}\n")
                    report_file.write("- Details: The following series were flagged as highly similar based on feature vectors:\n")

                    for i, group in enumerate(grouped_series):
                        group_list_std = sorted(list(group))
                        group_list_orig = []
                        
                        for label in group_list_std:
                            std_pid, std_sid = label.split("/")
                            orig_pid = reversed_patient_id_map.get(std_pid, std_pid)
                            orig_sid = os.path.basename(reversed_series_id_map.get(std_sid, std_sid))
                            group_list_orig.append(f"{orig_pid}/{orig_sid}")
                        
                        group_report = " and ".join(group_list_orig)
                        report_file.write(f"  - {group_report}.\n")

                    if inter_replicate_patients:
                        report_file.write("\n  Deleted patient folders (inter-series replicates):\n")
                        for pid in sorted(inter_replicate_patients):
                            orig_pid = reversed_patient_id_map.get(pid, pid)
                            report_file.write(f"  - {orig_pid}\n")

                    report_file.write(f"\nTotal inter-series replicates: {len(inter_replicate_patients)}\n\n\n")

                if flagged_series_std:
                    report_file.write(f"Warning W3602: {warning_descriptions.get('W3602', None)}\n")
                    report_file.write("- Details: The following series have suspiciously low slice variation (std of SSIM < 0.005):\n")
                    for label, info in flagged_series_std.items():
                        pid, sid = label.split("/")
                        orig_pid = reversed_patient_id_map.get(pid, pid)
                        orig_sid = os.path.basename(reversed_series_id_map.get(sid, sid))
                        report_file.write(f"  - {orig_pid}/{orig_sid}: std_dev = {info['std_dev']:.6f}, min = {min(info['similarities']):.4f}, max = {max(info['similarities']):.4f}\n")

                    if intra_replicate_patients:
                        report_file.write("\n  Deleted patient folders (intra-series replicates):\n")
                        for pid in sorted(intra_replicate_patients):
                            orig_pid = reversed_patient_id_map.get(pid, pid)
                            report_file.write(f"  - {orig_pid}\n")

                    report_file.write(f"\nTotal series with low slice variation: {len(intra_replicate_patients)}\n\n")
   
            print(f"Highly similar slices detected. "
                             f"See the detailed report at: {host_report_path}") #!! Path del container, metto quello locale.ok
            
        if len(inter_replicate_patients) > 0 and len(intra_replicate_patients) > 0:
            replicate_slice_message = (
                f"Warning W3601: {len(inter_replicate_patients)} patient folder(s) removed due to inter-series replicates; "
                f"Warning W3602: {len(intra_replicate_patients)} patient folder(s) removed due to intra-series replicates."
            )
        elif len(inter_replicate_patients) > 0:
            replicate_slice_message = (
                f"Warning W3601: {len(inter_replicate_patients)} patient folder(s) removed due to inter-series replicates."
            )
        elif len(intra_replicate_patients) > 0:
            replicate_slice_message = (
                f"Warning W3602: {len(intra_replicate_patients)} patient folder(s) removed due to intra-series replicates."
            )
        else:
            replicate_slice_message = "No highly similar inter-series middle slices or intra-series slice redundancies detected."
            
        warnings_replicates = len(removed_patient_folders)

        num_patients_new = num_patients_with_image_data

        return replicate_slice_message, num_slices_group_new, total_series, total_slices, warnings_replicates, num_patients_new
    

    def analyze_series_alignment(self): #3D only
        
        phase_number = 37  
        phase_data = self.mapping_file.get("check_phase", {}).get(str(phase_number), {})
        warning_descriptions = phase_data.get("warnings", {})
        phase_name = phase_data.get("name", f"Phase {phase_number}")  
        
        # Format check and report names dynamically
        check_name = phase_name.lower().replace(" ", "_") 
        report_name = f"2.037.{check_name}_report"

        series_group = self.series_group_name
        image_group_config = self.protocol.get(series_group, {}).get("image", {}) 

        output_directory_checkfile  = self.output_directory_checkfile
        _, num_slices_group, total_series, total_slices = extract_metadata_from_check_file(output_directory_checkfile)

        # Generate report content
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")      
        total_misaligned_slices = 0
        report_details = []
        message_series_alignment = ""
        warning_counts = {"W3701": None}

        check_file_path = os.path.join(self.output_directory_checkfile, "image_data_check_file.json")
        existing_warnings = retrieve_existing_warnings_image_data(check_file_path)
        
        # Iterate through patient folders
        for patient_folder in os.listdir(self.output_directory_data):
            patient_folder_path = os.path.join(self.output_directory_data, patient_folder)
            if os.path.isdir(patient_folder_path):
                # Find the series folder that starts with series_group
                series_folder = next(
                    (folder for folder in os.listdir(patient_folder_path)
                        if folder.startswith(series_group) and os.path.isdir(os.path.join(patient_folder_path, folder))),
                    None
                )
                if series_folder:
                    series_folder_path = os.path.join(patient_folder_path, series_folder)
                    series_name = series_folder
                        
                    # Extract modality, five random letters, and series number from the series folder name
                    try:
                        modality = series_folder.split('_')[1]           # Extracts 'mri' or 'ct'
                        five_letters = series_folder.split('_')[-1]     # Extracts 'HEEQN' or other letters
                        series_number = series_folder.split('_')[0][-3:]  # Extracts '001'
                    except IndexError:
                        raise ValueError(f"Series folder {series_folder} does not follow the expected naming convention.")
                        
                     # Iterate through npy arrays in the series folder
                    image_series = []
                
                    for npy_file in os.listdir(series_folder_path):
                        if npy_file.startswith("tensor") and npy_file.endswith(".npy"):
                            npy_file_path = os.path.join(series_folder_path, npy_file)
                            
                            # Load the 3D numpy array from the .npy file (shape: z, y, x)
                            loaded_array = np.load(npy_file_path)
                            num_slices = loaded_array.shape[0]

                            # Iterate over each slice (z-axis) and append it to the image_series list
                            for z in range(num_slices):
                                slice_2d = loaded_array[z, :, :]  # Extract the 2D slice from the 3D array
                                slice_name = f"slice{z+1:03d}_{modality}_{five_letters}_series{series_number}"
                                image_series.append((slice_name, slice_2d))
                    

                    similarity_scores = compute_adjacent_slice_ncc_scores([array for _, array in image_series])
                    image_type = image_group_config["type"]["selected"]
            
                    # Set NCC threshold based on image type
                    if image_type == "CT":
                        ncc_threshold = 0.9
                    elif image_type == "MRI":
                        ncc_threshold = 0.8
                    
                    num_scores = len(similarity_scores)
                
                    if num_scores % 2 == 1:  # If the number of scores is odd
                        middle_index = num_scores // 2 + 1
                    else:  # If the number of scores is even
                        middle_index = num_scores // 2
                
                    first_half_reversed = similarity_scores[:middle_index][::-1]
                    second_half = similarity_scores[middle_index:]
                
                    modified_scores = first_half_reversed + second_half
                    #print(modified_scores)
                    x_values = []
                    ncc_scores = []
                    misaligned_slices = []
                
                    for score in modified_scores:
                        adjusted_x = score[1] - middle_index
                        x_values.append(adjusted_x)
                        ncc_scores.append(score[2])
                        if score[2] < ncc_threshold:
                            misaligned_slices.append(score)
                            #print(misaligned_slices)
                            
                    if misaligned_slices:
                        # Calculate mean, std deviation, and median only if ncc_scores is not empty
                        if ncc_scores:
                            mean_score = np.mean(ncc_scores)
                            std_deviation = np.std(ncc_scores)
                            median_score = np.median(ncc_scores)
                    
                            cv_score = std_deviation / mean_score if mean_score != 0 else 0

                        series_report  = f"  Series: {series_name}\n"
                        series_report += f"  Number of unexpected slices: {len(misaligned_slices)}\n"
                        series_report += f"  Mean NCC score: {mean_score}\n"
                        series_report += f"  Median NCC score: {median_score}\n"
                        
                        for slice_info in misaligned_slices:
                            filename_1 = image_series[slice_info[0]][0]
                            filename_2 = image_series[slice_info[1]][0]
                            series_report += f"  {filename_1} and {filename_2} with NCC: {slice_info[2]}\n"
    
                        report_details.append(series_report)
                        total_misaligned_slices += len(misaligned_slices)
                            
        warning_counts = {"W3701": total_misaligned_slices if total_misaligned_slices > 0 else None} 
        
        generate_check_file_image_data(
            check_name=phase_name,
            phase_number=phase_number,
            series_group_name=series_group,
            num_slices_group=num_slices_group,
            num_patients_with_image_data=len(os.listdir(self.output_directory_data)),
            num_series=total_series,
            num_tot_slices=total_slices,
            timestamp=current_datetime,
            error_counts={},  
            warning_counts=warning_counts,
            error_descriptions={},
            warning_descriptions=warning_descriptions,
            output_dir=self.output_directory_checkfile,
            existing_warnings=existing_warnings
        )
        
        # Finalize the report content only if there are misaligned slices
        if total_misaligned_slices > 0:
            # Compile report content
            report_content = f"Report generated on: {current_datetime}\n\n"
            report_content += f"{phase_name} report:\n\n"
            report_content += f"Warning W3701: {warning_descriptions.get('W3701', None)}\n"
            report_content += "- Details:\n"
            report_content += "\n".join(report_details)  #report_content += "\n".join("  " + report_details)
            report_content += f"\nTotal Warnings: {total_misaligned_slices}\n\n"

            
            # Save the report to the "report_output" folder
            report_filename = f"{report_name}.txt"
            report_filepath = os.path.join(self.output_directory_report, report_filename)
            
            with open(report_filepath, 'w') as report_file:
                report_file.write(report_content)
                
            message_series_alignment += f"Warning W3701: {total_misaligned_slices} misaligned slices."
        else:
            message_series_alignment += "All slices across all series are consistently aligned."
            
        return message_series_alignment, total_misaligned_slices
    

    def create_and_save_MIP_image(self):
        
        mip_message = ""
        
        series_group = self.series_group_name
        image_group_config = self.protocol.get(series_group, {}).get("image", {}) #original self.protocol["image"]

        normalization_option = image_group_config["normalization"]["selected"]
        scaling_option = image_group_config["pixel scaling"]["selected"]
        
        # Check if scaling is needed based on the normalization and scaling options
        should_scale = not (normalization_option == "intensity_scaling" and scaling_option == [0, 255])

        for patient_folder in os.listdir(self.output_directory_data):
            patient_folder_path = os.path.join(self.output_directory_data, patient_folder)
            if not os.path.isdir(patient_folder_path):
                continue

            # Find the series folder that starts with series_group
            series_folder = next(
                (folder for folder in os.listdir(patient_folder_path)
                    if folder.startswith(series_group) and os.path.isdir(os.path.join(patient_folder_path, folder))),
                None
            )
                
            if series_folder:
                series_path = os.path.join(patient_folder_path, series_folder)

                for file_name in os.listdir(series_path):
                    if file_name.startswith("tensor") and file_name.endswith('.npy'):
                        file_path = os.path.join(series_path, file_name)
        
                        # Load the 3D MRI volume from the .npy file
                        volume = np.load(file_path)
                        
                        # Choose the axis for MIP (e.g., axial view)
                        axis = 0  # Set this to the desired axis (0 for sagittal, 1 for coronal, 2 for axial)
                        central_slice = volume.shape[axis] // 2
                        
                        # Define the range of slices: 5 slices before and after the central slice
                        start_slice = max(0, central_slice - 5)
                        end_slice = min(volume.shape[axis], central_slice + 5)
                
                        # Extract the subset of slices along the chosen axis
                        if axis == 0:
                            subset_volume = volume[start_slice:end_slice + 1, :, :]
                        elif axis == 1:
                            subset_volume = volume[:, start_slice:end_slice + 1, :]
                        elif axis == 2:
                            subset_volume = volume[:, :, start_slice:end_slice + 1]
                
                        # Calculate the Maximum Intensity Projection (MIP)
                        mip_image = np.max(subset_volume, axis=axis)
                
                        # Scale the MIP image if needed
                        if should_scale:
                            scaled_mip_image = ((mip_image - mip_image.min()) / (mip_image.max() - mip_image.min())) * 255
                            final_mip_image = np.uint8(scaled_mip_image)
                        else:
                            final_mip_image = np.uint8(mip_image)
                
                        # Save the resulting MIP image as a PNG
                        output_file_name = file_name.replace('.npy', '.png').replace('tensor', 'MIP')
                        output_file_path = os.path.join(series_path, output_file_name)                
                        final_image = Image.fromarray(final_mip_image)
                        final_image.save(output_file_path)
                        print(f"Saved MIP image for {file_path} to {output_file_path}")
                        
        # Generate and return a summary message
        mip_message += (
            f"MIP images have been saved in .png format into series folders within patient folders "
            f"in {self.host_output_directory_data}. " #!! Path del container, deve essere quello locale.ok
        )
        return mip_message
    

    def extract_features_from_MIP(self): #solo 3D (using Pytorch)
        
        series_group = self.series_group_name
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load pretrained InceptionV3 with appropriate config
        weights = Inception_V3_Weights.DEFAULT
        inception = inception_v3(weights=weights, aux_logits=True)
        inception.fc = nn.Identity()  # remove final classification layer
        inception.to(device)
        inception.eval()

        # Image transform pipeline
        transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],  # ImageNet mean
                                 [0.229, 0.224, 0.225])  # ImageNet std
        ])
    
        features = []  # List to store features from each MIP image
    
        # Iterate over each patient folder in the output directory
        for patient_folder in os.listdir(self.output_directory_data):
            patient_folder_path = os.path.join(self.output_directory_data, patient_folder)
            if os.path.isdir(patient_folder_path):
                
                # Find the series folder that starts with series_group
                series_folder = next(
                    (folder for folder in os.listdir(patient_folder_path)
                        if folder.startswith(series_group) and os.path.isdir(os.path.join(patient_folder_path, folder))),
                    None
                )
                    
                if series_folder:
                    series_path = os.path.join(patient_folder_path, series_folder)
                        
                    # Look for MIP images
                    for file_name in os.listdir(series_path):
                        if file_name.startswith('MIP'):
                            img_path = os.path.join(series_path, file_name)
                            try:
                                img = Image.open(img_path).convert('RGB')
                                img_tensor = transform(img).unsqueeze(0).to(device)
            
                                with torch.no_grad():
                                    feat = inception(img_tensor).cpu().numpy().flatten()
                                    features.append(feat)
    
                                # Save individual feature vector as .npy
                                feature_file_name = file_name.replace('MIP', 'features').replace('.png', '.npy')
                                feature_file_path = os.path.join(series_path, feature_file_name)
                                np.save(feature_file_path, feat)
                                print(f"Saved feature vector to {feature_file_path}") #!! Path del container, si può non cambiare perchè è un print 
                                
                            except Exception as e:
                                print(f"Failed to process image {img_path}: {e}") #!! Path del container, sì può non cambiare perchè è un print 
    
        # Convert list of features to a NumPy array
        features = np.array(features)
        
        # Compute mu and sigma
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
    
        message_extract_feat = "Computed mean and covariance of MIP image features, and saved per-image feature vectors."
    
        return mu, sigma, message_extract_feat
    

    def generate_NiftiReorientPreprocess_final_report(self, reoriented_occurred, final_processing_summary, final_message_resize, replicate_slice_message, warnings_replicates, final_message_npy_and_labels, message_from_npy_to_image, message_series_alignment, total_misaligned_slices, mip_message, message_extract_feat):
        # Get the current datetime
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
        
        # Define the report filename and path
        report_filename = "2.NiftiReorientPreprocess_final_report.txt"
        report_filepath = os.path.join(self.output_directory_report, report_filename)

        # Initialize report content with ordered titles and messages
        report_content = []

        report_content.append(f"Report generated on: {formatted_datetime}\n")
        
        report_content.append("NIfTI reorientation and preprocessing final report:\n")
        
        report_content.append("1. Volume reorientation check:")
        # Volume reorientation check
        if reoriented_occurred:
            report_content.append("- Volumes reorientation occurred.")
        report_content.append("- Reference volume orientation: LPS.\n")

        report_content.append("2. Processing summary:")
        report_content.append(final_processing_summary + "\n")

        report_content.append("3. Resizing status:")
        report_content.append("- " + final_message_resize + "\n")

        report_content.append("4. Replicate image check:")
        report_content.append("- " + replicate_slice_message + "\n")

        report_content.append("5. Tensors and labels save status:")
        report_content.append("- " + final_message_npy_and_labels + "\n")
        
        report_content.append("6. Image generation status:")
        report_content.append("- " + message_from_npy_to_image + "\n")
        
        report_content.append("7. Slices alignment summary:")
        report_content.append("- " + message_series_alignment + "\n")

        report_content.append("8. MIP image generation status:")
        report_content.append("- " + mip_message + "\n")

        # Add feature extraction status
        report_content.append("9. Feature extraction status:")
        report_content.append("- " + message_extract_feat + "\n")

        # Count the total number of warnings
        total_warnings = total_misaligned_slices + warnings_replicates
        
        # Add warning summary
        report_content.append(f"Total warnings detected: {total_warnings}.")
        
        # Write the report to the specified file
        with open(report_filepath, 'w') as report_file:
            report_file.write("\n".join(report_content))
            
        print(f"Final report saved to {report_filepath}") #!! Path del container, si può non cambiare (print)


def run_nifti_reorient_preprocessor(protocol, local_config, mapping_file, series_group_name):
    
    # Define the input directory path
    input_dir = os.getenv("INPUT_DIR")

    # Clear log at the start of validation
    clear_log_file(input_dir, LOG_FILENAME)

    # Load the input validation state to extract num_image_patients
    input_state_file = os.path.join(input_dir, "input_validation_state.json")

    try:
        with open(input_state_file, "r") as f:
            input_state = json.load(f)
        num_image_patients = input_state.get("num_patients_with_image_data", 0)
    except Exception as e:
        raise RuntimeError(f"Failed to load state file '{input_state_file}': {e}")
    
    nifti_preprocessor = NiftiReorientPreprocessor(protocol, local_config, mapping_file, num_image_patients, series_group_name)

    state = load_state(nifti_preprocessor.state_file)
    series_progress_state =  load_state(nifti_preprocessor.series_progress_file)

    # Access or create the per-series progress dictionary
    series_state = series_progress_state.setdefault(series_group_name, {})
    last_phase_done = series_state.get("last_successful_phase", 0)

    if last_phase_done < 36:

        print("Running check_and_reorient_niftivolumes function...")
        try:
           reoriented_occurred = nifti_preprocessor.check_and_reorient_niftivolumes()
        except Exception as e:
            log_error(input_dir, "check_and_reorient_niftivolumes", e, LOG_FILENAME)
            print(f"An unexpected error occurred during check_and_reorient_niftivolumes. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
            raise 

        print("Running process_array function...")
        try:
            final_processing_summary = nifti_preprocessor.process_array()
        except Exception as e:
            log_error(input_dir, "process_array", e, LOG_FILENAME)
            print(f"An unexpected error occurred during process_array. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
            raise 

        print("Running resize_array function...")
        try:
            final_message_resize = nifti_preprocessor.resize_array()
        except Exception as e:
            log_error(input_dir, "resize_array", e, LOG_FILENAME)
            print(f"An unexpected error occurred during resize_array. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
            raise 

        print("Running save_labels function...")
        try:
            final_message_npy_and_labels = nifti_preprocessor.save_labels()
        except Exception as e:
            log_error(input_dir, "save_labels", e, LOG_FILENAME)
            print(f"An unexpected error occurred during save_labels. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
            raise

        print("Running convert_npy_to_image function...")
        try:
            message_from_npy_to_image = nifti_preprocessor.convert_npy_to_image()
        except Exception as e:
            log_error(input_dir, "convert_npy_to_image", e, LOG_FILENAME)
            print(f"An unexpected error occurred during convert_npy_to_image. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
            raise

        print("Running create_and_save_MIP_image function...")
        try:
            mip_message = nifti_preprocessor.create_and_save_MIP_image()
        except Exception as e:
            log_error(input_dir, "create_and_save_MIP_image", e, LOG_FILENAME)
            print(f"An unexpected error occurred during create_and_save_MIP_image. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
            raise

        print("Running extract_features_from_MIP function...")
        try:
            _, _, message_extract_feat = nifti_preprocessor.extract_features_from_MIP()
        except Exception as e:
            log_error(input_dir, "extract_features_from_MIP", e, LOG_FILENAME)
            print(f"An unexpected error occurred during extract_features_from_MIP. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
            raise

        print("Running check_image_replicates_after_processing function...")  # check phase 36
        try:
            replicate_slice_message, state["num_slices_group"], state["total_series"], state["total_slices"], state["warnings_replicates"], input_state["num_patients_with_image_data"] = nifti_preprocessor.check_image_replicates_after_processing()
            save_state(state, nifti_preprocessor.state_file)
            save_state(input_state, input_state_file)
            
            series_state["last_successful_phase"] = 36
            series_progress_state[series_group_name] = series_state
            save_state(series_progress_state, nifti_preprocessor.series_progress_file)
        except Exception as e:
            log_error(input_dir, "check_image_replicates_after_processing", e, LOG_FILENAME)
            print(f"An unexpected error occurred during check_image_replicates_after_processing. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
            raise

        print("Running analyze_series_alignment function...")
        try:
            message_series_alignment, total_misaligned_slices = nifti_preprocessor.analyze_series_alignment()
        except Exception as e:
            log_error(input_dir, "analyze_series_alignment", e, LOG_FILENAME)
            print(f"An unexpected error occurred during analyze_series_alignment. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
            raise

        print("Running generate_NiftiReorientPreprocess_final_report function...")
        try:
            nifti_preprocessor.generate_NiftiReorientPreprocess_final_report(reoriented_occurred, final_processing_summary, final_message_resize, replicate_slice_message, state.get("warnings_replicates"), final_message_npy_and_labels, message_from_npy_to_image, message_series_alignment, total_misaligned_slices, mip_message, message_extract_feat)
        except Exception as e:
            log_error(input_dir, "generate_NiftiReorientPreprocess_final_report", e, LOG_FILENAME)
            print(f"An unexpected error occurred during generate_NiftiReorientPreprocess_final_report. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
            raise





    
        
    



    


