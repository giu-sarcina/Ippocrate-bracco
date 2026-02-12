import numpy as np
import os 
from datetime import datetime
import glob
from PIL import Image
import shutil
import json
from skimage.transform import resize
from skimage.metrics import structural_similarity as ssim
import re 
import stat
import torch
import torchvision.transforms as transforms
from torchvision.models import inception_v3, Inception_V3_Weights
from torch import nn
from scipy.spatial.distance import pdist, squareform
from utils import save_state, load_state, clear_log_file, log_error
from ...helpers import (
    generate_check_file_image_data,
    retrieve_existing_warnings_image_data,
    extract_metadata_from_check_file,
    load_patient_series_data,
    load_patient_feature_vectors,
    load_patient_mappings,
    load_series_mappings,
    image_wise_clipping,
    intensity_scaling,
    standardization_mean0_std1,
    apply_clahe,
    count_series_group_slices_from_npy,
    count_tot_num_series_per_dataset,
    count_tot_num_slices_per_dataset
)

LOG_FILENAME = "2D_image_preprocessing_error.log"

class ImagePreprocessor:

    def __init__(self, protocol, local_config, mapping_file, num_image_patients, series_group_name):
        self.protocol = protocol
        self.local_config = local_config
        self.mapping_file = mapping_file
        self.series_group_name = series_group_name
        self.input_dir = os.getenv("INPUT_DIR")
        self.images_dir = os.path.join(self.input_dir, "IMAGES")
        self.host_input_dir = os.getenv("HOST_BASE_DATA_DIR")
        self.state_file = os.path.join(self.input_dir, "image_validation_state.json")
        self.series_progress_file = os.path.join(self.input_dir, "validation_progress_by_series.json")
        self.num_image_patients = num_image_patients
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


    def process_array(self):
        """
        Processes clinical images (PNG, JPG, TIFF) in the input directory, applying clipping and normalization.
    
        Returns:
        final_processing_summary: A string summarizing the processing steps.
        """

        local_config = self.local_config
        series_group = self.series_group_name
        
        patient_id_map = load_patient_mappings(local_config, self.input_dir)
        series_id_map = load_series_mappings(self.images_dir, series_group)
       
        series_group_mapping = self.local_config["Local config"]["radiological data"]["series_mapping"]
        series_group_subfolder = series_group_mapping.get(series_group) # original series group name (ex. "*/series1")
        image_group_config = self.protocol.get(series_group, {}).get("image", {})
        
        image_type = image_group_config["type"]["selected"]
        image_format_selected = image_group_config["image file format"]["selected"]
        clipping_option = image_group_config["clipping"]["selected"]
        normalization_option = image_group_config["normalization"]["selected"]
        pixel_scaling = image_group_config["pixel scaling"]["selected"]
        contrast_optimization = image_group_config.get("contrast_optimization_DX/CR", {}).get("selected", "none")

        processing_summary = []

        # Track whether clipping or normalization was applied
        clipping_applied = False
        normalization_applied = False
        clahe_applied = False

        patient_folders = [f for f in os.listdir(self.images_dir) if os.path.isdir(os.path.join(self.images_dir, f)) and f != "Reports"]
        
        # Iterate over patient folders
        for orig_patient_id in patient_folders:
            patient_folder_path = os.path.join(self.images_dir, orig_patient_id)

            # Map to new patient ID
            new_patient_id = patient_id_map.get(orig_patient_id)
            
            series_folder_name = os.path.normpath(series_group_subfolder).split(os.sep)[-1].strip() # ex. series1
            series_folder_path = os.path.join(patient_folder_path, series_folder_name) #ex. /data/cristina.iudica/patient1/series1
            if os.path.isdir(series_folder_path):
                full_series_key = os.path.join(orig_patient_id, series_folder_name)
                # If mappings use Windows-style separators, normalize only for lookup
                if "\\" in list(series_id_map.keys())[0]:  # Detect if mapping keys use '\'
                    full_series_key = full_series_key.replace("/", "\\")
                new_series_name = series_id_map.get(full_series_key)

                if new_series_name and "_" in new_series_name:
                    parts = new_series_name.split("_")
                    if len(parts) >= 3:
                        # Reorder: ["series001", "ct", "YWPXH"] → "ct_YWPXH_series001"
                        new_series_name2 = f"{parts[1]}_{parts[2]}_{parts[0]}"


            # Read all image files inside the series folder
            image_files = [
                f for f in os.listdir(series_folder_path)
                if f.lower().endswith(image_format_selected)
            ]

            processed_slices = []

            # Iterate through the DICOM files in the series (in the order they appear in the dictionary)
            for slice_idx, img_file in enumerate(image_files, start=1):
                img_path = os.path.join(series_folder_path, img_file)
                image = Image.open(img_path).convert("L") # Open the image and convert to grayscale
                image_array = np.array(image)
                normalized_image = None # Initialize to ensure it's always defined

                if image_type in ["CR", "DX", "RX"]:
                    # Apply clipping
                    if clipping_option == "image-wise":
                        clipped_image = image_wise_clipping(image_array, image_group_config)
                        clipping_applied = True
                    elif clipping_option == "db-wise (only MRI)":
                        raise ValueError("Selected clipping method is not valid for X-ray images!")
                    elif clipping_option == "none":
                        clipped_image = image_array  # No clipping, use the original image 

                    # Apply CLAHE if required
                    if contrast_optimization == "yes":
                        clipped_image = apply_clahe(clipped_image)
                        clahe_applied = True

                    # Apply normalization
                    if normalization_option == "intensity_scaling" and not (clahe_applied and pixel_scaling == [0, 255]):
                        normalized_image = intensity_scaling(clipped_image, image_group_config)
                        normalization_applied = True
                    elif normalization_option == "standardization_mean0_std1":
                        normalized_image = standardization_mean0_std1(clipped_image)
                        normalization_applied = True
                    elif normalization_option == "none":
                        normalized_image = clipped_image  # No normalization
                    else:
                        normalized_image = clipped_image

                # Convert the processed image to float64 before adding it to the list
                if normalized_image is not None:
                    normalized_image = normalized_image.astype(np.float64)

                # Append the processed image to the list of slices for this patient
                processed_slices.append(normalized_image.astype(np.float64))     

            print(f"Processing for patient {orig_patient_id}, series {series_folder_name} completed.")
            # Prepare output paths (anonymized)
            patient_out_path = os.path.join(self.output_directory_data, new_patient_id)
            series_out_path = os.path.join(patient_out_path, new_series_name)
            os.makedirs(series_out_path, exist_ok=True)

            # Save .npy
            npy_filename = f"tensor_{new_series_name2}.npy"
            np.save(os.path.join(series_out_path, npy_filename), processed_slices)

        # Summary details based on protocol settings
        if clipping_applied:
            processing_summary.append(f"- Clipping applied according to protocol: {clipping_option}.")
        else:
            processing_summary.append("- Clipping not required.")

        if clahe_applied:
            processing_summary.append("- CLAHE applied for contrast optimization.")
    
        if normalization_applied:
            processing_summary.append(f"- Normalization applied according to protocol: {normalization_option}.")
        else:
            processing_summary.append("- Normalization not required.")

        # Join summary statements and return it along with processed slices
        final_processing_summary = "\n".join(processing_summary)
        
        return final_processing_summary
    
    
    def resize_array(self):
        """
        Loads .npy tensors, resizes image slices to match the target size in the protocol,
        and overwrites the original .npy tensor file if resizing is needed.
        """
        output_directory_data = self.output_directory_data
        series_group = self.series_group_name
        image_group_config = self.protocol.get(series_group, {}).get("image", {}) #original self.protocol["image"]
        
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
        
        def on_rm_error(func, path, exc_info):
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
        if image_type in ["CR", "DX", "RX"]:
            warning_counts = {"W3601": None} 
        else:
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
            input_folder_path = os.path.join(self.images_dir, orig_patient_id)
            if os.path.exists(input_folder_path):
                shutil.rmtree(input_folder_path, onerror=on_rm_error)

        warning_counts["W3601"] = len(inter_replicate_patients) if inter_replicate_patients else None
        if image_type not in ["CR", "DX", "RX"]:
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
                             f"See the detailed report at: {host_report_path}") #!! Path del container, deve essere quello locale.OK
            
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

    
    def save_labels(self): #just for 2D scenario
        """
        Save patient label TXT files (if present) into the correct folder structure.
        Uses a simple counter to report actual saved labels.
        """

        local_config = self.local_config
        output_directory_data = self.output_directory_data

        series_group = self.series_group_name
        image_group_config = self.protocol.get(series_group, {}).get("image", {}) 

        patient_id_map = load_patient_mappings(local_config, self.input_dir)
        series_mapping = load_series_mappings(self.images_dir, series_group) 

        # normalize mapping keys 
        series_mapping = {os.path.normpath(k): v for k, v in series_mapping.items()}

        # Reverse maps for original IDs
        reversed_patient_id_map = {v: k for k, v in patient_id_map.items()}
        reversed_series_id_map = {v: k for k, v in series_mapping.items()}
           
        final_message_npy_and_labels = ""
        saved_label_count = 0
  
        # Load patient series data and iterate over them
        for patient_id, series_id, _, _, _, _ in load_patient_series_data(series_group, output_directory_data):
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
                os.path.join(orig_patient_id, orig_series_id)
            )
            
            if full_series_name not in series_mapping:
                raise KeyError(
                    f"Series mapping missing for '{full_series_name}'."
                )

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

                    saved_label_count += 1

                    print(f"Saved patient label file as {new_txt_filename} in {new_series_folder_path}") 

        final_message_npy_and_labels += f"Tensors saved in .npy format into series folders within patient folders in {self.host_output_directory_data}." 

        if "patient_label" in image_group_config:
            if saved_label_count > 0:
                final_message_npy_and_labels += "\n- Patient labels were saved as TXT files in the respective series folders."
            else:
                final_message_npy_and_labels += (
                    "\n- WARNING: Patient labels were expected by the protocol, "
                    "but no TXT label files were saved."
                )
            
        return final_message_npy_and_labels 
    

    def convert_npy_to_image(self):
        """Convert npy files to images."""

        series_group = self.series_group_name
        image_group_config = self.protocol.get(series_group, {}).get("image", {}) 
        
        # Extract image format from the protocol
        output_format = image_group_config["output format"]["selected"]
        normalization_option = image_group_config["normalization"]["selected"]
        scaling_option = image_group_config["pixel scaling"]["selected"]
        contrast_optimization = image_group_config.get("contrast_optimization_DX/CR", {}).get("selected", "none")
        
        # Check if scaling is needed based on the normalization and scaling options
        should_scale = not (
            (normalization_option == "intensity_scaling" and scaling_option == [0, 255]) or 
            (contrast_optimization == "yes" and normalization_option == "none")
        )

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
                                print(f"Saved {output_file_name} in {image_folder_path}")
                                    
        message_from_npy_to_image += f"Images in {output_format} format have been generated from npy arrays and saved into series folders within patient folders in {self.host_output_directory_data}."  #!! Path del container, deve essere quello locale.OK              
        return message_from_npy_to_image
    

    def extract_features_from_2Dimages(self):  # 2D scenario (Pytorch-based)

        series_group = self.series_group_name
        # chooses GPU if available, otherwise uses CPU.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        weights = Inception_V3_Weights.DEFAULT
        inception = inception_v3(weights=weights, aux_logits=True)  # Required to match the weight's config
        inception.fc = nn.Identity()  # Remove final classification layer since we're not classifying
        inception.to(device)
        inception.eval()
    
        transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            # normalization using ImageNet statistics (since the InceptionV3 model is pretrained on ImageNet
            transforms.Normalize([0.485, 0.456, 0.406],  # Imagenet mean
                                 [0.229, 0.224, 0.225])  # Imagenet std
        ])
    
        features = []
    
        for patient_folder in os.listdir(self.output_directory_data):
            patient_folder_path = os.path.join(self.output_directory_data, patient_folder)
            if not os.path.isdir(patient_folder_path):
                continue
    
            # Locate the series folder
            series_folder = next(
                (folder for folder in os.listdir(patient_folder_path)
                 if folder.startswith(series_group) and os.path.isdir(os.path.join(patient_folder_path, folder))),
                None
            )
    
            if not series_folder:
                continue
    
            series_path = os.path.join(patient_folder_path, series_folder)
    
            for subfolder in os.listdir(series_path):
                if subfolder.startswith('image'):
                    image_folder_path = os.path.join(series_path, subfolder)
                    if not os.path.isdir(image_folder_path):
                        continue
    
                    image_files = glob.glob(os.path.join(image_folder_path, '*.[pjJPtT][npNPiI][gGeEfF]'))  # jpg/jpeg/png/tiff/tif
                    if image_files:
                        img_path = image_files[0]
                        try:
                            img = Image.open(img_path).convert('RGB')
                            img_tensor = transform(img).unsqueeze(0).to(device)
    
                            with torch.no_grad(): #since during inference you don't need gradients
                                feat = inception(img_tensor).cpu().numpy().flatten()
                                features.append(feat)
    
                            # Create output file name
                            base_name = os.path.basename(img_path)
                            clean_name = re.sub(r'^image\d*', '', os.path.splitext(base_name)[0]).lstrip('_')
                            feature_name = f"features_{clean_name}.npy"
                            feature_path = os.path.join(series_path, feature_name)
    
                            np.save(feature_path, feat)
                            print(f"Saved features to {feature_path}") 
    
                        except Exception as e:
                            print(f"Failed to process image {img_path}: {e}")
    
        features = np.array(features)
        print(f"Number of feature vectors extracted: {len(features)}")
        if len(features) == 0:
            raise ValueError("No features were extracted. Check your input folders and image files.")
    
        # Compute statistics
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
    
        message_extract_feat = "Computed mean and covariance of 2D image features using PyTorch, and saved per-image feature vectors."

        return mu, sigma, message_extract_feat


    def generate_ImagePreprocess_final_report(self, final_processing_summary, final_message_resize, replicate_slice_message, warnings_replicates, final_message_npy_and_labels, message_from_npy_to_image, message_extract_feat):
        # Get the current datetime
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
        
        # Define the report filename and path
        report_filename = "2.ImagePreprocess_final_report.txt"
        report_filepath = os.path.join(self.output_directory_report, report_filename)

        # Initialize report content with ordered titles and messages
        report_content = []

        report_content.append(f"Report generated on: {formatted_datetime}\n")
        
        report_content.append("Image preprocessing final report:\n")
        
        report_content.append("1. Processing summary:")
        report_content.append(final_processing_summary + "\n")

        report_content.append("2. Resizing status:")
        report_content.append("- " + final_message_resize + "\n")

        report_content.append("3. Replicate image check:")
        report_content.append("- " + replicate_slice_message + "\n")

        report_content.append("4. Tensors and labels save status:")
        report_content.append("- " + final_message_npy_and_labels + "\n")
        
        report_content.append("5. Image generation status:")
        report_content.append("- " + message_from_npy_to_image + "\n")

        # Add feature extraction status
        report_content.append("6. Feature extraction status:")
        report_content.append("- " + message_extract_feat + "\n")

        # Add warning summary
        report_content.append(f"Total warnings detected: {warnings_replicates}.")
        
        # Write the report to the specified file
        with open(report_filepath, 'w') as report_file:
            report_file.write("\n".join(report_content))
            
        print(f"Final report saved to {report_filepath}")


def run_image_preprocessor(protocol, local_config, mapping_file, series_group_name):

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
    
    image_preprocessor = ImagePreprocessor(protocol, local_config, mapping_file, num_image_patients, series_group_name)

    state = load_state(image_preprocessor.state_file)
    series_progress_state =  load_state(image_preprocessor.series_progress_file)

    # Access or create the per-series progress dictionary
    series_state = series_progress_state.setdefault(series_group_name, {})
    last_phase_done = series_state.get("last_successful_phase", 0)

    if last_phase_done < 36:

        print("Running process_array function...")
        try:
            final_processing_summary = image_preprocessor.process_array()
        except Exception as e:
            log_error(input_dir, "process_array", e, LOG_FILENAME)
            print(f"An unexpected error occurred during process_array. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
            raise 

        print("Running resize_array function...")
        try:
            final_message_resize = image_preprocessor.resize_array()
        except Exception as e:
            log_error(input_dir, "resize_array", e, LOG_FILENAME)
            print(f"An unexpected error occurred during resize_array. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
            raise 

        print("Running save_labels function...")
        try:
            final_message_npy_and_labels = image_preprocessor.save_labels()
        except Exception as e:
            log_error(input_dir, "save_labels", e, LOG_FILENAME)
            print(f"An unexpected error occurred during save_labels. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
            raise  

        print("Running convert_npy_to_image function...")
        try:
            message_from_npy_to_image = image_preprocessor.convert_npy_to_image()
        except Exception as e:
            log_error(input_dir, "convert_npy_to_image", e, LOG_FILENAME)
            print(f"An unexpected error occurred during convert_npy_to_image. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
            raise  

        print("Running extract_features_from_2Dimages function...")
        try:
            _, _, message_extract_feat = image_preprocessor.extract_features_from_2Dimages()
        except Exception as e:
            log_error(input_dir, "extract_features_from_2Dimages", e, LOG_FILENAME)
            print(f"An unexpected error occurred during extract_features_from_2Dimages. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
            raise

        print("Running check_image_replicates_after_processing function...")  # check phase 36
        try:
            replicate_slice_message, state["num_slices_group"], state["total_series"], state["total_slices"], state["warnings_replicates"], input_state["num_patients_with_image_data"] = image_preprocessor.check_image_replicates_after_processing()
            save_state(state, image_preprocessor.state_file)  # Save updated state
            save_state(input_state, input_state_file)
            
            series_state["last_successful_phase"] = 36
            series_progress_state[series_group_name] = series_state
            save_state(series_progress_state, image_preprocessor.series_progress_file)
        except Exception as e:
            log_error(input_dir, "check_image_replicates_after_processing", e, LOG_FILENAME)
            print(f"An unexpected error occurred during check_image_replicates_after_processing. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
            raise    

        print("Running generate_ImagePreprocess_final_report function...")
        try:
            image_preprocessor.generate_ImagePreprocess_final_report(final_processing_summary, final_message_resize, replicate_slice_message, state.get("warnings_replicates"), final_message_npy_and_labels, message_from_npy_to_image, message_extract_feat)
        except Exception as e:
            log_error(input_dir, "generate_ImagePreprocess_final_report", e, LOG_FILENAME)
            print(f"An unexpected error occurred during generate_ImagePreprocess_final_report. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
            raise  






    
