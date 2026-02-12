import pandas as pd
import numpy as np
import os 
from datetime import datetime
import shutil
import json
from collections import defaultdict
import nibabel as nib
import scipy.ndimage 
import re
import gc
import tempfile
from utils import save_state, load_state, clear_log_file, log_error
from ...helpers import (
    generate_check_file_image_data,
    retrieve_existing_warnings_image_data,
    load_patient_series_seg_data,
    load_patient_mappings,
    load_series_mappings
)

LOG_FILENAME = "niftiseg_preprocessing_error.log"

class NiftiSegReorientPreprocessor:

    def __init__(self, protocol, local_config, mapping_file, num_image_patients, num_slices_group, total_series, total_slices, series_group_name):
        self.protocol = protocol
        self.local_config = local_config
        self.mapping_file = mapping_file
        self.num_image_patients = num_image_patients
        self.num_slices_group = num_slices_group
        self.total_series = total_series
        self.total_slices = total_slices
        self.series_group_name = series_group_name
        self.study_path = os.getenv("ROOT_NAME")
        self.host_study_path = os.getenv("HOST_ROOT_DIR")
        self.input_dir = os.getenv("INPUT_DIR")
        self.host_input_dir = os.getenv("HOST_BASE_DATA_DIR")
        self.images_dir = os.path.join(self.input_dir, "IMAGES")
        self.state_file = os.path.join(self.input_dir, "image_validation_state.json")
        self.series_progress_file = os.path.join(self.input_dir, "validation_progress_by_series.json")
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
    

    def check_and_reorient_niftiseg(self):
        """
        Checks orientation of NIfTI segmentation files, reorients to LPS if necessary,
        and saves each segment as .npy in the output segmentation folder.
        """
        reoriented_occurred = False  # Flag to track if any segmentation was reoriented

        series_group = self.series_group_name
        series_group_mapping = self.local_config["Local config"]["radiological data"]["series_mapping"]
        series_group_subfolder = series_group_mapping.get(series_group) #input name of the series

        segmentation_format = self.protocol.get(series_group, {}).get("segmentation", {}).get("segmentation_input_format", {}).get("selected")
        segment_labels = self.protocol.get(series_group, {}).get("segmentation", {}).get("segments_labels", [])
        single_mask = segmentation_format == "single-mask"
        multi_segment = len(segment_labels) > 2

        patient_id_map = load_patient_mappings(self.local_config, self.input_dir)
        series_id_map = load_series_mappings(self.images_dir, series_group)
        
        patient_folders = [f for f in os.listdir(self.images_dir) if os.path.isdir(os.path.join(self.images_dir, f)) and f != "Reports"]

        # Iterate through all patient folders
        for orig_patient_id in patient_folders:
            patient_folder_path = os.path.join(self.images_dir, orig_patient_id)
            if os.path.isdir(patient_folder_path):
                series_folder = os.path.normpath(series_group_subfolder).split(os.sep)[-1].strip()
                series_folder_path = os.path.join(patient_folder_path, series_folder)
                
                if os.path.isdir(series_folder_path):
                    full_series_key = os.path.join(orig_patient_id, series_folder)
                    # If mappings use Windows-style separators, normalize only for lookup
                    if "\\" in list(series_id_map.keys())[0]:  # Detect if mapping keys use '\'
                        full_series_key = full_series_key.replace("/", "\\")
                   
                    new_series_name = series_id_map.get(full_series_key)
                    new_patient_id = patient_id_map.get(orig_patient_id)

                    if new_series_name and "_" in new_series_name:
                        parts = new_series_name.split("_")
                        new_series_name2 = f"{parts[1]}_{parts[2]}_{parts[0]}"
                        seg_folder_name = f"seg_{parts[1]}_{parts[2]}_{parts[0]}"
                    else:
                        continue  # Skip invalid series names

                    # Get segmentation folder (first subfolder)
                    subfolders = [f for f in os.listdir(series_folder_path) if os.path.isdir(os.path.join(series_folder_path, f))]
                    segmentation_folder = os.path.join(series_folder_path, subfolders[0])
                    nifti_files = [f for f in os.listdir(segmentation_folder) if f.endswith('.nii') or f.endswith('.nii.gz')]
                    
                    for nifti_filename in nifti_files:
                        nifti_file_path = os.path.join(segmentation_folder, nifti_filename)
                        nifti_img = nib.load(nifti_file_path)
                        nifti_data = nifti_img.get_fdata()
                        affine = nifti_img.affine
                        
                        # Check the current orientation and convert to LPS if needed
                        orientation_string = self.orientation_to_string(affine)
                        print(f"Current orientation for {nifti_filename}: {orientation_string}")

                        if orientation_string != "LPS":
                            # Reorient the data to LPS if needed
                            nifti_data, affine, reoriented_occurred_here = self.convert_to_lps(nifti_data, affine)

                            if reoriented_occurred_here:
                                reoriented_occurred = True  # Mark reorientation occurred
                                print(f"Reoriented {nifti_filename} to LPS.")

                        # Prepare output path
                        patient_out_path = os.path.join(self.output_directory_data, new_patient_id)
                        series_out_path = os.path.join(patient_out_path, new_series_name)
                        seg_out_path = os.path.join(series_out_path, seg_folder_name)
                        os.makedirs(seg_out_path, exist_ok=True)

                        # Handle multi-segment in single-mask format
                        if single_mask and multi_segment and nifti_data.ndim == 3:
                            label = nifti_filename
                            if label.endswith('.nii.gz'):
                                label = label[:-7]  # remove '.nii.gz'
                            else:
                                label = label[:-4]  # remove '.nii'
                            npy_filename = f"seg_{new_series_name2}_segment{label}.npy"
                        else:
                            npy_filename = f"seg_{new_series_name2}.npy"

                        nifti_data = nifti_data.astype(np.int16)
                        save_path = os.path.join(seg_out_path, npy_filename)
                        nifti_data = np.transpose(nifti_data, (2, 1, 0)) # put the number of slices in the first position of the array
                        np.save(save_path, nifti_data)
                        print(f"Saved {npy_filename} for {new_patient_id}/{new_series_name}")

        # Set reorientation message if any segmentation was reoriented
        reorient_message = "Segmentations reorientation occurred." if reoriented_occurred else ""
        
        return reorient_message
    

    def _build_multilabel(self, segmentations, label_map):
        """
        Helper method to build multi-label tensor.
        
        Args:
            segmentations (dict): {segment_number: (tensor, npy_path), ...}
            label_map (dict): {normalized_label (str): segment_code (int)}
        
        Returns:
            multi_label_tensor (np.ndarray): Combined multi-label segmentation tensor.
        """
        # Assumes all tensors have same shape
        first_tensor = next(iter(segmentations.values()))[0]
        num_slices, height, width = first_tensor.shape
        multi_label_tensor = np.zeros((num_slices, height, width), dtype=np.int16)
    
        for segment_label, (tensor, _, _) in segmentations.items():
            # Normalize the label to match keys in label_map
            normalized_label = segment_label.lower().replace("_", " ").replace("-", " ").replace("  ", " ")
            segment_code = label_map.get(normalized_label)
            if segment_code is None:
                raise ValueError(f"Label '{normalized_label}' not found in label_map.")
            
            for z in range(num_slices):
                binary_mask = tensor[z] != 0 # instead of == 1 (accept all values different from 0 for the foreground
                multi_label_tensor[z][binary_mask] = segment_code
    
        return multi_label_tensor

    
    def single_to_multi_label_conversion(self, multiple_seg_file_flag): # 46
        """
        Converts multiple single-label NIfTI files into a single multi-label NIfTI file.

        Parameters:
            multiple_seg_file_flag (bool): Flag indicating whether multiple segment files are expected.
        
        Returns:
            str: Success or skip message.
        
        Raises:
            ValueError: If multiple_seg_file_flag is None.
        """
        if multiple_seg_file_flag is None:
            raise ValueError("Missing required 'multiple_seg_file_flag'. Cannot proceed with multi-label conversion.")
        
        series_group = self.series_group_name
        seg_group_config = self.protocol.get(series_group, {}).get("segmentation", {}) #original self.protocol["segmentation"]

        seg_input_format = seg_group_config["segmentation_input_format"]["selected"]

        series_group = self.series_group_name
        output_directory_data = self.output_directory_data
        
        # Check if multi-label conversion is necessary
        if seg_input_format == "multi-mask":
            single_to_multilabel_message = "Multi-label conversion skipped: input format is already 'multi-mask'."
            return single_to_multilabel_message
 
        if seg_input_format == "single-mask" and len(seg_group_config["segments_labels"]) == 2: # 2 since segments_labels includes the background
            single_to_multilabel_message = "Multi-label conversion skipped: input format is 'single-mask' with 1 label."
            return single_to_multilabel_message

        if seg_input_format == "single-mask" and multiple_seg_file_flag:
            
            # Extract segmentation labels and codes from the protocol
            segment_labels = [label.lower().replace("_", " ").replace("-", " ").replace("  ", " ") 
                              for label in seg_group_config["segments_labels"][1:]]  # Skip "background"
            segment_codes = seg_group_config["segmentation_code"][1:]  # Skip background code (assumed 0)
            
            # Create the label_map dictionary
            label_map = {label.lower(): code for label, code in zip(segment_labels, segment_codes)}

            # Create a mapping of original filenames to normalized filenames (lowercase, no underscores or hyphens)
            filename_mapping = {}            
            all_segment_file_paths = set()

            # For per-patient/series grouping
            current_key = None
            segmentations = {}

            # Iterate over patients and series
            for patient_id, series_id, npy_path, tensor, segment_label in load_patient_series_seg_data(series_group, output_directory_data):
                if segment_label is None:
                    continue  # Skip entries with no segment label

                # Build filename mapping on the fly, only once per segment_label
                if segment_label not in filename_mapping and "_segment" in os.path.basename(npy_path):
                    normalized_label = segment_label.lower().replace("_", " ").replace("-", " ").replace("  ", " ")
                    filename_mapping[segment_label] = normalized_label
                    
                key = (patient_id, series_id)

                if current_key is not None and key != current_key:
                    # Process previous group
                    multi_label_tensor = self._build_multilabel(segmentations, label_map)

                    # Save
                    example_path = next(iter(segmentations.values()))[1]
                    directory, filename = os.path.split(example_path)
                    base_filename = re.sub(r'_segment[^.]+', '', filename)
                    output_path = os.path.join(directory, base_filename)
                    np.save(output_path, multi_label_tensor)

                    # Delete single-label .npy files for this group
                    for path in all_segment_file_paths:
                        try:
                            os.remove(path)
                        except Exception as e:
                            print(f"Warning: Could not delete {path}: {e}")
                    all_segment_file_paths.clear()  # Reset for next group
                    
                    # Clear for next patient-series
                    segmentations.clear()
                    current_key = key
               
                # Extract the original segmentation file name from the npy path
                segment_match = re.search(r'segment(.+)\.npy$', os.path.basename(npy_path))
                original_filename = segment_match.group(1) + ".nii.gz" if segment_match else os.path.basename(npy_path).replace(".npy", ".nii.gz")
                # Accumulate segmentations for current patient-series
                segmentations[segment_label] = (tensor, npy_path, original_filename)
                
                if re.search(r'_segment[^.]+\.npy$', npy_path):
                    all_segment_file_paths.add(npy_path)
                current_key = key
        
            # Process last patient-series after loop ends
            if current_key is not None and segmentations:
                multi_label_tensor = self._build_multilabel(segmentations, label_map)
                example_path = next(iter(segmentations.values()))[1]
                directory, filename = os.path.split(example_path)
                base_filename = re.sub(r'_segment[^.]+', '', filename)
                
                output_path = os.path.join(directory, base_filename)
                np.save(output_path, multi_label_tensor)

                # Delete remaining single-label .npy files
                for path in all_segment_file_paths:
                    try:
                        os.remove(path)
                    except Exception as e:
                        print(f"Warning: Could not delete {path}: {e}")
                        
            single_to_multilabel_message = "Multi-label conversion successful."
            return single_to_multilabel_message
        
    
    def discard_black_seg_slices(self):
        """
        Reads discarded_slices.csv to identify black slices removed from volume data,
        and removes the corresponding slices from segmentation .npy files on disk.
        If no changes are needed, leaves the files untouched.
        """
        removal_message = ""
        discarded_slices_file = os.path.join(self.images_dir, "discarded_slices.csv")
        series_group = self.series_group_name

        if not os.path.exists(discarded_slices_file):
            # If the CSV is not found, return a message indicating no slices were removed
            removal_message += "No segmentation slices were removed."
            return removal_message

        # Load discarded slices data from CSV
        try:
            discarded_slices_df = pd.read_csv(discarded_slices_file)
        except Exception as e:
            raise Exception(f"Failed to read discarded slices CSV: {str(e)}")

        # Load patient and series ID maps
        patient_id_map = load_patient_mappings(self.local_config, self.input_dir)
        series_id_map = load_series_mappings(self.images_dir, series_group)

        # Iterate through each row in the CSV
        discarded_by_key = defaultdict(list)
        for _, row in discarded_slices_df.iterrows():
            orig_patient = row['Patient Name']
            orig_series = row['Series Folder Name']
            lookup_key = f"{orig_patient}\\{orig_series}"
            discarded_index = row['Discarded Slice Index'] - 1  # Convert 1-based to 0-based

            patient_id = patient_id_map.get(orig_patient)
            series_id = series_id_map.get(lookup_key)

            if patient_id is None or series_id is None:
                continue
            
            key = (patient_id, series_id)
            discarded_by_key[key].append(discarded_index)

        total_removed = 0
        total_invalid = 0

        for (patient, series), indices in discarded_by_key.items():
            # Locate the segmentation folder
            series_folder  = os.path.join(self.output_directory_data, patient, series)
            if not os.path.exists(series_folder ):
                print(f"Warning: Series folder not found for {patient}/{series}")
                continue

            # Find the segmentation folder inside the series folder (e.g. folder starting with "seg")
            segmentation_folder = next(
                (sf for sf in os.listdir(series_folder) if sf.startswith("seg") and os.path.isdir(os.path.join(series_folder, sf))),
                None
            )
            if not segmentation_folder:
                print(f"Warning: Segmentation folder not found in {series_folder}")
                continue

            seg_folder = os.path.join(series_folder, segmentation_folder)

            # List .npy segmentation files
            npy_files = [f for f in os.listdir(seg_folder) if f.endswith('.npy')]
            if not npy_files:
                print(f"Warning: No .npy files in {seg_folder}")
                continue
    
            npy_path = os.path.join(seg_folder, npy_files[0])

            try:
                tensor = np.load(npy_path)
                if len(tensor.shape) < 3:
                    print(f"Warning: tensor in {npy_path} has less than 3 dimensions: {tensor.shape}, skipping slice removal.")
                    continue
            except Exception as e:
                print(f"Failed to load {npy_path}: {e}")
                continue

            # Validate and remove only valid slice indices
            valid_indices = [i for i in indices if 0 <= i < tensor.shape[0]] #2
            invalid_indices = [i for i in indices if i < 0 or i >= tensor.shape[0]] #2
            total_invalid += len(invalid_indices)

            if valid_indices:
                # Sort in reverse to avoid reindexing issues
                valid_indices.sort(reverse=True)
                for idx in valid_indices:
                    tensor = np.delete(tensor, idx, axis=0)

                tensor = tensor.astype(np.int16)
                # Overwrite .npy file with modified tensor
                np.save(npy_path, tensor)
                total_removed += len(valid_indices)
                print(f"[{patient}/{series}] Removed {len(valid_indices)} black slice(s) from {os.path.basename(npy_path)}")

        # Build final message
        if total_removed > 0:
            removal_message += f"Number of removed segmentation slices: {total_removed}."
        else:
            removal_message += "No segmentation slices were removed."
    
        if total_invalid > 0:
            removal_message += f" (Warning: {total_invalid} slice indices were out of bounds.)"
    
        return removal_message
    

    def resize_seg(self):  
        """
        Loads and resizes all .npy segmentation files in the output directory to the target size defined in the protocol.
        Only overwrites the files if resizing is needed.
        
        Returns:
        str: Message indicating whether resizing was performed.
        """
        series_group = self.series_group_name
        seg_group_config = self.protocol.get(series_group, {}).get("segmentation", {}) #original self.protocol["segmentation"]
        output_directory_data = self.output_directory_data

        # Retrieve the target size from the protocol
        target_size = seg_group_config.get("segments_size", [256, 256])
        output_shape = (target_size[0], target_size[1])  # Set the desired output shape
        any_resized = False   # Flag to check if any slices were resized
        final_message_resize = ""

        for patient_id, series_id, npy_path, tensor, _ in load_patient_series_seg_data(series_group, output_directory_data):
            print(f"Processing {patient_id}/{series_id} - File: {os.path.basename(npy_path)}")

            # Check if resizing is necessary
            if tensor.shape[1] == output_shape[0] and tensor.shape[2] == output_shape[1]:
                #any_resized = False
                print(f"Already correct shape: {tensor.shape}. Skipping.")
                continue

            # Mark that resizing occurred
            any_resized = True
            
            # Reshape each slice independently along the third dimension
            num_slices = tensor.shape[0]
            resized_slices = []
            
            for slice_idx in range(num_slices):
                # Extract the 2D slice
                slice_2d = tensor[slice_idx, :, :]
                
                # Calculate zoom factors for resizing the slice
                zoom_factors = (output_shape[0] / slice_2d.shape[0], output_shape[1] / slice_2d.shape[1])
                
                # Resize the 2D slice using nearest-neighbor interpolation (order=0)
                resized_slice = scipy.ndimage.zoom(slice_2d, zoom_factors, order=0)
                
                # Ensure the resized slice is of integer type
                resized_slices.append(resized_slice) 
            
            # Stack the resized slices back into a 3D array
            resized_segmentation = np.stack(resized_slices, axis=0)

            resized_segmentation = resized_segmentation.astype(np.int16)
            # Overwrite the file with the resized data
            np.save(npy_path, resized_segmentation)
            any_resized = True
            print(f"Resized and saved: {npy_path} -> New shape: {resized_segmentation.shape}")

            # Manually clean up memory to avoid RAM buildup
            del resized_slices
            del resized_segmentation
            gc.collect()

        # Construct the final message based on whether any slices were resized
        if any_resized:
            final_message_resize += "Resizing completed."
        else:
            final_message_resize += "Resizing not needed: segmentations slices are already of the correct size."
    
        print(final_message_resize)

        return final_message_resize
    

    def check_seg_replicates_after_processing(self): #47 
        """
        Checks for identical 3D segmentation masks within a central slice range,
        excluding fully black slices.
    
        Returns:
        - str: Success message if no replicates are found.
        """
        
        local_config = self.local_config
        series_group = self.series_group_name
        output_directory_data = self.output_directory_data
        
        patient_id_map = load_patient_mappings(local_config, self.input_dir)
        series_id_map = load_series_mappings(self.images_dir, series_group)

        # Reverse maps to retrieve original keys from standardized IDs
        reversed_patient_id_map = {v: k for k, v in patient_id_map.items()}
        reversed_series_id_map = {v: k for k, v in series_id_map.items()}
        
        phase_number = 47  
        phase_data = self.mapping_file.get("check_phase", {}).get(str(phase_number), {})
        error_descriptions = phase_data.get("errors", {})
        phase_name = phase_data.get("name", f"Phase {phase_number}")  
        
        # Format check and report names dynamically
        check_name = phase_name.lower().replace(" ", "_")
        report_name = f"4.047.{check_name}_report"
        # Retrieve existing warnings
        check_file_path = os.path.join(self.output_directory_checkfile, "image_data_check_file.json")
        existing_warnings = retrieve_existing_warnings_image_data(check_file_path)
        
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        checked_pairs = set()
        grouped_series = []
        seg_index = {}
        total_replicates = 0
        error_counts = {"E4701": None}

        temp_dir = tempfile.mkdtemp()

        try:
            # Step 1: Save all segmentations temporarily, track by label
            for patient_id, series_id, _, tensor, _ in load_patient_series_seg_data(series_group, output_directory_data):
                
                orig_patient_id = reversed_patient_id_map.get(patient_id, patient_id)
                orig_series_id = reversed_series_id_map.get(series_id, series_id)
                
                # For labels used in reporting
                label = f"{orig_patient_id}/{os.path.basename(orig_series_id)}"
                temp_path = os.path.join(temp_dir, f"{label.replace('/', '_')}.npy")
                np.save(temp_path, tensor)
                seg_index[label] = temp_path

            labels = list(seg_index.keys())
    
            for i, label1 in enumerate(labels):
                vol1 = np.load(seg_index[label1])
            
                # Discard fully black slices
                non_black_slices_1 = [z for z in range(vol1.shape[0]) if not np.all(vol1[z, :, :] == 0)]
                if not non_black_slices_1:
                    continue  # Skip if fully black
            
                middle_idx_1 = non_black_slices_1[len(non_black_slices_1) // 2]
                start_1 = max(middle_idx_1 - 5, 0)
                end_1 = min(middle_idx_1 + 6, vol1.shape[0])
                cropped_vol1 = vol1[start_1:end_1, :, :]
            
                for j in range(i + 1, len(labels)):
                    label2 = labels[j]
                    pair = tuple(sorted([label1, label2]))
            
                    if pair in checked_pairs:
                        continue
                    checked_pairs.add(pair)
            
                    vol2 = np.load(seg_index[label2])
            
                    non_black_slices_2 = [z for z in range(vol2.shape[0]) if not np.all(vol2[z, :, :] == 0)]
                    if not non_black_slices_2:
                        continue  # Skip if fully black
            
                    middle_idx_2 = non_black_slices_2[len(non_black_slices_2) // 2]
                    start_2 = max(middle_idx_2 - 5, 0)
                    end_2 = min(middle_idx_2 + 6, vol2.shape[0])
                    cropped_vol2 = vol2[start_2:end_2, :, :]
            
                    if cropped_vol1.shape != cropped_vol2.shape:
                        continue  # Skip if shapes don't match
            
                    if np.array_equal(cropped_vol1, cropped_vol2):
                        # Identical: add to group
                        group_found = False
                        for group in grouped_series:
                            if label1 in group or label2 in group:
                                group.add(label1)
                                group.add(label2)
                                group_found = True
                                break
            
                        if not group_found:
                            grouped_series.append(set([label1, label2]))
    
            total_replicates = sum(len(group) - 1 for group in grouped_series)
            error_counts["E4701"] = total_replicates if total_replicates > 0 else None
        
            # Generate Check File
            generate_check_file_image_data(
                check_name=phase_name,
                phase_number=phase_number,
                series_group_name=series_group,
                num_slices_group=self.num_slices_group,
                num_patients_with_image_data=self.num_image_patients,
                num_series=self.total_series,
                num_tot_slices=self.total_slices,
                timestamp=current_datetime,
                error_counts=error_counts,
                warning_counts={},
                error_descriptions=error_descriptions,
                warning_descriptions={},
                output_dir=self.output_directory_checkfile,
                existing_warnings=existing_warnings
            )
        
            # Generate Report
            if grouped_series:
                os.makedirs(self.output_directory_report, exist_ok=True)
                report_path = os.path.join(self.output_directory_report, f"{report_name}.txt")
                host_report_path = os.path.join(self.host_output_directory_report, f"{report_name}.txt")

                with open(report_path, "w") as report_file:
                    report_file.write(f"Report generated on: {current_datetime}\n\n")
                    report_file.write(f"{phase_name} report:\n\n")
                    report_file.write(f"Error E4701: {error_descriptions.get('E4701', None)}\n")
                    report_file.write("- Details: The following patient/series groups have identical segmentations within the central slice range:\n")
        
                    for i, group in enumerate(grouped_series):
                        group_list = sorted(list(group))
                        group_report = " and ".join(group_list)
                        report_file.write(f"  - {group_report}.\n")
        
                    report_file.write(f"\nTotal Errors: {total_replicates}\n")
        
                raise ValueError(f"Identical segmentations detected. "
                                 f"See the detailed report at: {host_report_path}") #!! Path del container, deve essere quello locale.ok
    
            no_equal_seg_message = "No identical segmentations detected."
        
            return no_equal_seg_message
            
        finally:
            # Clean up temp directory
            shutil.rmtree(temp_dir)


    def validate_npy_dimension_match(self):
        """
        Validates that the final .npy volumes files in series folders within each patient folder in output_directory_data
        have matching dimensions with the corresponding .npy files in segmentation folders.
        Generates a report if mismatches are found and a check file regardless.
        """
        phase_number = 48  
        phase_data = self.mapping_file.get("check_phase", {}).get(str(phase_number), {})
        error_descriptions = phase_data.get("errors", {})
        phase_name = phase_data.get("name", f"Phase {phase_number}")  
        
        # Format check and report names dynamically
        check_name = phase_name.lower().replace(" ", "_") 
        report_name = f"4.048.{check_name}_report.txt"

        series_group = self.series_group_name 
        
        num_errors = 0
        has_error = False
        discrepancy_info = ""  
        npy_dim_success_message = ""
        error_counts = {"E4801": None}
    
        # Retrieve existing warnings
        check_file_path = os.path.join(self.output_directory_checkfile, "image_data_check_file.json")
        existing_warnings = retrieve_existing_warnings_image_data(check_file_path)
        
        # Get the current datetime for file naming and report generation
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Iterate over patient folders in output_directory_data
        patient_folders = [f for f in os.listdir(self.output_directory_data) if os.path.isdir(os.path.join(self.output_directory_data, f))]
        
        for patient_folder in patient_folders:
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
            
                    if os.path.isdir(series_folder_path): 
                        # Locate the .npy files in both the image and segmentation subfolders
                        image_npy_files = [f for f in os.listdir(series_folder_path) if f.endswith('.npy') and f.startswith("tensor")]
                        if not image_npy_files:
                            discrepancy_info += f"  - Missing image .npy file for patient {patient_folder}, series {series_folder}.\n"
                            num_errors += 1
                            has_error = True
                            continue
        
                        # Locate the segmentation folder dynamically
                        seg_folder = None
                        for subfolder in os.listdir(series_folder_path):
                            subfolder_path = os.path.join(series_folder_path, subfolder)
                            if os.path.isdir(subfolder_path):  # Check if it's a folder
                                seg_npy_files = [f for f in os.listdir(subfolder_path) if f.endswith('.npy')]
                                if seg_npy_files:  # Found segmentation .npy files
                                    seg_folder = subfolder_path
                                    break
                        
                        if not seg_folder:
                            discrepancy_info += f"  - Missing segmentation folder for patient {patient_folder}, series {series_folder}.\n"
                            num_errors += 1
                            has_error = True
                            continue
        
                        # Load the first image and segmentation .npy files for dimension matching
                        image_npy_path = os.path.join(series_folder_path, image_npy_files[0])
                        seg_npy_path = os.path.join(seg_folder, seg_npy_files[0]) 
                        
                        try:
                            image_data = np.load(image_npy_path)
                            seg_data = np.load(seg_npy_path)
                        except Exception as e:
                            discrepancy_info += f"  - Failed to load .npy file for patient {patient_folder}, series {series_folder}. Error: {str(e)}\n"
                            num_errors += 1
                            has_error = True
                            continue
                            
                        # Check if dimensions match
                        if image_data.shape != seg_data.shape:
                            discrepancy_info += (
                                f"  - Dimension mismatch for patient {patient_folder}, series {series_folder}. "
                                f"Image dimensions: {image_data.shape}, Segmentation dimensions: {seg_data.shape}\n"
                            )
                            num_errors += 1
                            has_error = True

        error_counts = {"E4801": num_errors if num_errors > 0 else None}
        # Use the helper function to generate the report and check file
        # Update the JSON check file
        generate_check_file_image_data(
            check_name=phase_name,
            phase_number=phase_number,
            series_group_name=series_group,
            num_slices_group=self.num_slices_group,
            num_patients_with_image_data=self.num_image_patients,
            num_series=self.total_series,
            num_tot_slices=self.total_slices,
            timestamp=current_datetime,
            error_counts=error_counts,
            warning_counts={},
            error_descriptions=error_descriptions,
            warning_descriptions={},
            output_dir=self.output_directory_checkfile,
            existing_warnings=existing_warnings
        )
    
        # Return appropriate message
        if has_error:
            report_content = ""
            report_filename = os.path.join(self.output_directory_report, report_name)
            host_report_filename = os.path.join(self.host_output_directory_report, report_name)
            report_content += f"Report generated on: {current_datetime}\n\n"
            report_content += f"{phase_name} report:\n\n"
            report_content += f"Error E4801: {error_descriptions.get('E4801', None)}\n"
            report_content += f"- Details:\n"
            report_content += discrepancy_info
            report_content += f"\nTotal Errors: {num_errors}"
            with open(report_filename, "w") as report_file:
                report_file.write(report_content)
            raise Exception(
                 f"Dimension mismatch detected between image and segmentation .npy files. "
                 f"See the detailed report at: {host_report_filename}" #!! Path del container, deve essere quello locale.ok
            )
        else:
            npy_dim_success_message += "Dimension validation successful: all corresponding .npy volumes and segmentation files match in dimensions."
            return npy_dim_success_message
        

    def generate_NiftiSegReorientPreprocess_final_report(self, reorient_message, single_to_multilabel_message, removal_message, final_message_resize, no_equal_seg_message, npy_dim_success_message):
        """
        Generate a comprehensive final report for NiftiSegReorientTransform process.
    
        Parameters:
        - reorient_message: Message about reorientation status.
        - single_to_multilabel_message: Message about single-to-multilabel conversion.
        - removal_message: Message about slice removal.
        - final_message_resize: Message about resizing status.
        - no_equal_seg_message: Message about replicates absence.
        - npy_dim_success_message: Message about NPY dimensions check. 
        """
        
        # Get the current datetime
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
        
        # Define the report filename and path
        report_filename = "4.NiftiSegReorientPreprocess_final_report.txt"
        report_filepath = os.path.join(self.output_directory_report, report_filename)
        
        # Initialize report content with ordered titles and messages
        report_content = []
        
        # Add report header
        report_content.append(f"Report generated on: {formatted_datetime}\n")
        report_content.append("NIfTI segmentation reorientation and transformation report:\n")
        
        # Segmentation reorientation status
        report_content.append("Segmentation reorientation check:")
        if reorient_message:
            report_content.append("- Segmentations reorientation occurred.")
        report_content.append("- Reference segmentation orientation: LPS.\n")
        
        # Single to multi-label conversion status
        report_content.append("Multi-label conversion status:")
        report_content.append("- " + single_to_multilabel_message + "\n")

        # Segmentation slice removal status
        report_content.append("Segmentation slice removal status:")
        if removal_message:
            report_content.append("- " + removal_message + "\n")

        # Add resizing status
        report_content.append("Segmentation resizing status:")
        if final_message_resize:
            report_content.append("- " + final_message_resize + "\n")

        # Replicate check status
        report_content.append("Replicate segmentations check:")
        report_content.append("- " + no_equal_seg_message + "\n")
        
        # Export to NPY format status
        report_content.append("Segmentation tensor save status:")
        final_message_npy = f"Tensors saved in .npy format into segmentation folders inside series folders within patient folders in {self.host_output_directory_data}." #!! Path del container, deve essere quello locale
        report_content.append("- " + final_message_npy + "\n")

        # NPY dimension check status
        report_content.append("NPY dimensions check status:")
        if npy_dim_success_message:
            report_content.append("- " + npy_dim_success_message + "\n")
            
        # Final report summary
        report_content.append("Summary:")
        report_content.append("All checks and transformations completed.")
        
        # Write the report to the specified file
        with open(report_filepath, 'w') as report_file:
            report_file.write("\n".join(report_content))
        
        # Print a message indicating that the final report has been generated
        print(f"NiftiSegReorientPreprocess final report saved to {report_filepath}")
        

def run_niftiseg_reorient_preprocessor(protocol, local_config, mapping_file, series_group_name):
    # Define the input directory path with 
    input_dir = os.getenv("INPUT_DIR")

    # Clear log at the start of validation
    clear_log_file(input_dir, LOG_FILENAME)

    # Load the input validation state to extract num_image_patients
    input_state_file = os.path.join(input_dir, "input_validation_state.json")
    image_state_file = os.path.join(input_dir, "image_validation_state.json")

    try:
        with open(input_state_file, "r") as f:
            input_state = json.load(f)
        num_image_patients = input_state.get("num_patients_with_image_data", 0)
    except Exception as e:
        raise RuntimeError(f"Failed to load input state file '{input_state_file}': {e}")
    
    try:
        with open(image_state_file, "r") as f:
            image_state = json.load(f)
        num_slices_group = image_state.get("num_slices_group", 0)
        total_series = image_state.get("total_series", 0)
        total_slices = image_state.get("total_slices", 0)
    except Exception as e:
        raise RuntimeError(f"Failed to load image state file '{image_state_file}': {e}")
    
    niftiseg_preprocessor = NiftiSegReorientPreprocessor(protocol, local_config, mapping_file, num_image_patients, num_slices_group, total_series, total_slices, series_group_name)

    series_progress_state =  load_state(niftiseg_preprocessor.series_progress_file)

    # Access or create the per-series progress dictionary
    series_state = series_progress_state.setdefault(series_group_name, {})
    last_phase_done = series_state.get("last_successful_phase", 0)

    if last_phase_done < 48:
    
        print("Running check_and_reorient_niftiseg function...")
        try:
            reorient_message = niftiseg_preprocessor.check_and_reorient_niftiseg()
        except Exception as e:
            log_error(input_dir, "check_and_reorient_niftiseg", e, LOG_FILENAME)
            print(f"An unexpected error occurred during check_and_reorient_niftiseg. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
            raise 

        print("Running single_to_multi_label_conversion function...")
        try:
            single_to_multilabel_message = niftiseg_preprocessor.single_to_multi_label_conversion(image_state.get("multiple_segmentation_flag"))
        except Exception as e:
            log_error(input_dir, "single_to_multi_label_conversion", e, LOG_FILENAME)
            print(f"An unexpected error occurred during single_to_multi_label_conversion. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
            raise 

        print("Running discard_black_seg_slices function...")
        try:
            removal_message = niftiseg_preprocessor.discard_black_seg_slices()
        except Exception as e:
            log_error(input_dir, "discard_black_seg_slices", e, LOG_FILENAME)
            print(f"An unexpected error occurred during discard_black_seg_slices. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
            raise 

        print("Running resize_seg function...")
        try:
            final_message_resize = niftiseg_preprocessor.resize_seg()
        except Exception as e:
            log_error(input_dir, "resize_seg", e, LOG_FILENAME)
            print(f"An unexpected error occurred during resize_seg. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
            raise 

        print("Running check_seg_replicates_after_processing function...") # check phase 47
        try:
            no_equal_seg_message = niftiseg_preprocessor.check_seg_replicates_after_processing()

            series_state["last_successful_phase"] = 47
            series_progress_state[series_group_name] = series_state
            save_state(series_progress_state, niftiseg_preprocessor.series_progress_file)
        except Exception as e:
            log_error(input_dir, "check_seg_replicates_after_processing", e, LOG_FILENAME)
            print(f"An unexpected error occurred during check_seg_replicates_after_processing. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
            raise 

        print("Running validate_npy_dimension_match function...") # check phase 48
        try:
            npy_dim_success_message = niftiseg_preprocessor.validate_npy_dimension_match()

            # Phase 48: Final step of niftiseg preprocessing; all prior validation assumed complete.
            series_state["last_successful_phase"] = 48
            series_progress_state[series_group_name] = series_state
            save_state(series_progress_state, niftiseg_preprocessor.series_progress_file)
        except Exception as e:
            log_error(input_dir, "validate_npy_dimension_match", e, LOG_FILENAME)
            print(f"An unexpected error occurred during validate_npy_dimension_match. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
            raise 

        print("Running generate_NiftiSegReorientPreprocess_final_report function...")
        try:
            niftiseg_preprocessor.generate_NiftiSegReorientPreprocess_final_report(reorient_message, single_to_multilabel_message, removal_message, final_message_resize, no_equal_seg_message, npy_dim_success_message)
        except Exception as e:
            log_error(input_dir, "generate_NiftiSegReorientPreprocess_final_report", e, LOG_FILENAME)
            print(f"An unexpected error occurred during generate_NiftiSegReorientPreprocess_final_report. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
            raise 