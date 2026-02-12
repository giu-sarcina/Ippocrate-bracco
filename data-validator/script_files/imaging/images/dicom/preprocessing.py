import pandas as pd
import numpy as np
import os 
import csv
import pydicom
from datetime import datetime
import glob
import torch
from PIL import Image
import shutil
import json
from kneed import KneeLocator
from skimage.transform import resize
import SimpleITK as sitk
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
    read_csv_file,
    generate_check_file_image_data,
    retrieve_existing_warnings_image_data,
    extract_metadata_from_check_file,
    load_patient_series_data,
    load_patient_feature_vectors,
    load_patient_mappings,
    load_series_mappings,
    get_orientation,
    ct_convert_to_hu_and_clip,
    image_wise_clipping,
    mr_db_wise_clipping,
    intensity_scaling,
    standardization_mean0_std1,
    apply_clahe,
    count_series_group_slices_from_npy,
    count_tot_num_series_per_dataset,
    count_tot_num_slices_per_dataset,
    compute_adjacent_slice_ncc_scores
)

LOG_FILENAME = "dicom_preprocessing_error.log"

class DicomReorientPreprocessor:
    
    def __init__(self, protocol, local_config, mapping_file, num_image_patients, series_group_name):
        self.protocol = protocol
        self.local_config = local_config
        self.mapping_file = mapping_file
        self.num_image_patients = num_image_patients
        self.series_group_name = series_group_name
        self.input_dir = os.getenv("INPUT_DIR")
        self.images_dir = os.path.join(self.input_dir, "IMAGES")
        self.host_input_dir = os.getenv("HOST_BASE_DATA_DIR")
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


    def sort_dicom_slices(self): #3D only
        
        local_config = self.local_config
        series_group = self.series_group_name
        
        patient_id_map = load_patient_mappings(local_config, self.input_dir)
        series_id_map = load_series_mappings(self.images_dir, series_group)
         
        any_reordering_occurred = False  
        slice_sorting_message = ""

        series_group = self.series_group_name
        series_group_mapping = self.local_config["Local config"]["radiological data"]["series_mapping"]
        series_group_subfolder = series_group_mapping.get(series_group) #input name of the series

        patient_folders = [f for f in os.listdir(self.images_dir) if os.path.isdir(os.path.join(self.images_dir, f)) and f != "Reports"]
        
        # Iterate through patient folders in the input folder
        for orig_patient_id in patient_folders:
            patient_folder_path = os.path.join(self.images_dir, orig_patient_id)
            if os.path.isdir(patient_folder_path):
                
                # Map to new patient ID
                new_patient_id = patient_id_map.get(orig_patient_id)
                
                series_folder_name = os.path.normpath(series_group_subfolder).split(os.sep)[-1].strip()
                series_folder_path = os.path.join(patient_folder_path, series_folder_name)
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
                    
                    dicom_files = []

                    # Iterate through DICOM files in the series folder
                    for filename in os.listdir(series_folder_path):
                        if filename.endswith(".dcm"):
                            file_path = os.path.join(series_folder_path, filename)
                            ds = pydicom.dcmread(file_path)
                            slice_location = ds.ImagePositionPatient[2]
                            dicom_files.append((filename, ds, slice_location))
    
                    if dicom_files:
                        # Check if slices are already sorted based on slice location (z-coordinate)
                        slice_locations = [item[2] for item in dicom_files]
                        if slice_locations == sorted(slice_locations):
                            print(f"Slices in {series_folder_name} are already in increasing order of z-coordinate.")
                            sorted_dicom_files = dicom_files  # No reordering needed
                            reordering_flag = False
                        else:
                            # Sort the DICOM files based on slice location in increasing order
                            sorted_dicom_files = sorted(dicom_files, key=lambda x: x[2])
                            any_reordering_occurred = True # Set flag to indicate reordering
                            reordering_flag = True
                            print(f"Slices in {series_folder_name} have been reordered based on z-coordinate.")

                        slices_np = []
                        slice_metadata_list = []

                        def to_list_if_needed(val):
                            if isinstance(val, (pydicom.multival.MultiValue, np.ndarray)):
                                return list(val)
                            return val

                        for filename, ds, _ in sorted_dicom_files:
                            try:
                                #image = sitk.ReadImage(os.path.join(series_folder_path, filename))
                                #image_array = sitk.GetArrayFromImage(image)[0].astype(np.int16) #!
                                image_array = ds.pixel_array.astype(np.int16)
                                slices_np.append(image_array)
                            except Exception as e:
                                raise ValueError(f"Error reading image {filename} with SimpleITK: {e}")
    
                            slice_metadata_list.append({
                                "ImageOrientationPatient": to_list_if_needed(ds.get("ImageOrientationPatient")),
                                "ImagePositionPatient": to_list_if_needed(ds.get("ImagePositionPatient")),
                                "RescaleSlope": float(ds.get("RescaleSlope", 1.0)),
                                "RescaleIntercept": float(ds.get("RescaleIntercept", -1024.0)), #-1024.0 for CT
                                "PhotometricInterpretation": ds.get("PhotometricInterpretation", None),
                                "PixelSpacing": to_list_if_needed(ds.get("PixelSpacing")),
                                "SpacingBetweenSlices": float(ds.get("SpacingBetweenSlices", 0.0)) if ds.get("SpacingBetweenSlices") is not None else None
                            })

                        # Prepare output paths (anonymized)
                        patient_out_path = os.path.join(self.output_directory_data, new_patient_id)
                        series_out_path = os.path.join(patient_out_path, new_series_name)
                        os.makedirs(series_out_path, exist_ok=True)
                
                        # Save .npy
                        npy_filename = f"tensor_{new_series_name2}.npy"
                        np.save(os.path.join(series_out_path, npy_filename), np.stack(slices_np, axis=0))
                
                        # Save metadata JSON
                        json_filename = f"metadata_{new_series_name2}.json"
                        with open(os.path.join(series_out_path, json_filename), "w") as f:
                            json.dump({
                                "new_patient_id": new_patient_id,
                                "new_series_name": new_series_name,
                                "num_slices": len(slices_np),
                                "reordering_flag": reordering_flag,
                                "slices": slice_metadata_list
                            }, f, indent=2)

                        print(f"Saved tensor and metadata for {new_patient_id}/{new_series_name}")

        if any_reordering_occurred:
            slice_sorting_message = "Slices have been reordered in increasing z order (from inferior to superior)."
        else:
            slice_sorting_message = "All slices were already in correct z order."
          
        return slice_sorting_message, any_reordering_occurred
    

    def check_and_reorient_slices(self): #3D only
        """
        Loads saved tensors and metadata, checks and reorients non-LP slices using SimpleITK if needed.
        Overwrites the existing .npy and .json files with updated data.
        
        Returns:
            reoriented_flag (bool): True if any reorientation occurred, False otherwise.
        """
        series_group = self.series_group_name
        output_directory_data = self.output_directory_data
        
        non_lp_slice_count = 0  # Counter for non-LP slices
        reoriented_flag = False  # Flag to indicate if any slices were reoriented
        reoriented_info = []
    
        for patient_id, series_id, metadata_file, tensor_file, metadata, tensor in load_patient_series_data(series_group, output_directory_data):
        
            reoriented_slices = []
            updated_metadata_slices = []
            reoriented_this_series = False
                
            for i, slice_meta in enumerate(metadata["slices"]):
                image_orientation = slice_meta.get("ImageOrientationPatient")
                x_orientation, y_orientation = get_orientation(image_orientation)
                
                #print(f"  Patient: {patient_id}, Series: {series_id}, Slice: {i}, Orientation: {x_orientation}{y_orientation}")
                reoriented_info.append((patient_id, series_id, i, x_orientation, y_orientation))

                pixel_array = tensor[i]
                needs_reorientation = (x_orientation != 'L' or y_orientation != 'P')

                # Check if the slice is not LP (Left-Posterior)
                if needs_reorientation:
                    # Increment non-LP slice count 
                    reoriented_flag = True
                    reoriented_this_series = True
                    non_lp_slice_count += 1
                    
                    # Convert to SimpleITK image
                    slice_image = sitk.GetImageFromArray(np.expand_dims(pixel_array, 0))
                    
                    # Set spacing, origin, direction
                    spacing = slice_meta.get("PixelSpacing", [1.0, 1.0])
                    slice_spacing = slice_meta.get("SpacingBetweenSlices", 1.0)
                    spacing.append(slice_spacing)
                    slice_image.SetSpacing(spacing)

                    if "ImagePositionPatient" in slice_meta:
                        slice_image.SetOrigin(slice_meta["ImagePositionPatient"])

                    if "ImageOrientationPatient" in slice_meta:
                        iop = slice_meta["ImageOrientationPatient"]
                        v1 = np.array(iop[:3])
                        v2 = np.array(iop[3:])
                        v3 = np.cross(v1, v2)
                        direction = np.array([v1, v2, v3]).T.flatten()
                        slice_image.SetDirection(direction)
                        
                    # Step 4: Set the pixel type (short, 3D image)
                    slice_image = sitk.Cast(slice_image, sitk.sitkInt16)
                    # Reorient the image to the desired coordinate system
                    reoriented_image = sitk.DICOMOrient(slice_image, 'LPS')                       
                    # Convert reoriented image back to numpy array and store it
                    reoriented_pixel_array = sitk.GetArrayFromImage(reoriented_image)[0]                       
                    
                    reoriented_slices.append(reoriented_pixel_array)
                    
                    # Optional: update orientation info in metadata?
                    slice_meta["Reoriented"] = True
                else:
                    reoriented_slices.append(pixel_array)

                updated_metadata_slices.append(slice_meta)

            np.save(tensor_file, np.stack(reoriented_slices, axis=0))

            # Update metadata and reordering flag
            metadata["slices"] = updated_metadata_slices
            metadata["reoriented_flag"] = reoriented_this_series

            # Overwrite JSON file
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2) 

            if reoriented_this_series:
                print(f"Reoriented slices for Patient {patient_id}, Series {series_id}")

        # After processing all series
        if reoriented_flag:
            print(f"Total non-LP slices reoriented: {non_lp_slice_count}")
            print("Detailed reorientation log:")
            for patient, series, idx, x_ori, y_ori in reoriented_info:
                print(f"  Patient: {patient}, Series: {series}, Slice: {idx}, Orientation: {x_ori}{y_ori}")
        else:
            print("All slices were already in LP orientation.")
    
        return reoriented_flag
    
    
    def extract_2Ddicom_tensors_and_metadata(self):  #just 2D
        """
        Extracts 2D DICOM data, saves tensors as .npy and metadata as .json
        in output_directory_data, using patient and series specific renaming.
        """
        local_config = self.local_config
        series_group = self.series_group_name
    
        patient_id_map = load_patient_mappings(local_config, self.input_dir)
        series_id_map = load_series_mappings(self.images_dir, series_group)
        series_group_mapping = local_config["Local config"]["radiological data"]["series_mapping"]
        series_group_subfolder = series_group_mapping.get(series_group)
    
        patient_folders = [f for f in os.listdir(self.images_dir) if os.path.isdir(os.path.join(self.images_dir, f)) and f != "Reports"]
    
        for orig_patient_id in patient_folders:
            patient_folder_path = os.path.join(self.images_dir, orig_patient_id)
            host_patient_folder_path = os.path.join(self.host_input_dir, orig_patient_id)

            new_patient_id = patient_id_map.get(orig_patient_id)
    
            series_folder_name = os.path.normpath(series_group_subfolder).split(os.sep)[-1].strip()
            series_folder_path = os.path.join(patient_folder_path, series_folder_name)
            host_series_folder_path = os.path.join(host_patient_folder_path, series_folder_name)

            if not os.path.isdir(series_folder_path):
                continue

            full_series_key = os.path.join(orig_patient_id, series_folder_name)
            # If mappings use Windows-style separators, normalize only for lookup
            if "\\" in list(series_id_map.keys())[0]:  # Detect if mapping keys use '\'
                full_series_key = full_series_key.replace("/", "\\")
                    
            new_series_name = series_id_map.get(full_series_key)
    
            if not (new_series_name and "_" in new_series_name):
                continue
    
            parts = new_series_name.split("_")
            new_series_name2 = f"{parts[1]}_{parts[2]}_{parts[0]}" if len(parts) >= 3 else new_series_name
    
            patient_out_path = os.path.join(self.output_directory_data, new_patient_id)
            series_out_path = os.path.join(patient_out_path, new_series_name)
            os.makedirs(series_out_path, exist_ok=True)
    
            for file in os.listdir(series_folder_path):
                if not file.lower().endswith(".dcm"):
                    continue
    
                file_path = os.path.join(series_folder_path, file)
                host_file_path = os.path.join(host_series_folder_path, file)
                try:
                    dicom_data = pydicom.dcmread(file_path)
                    array = np.expand_dims(dicom_data.pixel_array, axis=0)
    
                    npy_filename = f"tensor_{new_series_name2}.npy"
                    json_filename = f"metadata_{new_series_name2}.json"
    
                    np.save(os.path.join(series_out_path, npy_filename), array.astype(np.float64))
 
                    # Extract and save metadata
                    photometric_interpretation = dicom_data.get("PhotometricInterpretation", None)

                    with open(os.path.join(series_out_path, json_filename), "w") as f:
                        json.dump({
                            "new_patient_id": new_patient_id,
                            "new_series_name": new_series_name,
                            "slices": [
                                {
                                    "PhotometricInterpretation": photometric_interpretation
                                }
                            ]
                        }, f, indent=2)

                    print(f"Saved tensor and metadata for {new_patient_id}/{new_series_name}/{file}")
    
                except Exception as e:
                    print(f"Failed to process {host_file_path}: {e}") #!! Path del container, deve essere quello locale.OK
    

    def convert_monochrome1_to_monochrome2(self): # 2D and 3D # AGG: se uso SimpleITK per leggere l'array e salvarlo 
        """
        Loads saved tensors and metadata, checks for MONOCHROME1 photometric interpretation,
        converts to MONOCHROME2 by inverting pixel values if needed, and overwrites the .npy and .json files.
    
        Returns:
            conversion_info (str): Summary of the conversions performed.
        """
        output_directory_data = self.output_directory_data
        series_group = self.series_group_name  
        image_group_config = self.protocol.get(series_group, {}).get("image", {}) #original self.protocol["image"]

        # Image and segmentation format validations
        image_type = image_group_config["type"]["selected"]
        
        converted_count = 0
        converted_files = []
    
        for patient_id, series_id, metadata_file, tensor_file, metadata, tensor in load_patient_series_data(series_group, output_directory_data):   
            modified = False
            print(tensor_file)

            for i, slice_meta in enumerate(metadata["slices"]):
                photometric_interpretation = slice_meta.get("PhotometricInterpretation", "MONOCHROME2")
                if photometric_interpretation == "MONOCHROME1":
                    #print(f"Converting MONOCHROME1 to MONOCHROME2 for patient {patient_id}, slice {i}")
                    before = tensor[i].copy()
                    tensor[i] = np.max(before) - before
                    #print(f"Before: min={before.min()}, max={before.max()} | After: min={tensor[i].min()}, max={tensor[i].max()}")

                    # Update metadata
                    slice_meta["PhotometricInterpretation"] = "MONOCHROME2"
                    slice_meta["ConvertedFromMonochrome1"] = True

                    converted_count += 1
                    modified = True
                    if image_type in ["CR", "DX", "RX"]:
                        converted_files.append(f"Patient: {patient_id}, Series: {series_id}")
                    else:
                        converted_files.append(f"Patient: {patient_id}, Series: {series_id}, Slice: {i+1}")

            if modified:
                # Save the updated tensor
                np.save(tensor_file, tensor)

                # Save updated metadata
                with open(metadata_file, "w") as f:
                    json.dump(metadata, f, indent=2)
    
        # Summary
        if converted_count > 0:
            if image_type in ["CR", "DX", "RX"]:
                word_singular = "image"
                word_plural = "images"
            else:
                word_singular = "slice"
                word_plural = "slices"
        
            slice_word = word_singular if converted_count == 1 else word_plural
        
            conversion_info = (
                f"MONOCHROME1 -> MONOCHROME2 conversion applied to {converted_count} {slice_word}:\n" +
                "\n".join(f"   - {entry}" for entry in converted_files)
            )
        else:
            conversion_info = f"No MONOCHROME1 {'images' if image_type in ['CR', 'DX'] else 'slices'} found. No conversion needed."
    
        return conversion_info
    

    def process_array(self):  #3D e 2D
        """
        Loads tensors and metadata, applies clipping and normalization, and overwrites them.
        Returns:
            final_processing_summary (str): Summary of the processing steps.
        """
        local_config = self.local_config
        series_group = self.series_group_name
        
        patient_id_map = load_patient_mappings(local_config, self.input_dir)
        series_id_map = load_series_mappings(self.images_dir, series_group)

        # Reverse maps to retrieve original keys from standardized IDs
        reversed_patient_id_map = {v: k for k, v in patient_id_map.items()}
        reversed_series_id_map = {v: k for k, v in series_id_map.items()}

        output_directory_data = self.output_directory_data
        series_group = self.series_group_name
        image_group_config = self.protocol.get(series_group, {}).get("image", {}) 
        
        image_type = image_group_config["type"]["selected"]
        clipping_option = image_group_config["clipping"]["selected"]
        normalization_option = image_group_config["normalization"]["selected"]
        pixel_scaling = image_group_config["pixel scaling"]["selected"]
        contrast_optimization = image_group_config.get("contrast_optimization_DX/CR", {}).get("selected", "none")

        processing_summary = []

        # Track whether clipping or normalization was applied
        clipping_applied = False
        normalization_applied = False
        clahe_applied = False

        # List to store discarded slices info
        discarded_slices = []
        
        for patient_id, series_id, _, tensor_file, metadata, tensor in load_patient_series_data(series_group, output_directory_data):
            
            orig_patient_id = reversed_patient_id_map.get(patient_id, patient_id)
            orig_series_id = reversed_series_id_map.get(series_id, series_id)
            orig_series_id = os.path.basename(orig_series_id)
            
            updated_tensor = []

            # Pre-calculate DB-wise knee point for MRI if needed
            p_knee_val = None          
            if image_type == "MRI" and clipping_option == "db-wise (only MRI)":
                all_pixels = np.concatenate([np.ravel(np.squeeze(t)) for t in tensor]).astype(float)
                
                percentiles = np.arange(95, 100, 0.01)
                percentile_values = np.percentile(all_pixels, percentiles)
                percentiles_desc_order = percentiles[::-1]
                percentile_values_desc_order = percentile_values[::-1]
                kl = KneeLocator(percentiles_desc_order, percentile_values_desc_order, curve="convex")
                kl.plot_knee()
                knee_value = kl.knee
                p_knee_val = np.percentile(all_pixels, knee_value)
                #print(p_knee_val)
                print("X-axis value corresponding to the knee point:", knee_value)

            for i, (slice_meta, image_array) in enumerate(zip(metadata["slices"], tensor), start=1):
                image_array = np.squeeze(image_array)
                rescale_slope = slice_meta.get("RescaleSlope")
                rescale_intercept = slice_meta.get("RescaleIntercept")         
                normalized_image = None # Initialize to ensure it's always defined

                if image_type == "CT":
                    # Apply clipping
                    if clipping_option == "image-wise":
                        ct_clipped_image = ct_convert_to_hu_and_clip(image_array, rescale_slope, rescale_intercept, image_group_config)
                        clipping_applied = True
                    elif clipping_option == "db-wise (only MRI)":
                        raise ValueError("Selected clipping method is not valid for CT images!")
                    elif clipping_option == "none":
                        ct_clipped_image = image_array  # No clipping, use the original image

                    # Apply normalization
                    if normalization_option == "intensity_scaling":
                        normalized_image = intensity_scaling(ct_clipped_image, image_group_config)
                        normalization_applied = True
                    elif normalization_option == "standardization_mean0_std1":
                        normalized_image = standardization_mean0_std1(ct_clipped_image)
                        normalization_applied = True
                    elif normalization_option == "none":
                        normalized_image = ct_clipped_image  # No normalization, use the clipped image

                elif image_type == "MRI":
                    # Apply clipping
                    if clipping_option == "image-wise":
                        mr_clipped_image = image_wise_clipping(image_array, image_group_config)
                        clipping_applied = True
                    elif clipping_option == "db-wise (only MRI)":
                        mr_clipped_image = mr_db_wise_clipping(image_array, p_knee_val)
                        clipping_applied = True
                    elif clipping_option == "none":
                        mr_clipped_image = image_array  # No clipping, use the original image
                        
                    # Apply normalization
                    if normalization_option == "intensity_scaling":
                        normalized_image = intensity_scaling(mr_clipped_image, image_group_config)
                        normalization_applied = True
                    elif normalization_option == "standardization_mean0_std1":
                        normalized_image = standardization_mean0_std1(mr_clipped_image)
                        normalization_applied = True
                    elif normalization_option == "none":
                        normalized_image = mr_clipped_image  # No normalization, use the clipped image

                elif image_type in ["CR", "DX", "RX"]:
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
                
                # Check if the image is all-black or degenerate
                if normalized_image is None or (np.std(normalized_image) == 0):
                    print(f"Discarded slice: {patient_id}/{series_id} - index {i}")
                    discarded_slices.append([orig_patient_id, orig_series_id, i])
                    continue
                        
                # Append the processed image to the list of slices for this patient
                updated_tensor.append(normalized_image.astype(np.float64))

            # Save updated tensor
            np.save(tensor_file, np.stack(updated_tensor, axis=0))
    
            print(f"Processed patient: {patient_id}, series: {series_id}")


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

        # Only handle discarded slices info if image_type is NOT CR or DX
        if image_type not in ["CR", "DX", "RX"]:
            if discarded_slices:
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
                    print(f"{len(new_rows)} new discarded slices written to {csv_path}") #!! Path del container.OK
                else:
                    print("No new discarded slices to add; CSV remains unchanged.")
            else:
                print("No discarded slices; CSV file will not be created/updated.")

            # Add number of discarded slices to the processing summary
            num_discarded_slices = len(discarded_slices)
            processing_summary.append(f"- Number of discarded slices (all black): {num_discarded_slices}.")

            # If any slices were discarded, add a message about the CSV file
            if num_discarded_slices > 0:
                processing_summary.append(f"- discarded_slices.csv file has been created/updated in the directory: {self.host_images_dir}") #!! Path del container, deve essere quello locale.OK

        # Join summary statements and return it along with processed slices
        final_processing_summary = "\n".join(processing_summary)
        
        return final_processing_summary
    

    def resize_array(self): #2D e 3D
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


    def save_labels(self):  #2D e 3D
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
                os.path.join(orig_patient_id, orig_series_id)
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
    

    def convert_npy_to_image(self): # 2D e 3D 
        """Convert npy files to images."""

        series_group = self.series_group_name
        image_group_config = self.protocol.get(series_group, {}).get("image", {}) #original self.protocol["image"]
        
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
                        if npy_file.startswith("tensor") and npy_file.endswith(".npy"): #!!NB
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
                                print(f"Saved {output_file_name} in {image_folder_path}") #!! Path del container.OK
                                    
        message_from_npy_to_image += f"Images in {output_format} format have been generated from npy tensors and saved into series folders within patient folders in {self.host_output_directory_data}." #!! Path del container, deve essere quello locale.OK               
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


    def analyze_series_alignment(self): #3D only
        
        phase_number = 37  
        phase_data = self.mapping_file.get("check_phase", {}).get(str(phase_number), {})
        warning_descriptions = phase_data.get("warnings", {})
        phase_name = phase_data.get("name", f"Phase {phase_number}")  
        
        # Format check and report names dynamically
        check_name = phase_name.lower().replace(" ", "_") 
        report_name = f"2.037.{check_name}_report"

        series_group = self.series_group_name
        image_group_config = self.protocol.get(series_group, {}).get("image", {}) #original self.protocol["image"]

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


    def create_and_save_MIP_image(self): #3D only
        
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
                        axis = 0  # Set this to the desired axis (0 for axial, 1 for coronal, 2 for sagittal)
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
            f"in {self.host_output_directory_data}. " #!! Path del container, deve essere quello locale.OK
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
                                print(f"Saved feature vector to {feature_file_path}") #!! Path del container
                                
                            except Exception as e:
                                print(f"Failed to process image {img_path}: {e}")#!! Path del container
    
        # Convert list of features to a NumPy array
        features = np.array(features)
        
        # Compute mu and sigma
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
    
        message_extract_feat = "Computed mean and covariance of MIP image features, and saved per-image feature vectors."
    
        return mu, sigma, message_extract_feat
    

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
                            print(f"Saved features to {feature_path}") #!! Path del container
    
                        except Exception as e:
                            print(f"Failed to process image {img_path}: {e}") #!! Path del container
    
        features = np.array(features)
    
        # Compute statistics
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
    
        message_extract_feat = "Computed mean and covariance of 2D image features using PyTorch, and saved per-image feature vectors."

        return mu, sigma, message_extract_feat
    

    def generate_DicomPreprocessing2D_final_report(self, conversion_info, final_processing_summary, final_message_resize, replicate_slice_message, warnings_replicates, final_message_npy, message_from_npy_to_image, message_extract_feat):
        # Get the current datetime
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
        
        # Define the report filename and path
        report_filename = "2.DicomPreprocessing2D_final_report.txt"
        report_filepath = os.path.join(self.output_directory_report, report_filename)

        # Initialize report content with ordered titles and messages
        report_content = []

        report_content.append(f"Report generated on: {formatted_datetime}\n")
        
        report_content.append("2D DICOM preprocessing final report:\n")

        report_content.append("1. MONOCHROME1 to MONOCHROME2 conversion summary:")
        report_content.append("- " + conversion_info + "\n")

        report_content.append("2. Processing summary:")
        report_content.append(final_processing_summary + "\n")

        report_content.append("3. Resizing status:")
        report_content.append("- " + final_message_resize + "\n")

        report_content.append("4. Replicate image check:")
        report_content.append("- " + replicate_slice_message + "\n")

        report_content.append("5. Tensor save status:")
        report_content.append("- " + final_message_npy + "\n")
        
        report_content.append("6. Image generation status:")
        report_content.append("- " + message_from_npy_to_image + "\n")

        # Add feature extraction status
        report_content.append("7. Feature extraction status:")
        report_content.append("- " + message_extract_feat + "\n")

        # Add warning summary
        report_content.append(f"Total warnings detected: {warnings_replicates}.")
        
        # Write the report to the specified file
        with open(report_filepath, 'w') as report_file:
            report_file.write("\n".join(report_content))
            
        print(f"Final report saved to {report_filepath}") #!! Path del container


    def generate_DicomReorientPreprocess3D_final_report(self, axial_success_message, reordering_flag, reoriented_flag, conversion_info, final_processing_summary, final_message_resize, replicate_slice_message, warnings_replicates, final_message_npy_and_labels, message_from_npy_to_image, message_series_alignment, total_misaligned_slices, mip_message, message_extract_feat):
        # Get the current datetime
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
        
        # Define the report filename and path
        report_filename = "2.3DDicomReorientPreprocess_final_report.txt"
        report_filepath = os.path.join(self.output_directory_report, report_filename)

        # Initialize report content with ordered titles and messages
        report_content = []

        report_content.append(f"Report generated on: {formatted_datetime}\n")
        
        report_content.append("3D DICOM reorientation and preprocessing final report:\n")
        
        report_content.append("1. Volume reorientation check:")
        # Volume reorientation check
        if reordering_flag or reoriented_flag:
            report_content.append("- Volumes reorientation occurred.")
        report_content.append("- Reference volume orientation: LPS.\n")

        # Add all messages with concise titles
        report_content.append("2. Anatomical plane:")
        report_content.append("- " + axial_success_message + "\n")

        report_content.append("3. MONOCHROME1 to MONOCHROME2 conversion summary:")
        report_content.append("- " + conversion_info + "\n")

        report_content.append("4. Processing summary:")
        report_content.append(final_processing_summary + "\n")

        report_content.append("5. Resizing status:")
        report_content.append("- " + final_message_resize + "\n")

        report_content.append("6. Replicate image check:")
        report_content.append("- " + replicate_slice_message + "\n")

        report_content.append("7. Tensors and labels save status:")
        report_content.append("- " + final_message_npy_and_labels + "\n")
        
        report_content.append("8. Image generation status:")
        report_content.append("- " + message_from_npy_to_image + "\n")
        
        report_content.append("9. Slices alignment summary:")
        report_content.append("- " + message_series_alignment + "\n")

        report_content.append("10. MIP image generation status:")
        report_content.append("- " + mip_message + "\n")

        # Add feature extraction status
        report_content.append("11. Feature extraction status:")
        report_content.append("- " + message_extract_feat + "\n")

        # Count the total number of warnings
        total_warnings = total_misaligned_slices + warnings_replicates
        
        # Add warning summary
        report_content.append(f"Total warnings detected: {total_warnings}.")
        
        # Write the report to the specified file
        with open(report_filepath, 'w') as report_file:
            report_file.write("\n".join(report_content))
            
        print(f"Final report saved to {report_filepath}") #!! Path del container


def run_dicom_reorient_preprocessor(protocol, local_config, mapping_file, series_group_name):
 
    # Define the input directory path
    input_dir = os.getenv("INPUT_DIR")

    # Clear log at the start of validation
    clear_log_file(input_dir, LOG_FILENAME)

    image_group_config = protocol.get(series_group_name, {}).get("image", {}) 
    image_type = image_group_config['type']['selected']
    # Load the input validation state to extract num_image_patients
    input_state_file = os.path.join(input_dir, "input_validation_state.json")

    try:
        with open(input_state_file, "r") as f:
            input_state = json.load(f)
        num_image_patients = input_state.get("num_patients_with_image_data", 0)
    except Exception as e:
        raise RuntimeError(f"Failed to load state file '{input_state_file}': {e}")
    
    dicom_preprocessor = DicomReorientPreprocessor(protocol, local_config, mapping_file, num_image_patients, series_group_name)

    state = load_state(dicom_preprocessor.state_file)
    series_progress_state =  load_state(dicom_preprocessor.series_progress_file)

    # Access or create the per-series progress dictionary
    series_state = series_progress_state.setdefault(series_group_name, {})
    last_phase_done = series_state.get("last_successful_phase", 0)

    if last_phase_done < 36:

        if image_type in ["CT", "MRI"]:
            print("Running sort_dicom_slices function...") #3D only
            try:
                axial_success_message, reordering_flag = dicom_preprocessor.sort_dicom_slices()
            except Exception as e:
                log_error(input_dir, "sort_dicom_slices", e, LOG_FILENAME)
                print(f"An unexpected error occurred during sort_dicom_slices. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
                raise 

            print("Running check_and_reorient_slices function...") #3D only
            try:
                reoriented_flag = dicom_preprocessor.check_and_reorient_slices()
            except Exception as e:
                log_error(input_dir, "check_and_reorient_slices", e, LOG_FILENAME)
                print(f"An unexpected error occurred during check_and_reorient_slices. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
                raise  

        if image_type in ["CR", "DX", "RX"]:
            print("Running extract_2Ddicom_tensors_and_metadata function...")
            try:
                dicom_preprocessor.extract_2Ddicom_tensors_and_metadata()
            except Exception as e:
                log_error(input_dir, "extract_2Ddicom_tensors_and_metadata", e, LOG_FILENAME)
                print(f"An unexpected error occurred during extract_2Ddicom_tensors_and_metadata. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
                raise  

        print("Running convert_monochrome1_to_monochrome2 function...")
        try:
            conversion_info = dicom_preprocessor.convert_monochrome1_to_monochrome2()
        except Exception as e:
            log_error(input_dir, "convert_monochrome1_to_monochrome2", e, LOG_FILENAME)
            print(f"An unexpected error occurred during convert_monochrome1_to_monochrome2. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
            raise 

        print("Running process_array function...")
        try:
            final_processing_summary = dicom_preprocessor.process_array()
        except Exception as e:
            log_error(input_dir, "process_array", e, LOG_FILENAME)
            print(f"An unexpected error occurred during process_array. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
            raise 

        print("Running resize_array function...")
        try:
            final_message_resize = dicom_preprocessor.resize_array()
        except Exception as e:
            log_error(input_dir, "resize_array", e, LOG_FILENAME)
            print(f"An unexpected error occurred during resize_array. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
            raise 

        print("Running save_labels function...")
        try:
            final_message_npy_and_labels = dicom_preprocessor.save_labels()
        except Exception as e:
            log_error(input_dir, "save_labels", e, LOG_FILENAME)
            print(f"An unexpected error occurred during save_labels. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
            raise 

        print("Running convert_npy_to_image function...")
        try:
            message_from_npy_to_image = dicom_preprocessor.convert_npy_to_image()
        except Exception as e:
            log_error(input_dir, "convert_npy_to_image", e, LOG_FILENAME)
            print(f"An unexpected error occurred during convert_npy_to_image. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
            raise 

        if image_type in ["CT", "MRI"]:
            print("Running create_and_save_MIP_image function...")
            try:
                mip_message = dicom_preprocessor.create_and_save_MIP_image()
            except Exception as e:
                log_error(input_dir, "create_and_save_MIP_image", e, LOG_FILENAME)
                print(f"An unexpected error occurred during create_and_save_MIP_image. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
                raise 

            print("Running extract_features_from_MIP function...")
            try:
                _, _, message_extract_feat = dicom_preprocessor.extract_features_from_MIP()
            except Exception as e:
                log_error(input_dir, "extract_features_from_MIP", e, LOG_FILENAME)
                print(f"An unexpected error occurred during extract_features_from_MIP. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
                raise 

            print("Running check_image_replicates_after_processing function...") # check phase 36
            try:
                replicate_slice_message, state["num_slices_group"], state["total_series"], state["total_slices"], state["warnings_replicates"], input_state["num_patients_with_image_data"] = dicom_preprocessor.check_image_replicates_after_processing()
                save_state(state, dicom_preprocessor.state_file)  # Save updated state
                save_state(input_state, input_state_file)

                series_state["last_successful_phase"] = 36
                series_progress_state[series_group_name] = series_state
                save_state(series_progress_state, dicom_preprocessor.series_progress_file)
            except Exception as e:
                log_error(input_dir, "check_image_replicates_after_processing", e, LOG_FILENAME)
                print(f"An unexpected error occurred during check_image_replicates_after_processing. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
                raise

            print("Running analyze_series_alignment function...")  # check phase 37
            try:
                message_series_alignment, total_misaligned_slices = dicom_preprocessor.analyze_series_alignment()
            except Exception as e:
                log_error(input_dir, "analyze_series_alignment", e, LOG_FILENAME)
                print(f"An unexpected error occurred during analyze_series_alignment. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
                raise 

            print("Running generate_DicomReorientPreprocess3D_final_report function...")
            try:
                dicom_preprocessor.generate_DicomReorientPreprocess3D_final_report(axial_success_message, reordering_flag, reoriented_flag, conversion_info, final_processing_summary, final_message_resize, replicate_slice_message, state.get("warnings_replicates"), final_message_npy_and_labels, message_from_npy_to_image, message_series_alignment, total_misaligned_slices, mip_message, message_extract_feat)
            except Exception as e:
                log_error(input_dir, "generate_DicomReorientPreprocess3D_final_report", e, LOG_FILENAME)
                print(f"An unexpected error occurred during generate_DicomReorientPreprocess3D_final_report. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
                raise 

        if image_type in ["CR", "DX", "RX"]:
            print("Running extract_features_from_2Dimages function...")
            try:
                _, _, message_extract_feat = dicom_preprocessor.extract_features_from_2Dimages()
            except Exception as e:
                log_error(input_dir, "extract_features_from_2Dimages", e, LOG_FILENAME)
                print(f"An unexpected error occurred during extract_features_from_2Dimages. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
                raise

            print("Running check_image_replicates_after_processing function...") # check phase 36
            try:
                replicate_slice_message, state["num_slices_group"], state["total_series"], state["total_slices"], state["warnings_replicates"], input_state["num_patients_with_image_data"] = dicom_preprocessor.check_image_replicates_after_processing()
                save_state(state, dicom_preprocessor.state_file)  # Save updated state
                save_state(input_state, input_state_file)
                
                series_state["last_successful_phase"] = 36
                series_progress_state[series_group_name] = series_state
                save_state(series_progress_state, dicom_preprocessor.series_progress_file)
            except Exception as e:
                log_error(input_dir, "check_image_replicates_after_processing", e, LOG_FILENAME)
                print(f"An unexpected error occurred during check_image_replicates_after_processing. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
                raise

            print("Running generate_DicomPreprocessing2D_final_report function...")
            try:
                dicom_preprocessor.generate_DicomPreprocessing2D_final_report(conversion_info, final_processing_summary, final_message_resize, replicate_slice_message, state.get("warnings_replicates"), final_message_npy_and_labels, message_from_npy_to_image, message_extract_feat)
            except Exception as e:
                log_error(input_dir, "generate_DicomPreprocessing2D_final_report", e, LOG_FILENAME)
                print(f"An unexpected error occurred during generate_DicomPreprocessing2D_final_report. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
                raise 












                

    
    
    
    






