import os
import fnmatch
from config import (
    run_protocol_load_and_check,
    run_config_load_and_check,
    run_mappingfile_load_and_check,
    run_taglist_load
)
from utils import RestartManager, clear_log_file, log_error
from input_checks import run_input_checks_patient_mapping
from clinical import (
    run_variable_mapping_creation,
    run_clinical_data_check,
    run_clinical_data_preprocessing

)
from imaging import (
    run_imagestructure_and_label_validator
)
from imaging.images import (
    run_dicom_validator,
    run_dicom_reorient_preprocessor,
    run_nifti_validator,
    run_nifti_reorient_preprocessor,
    run_nrrd_validator,
    run_nrrd_reorient_preprocessor,
    run_image_validator,
    run_image_preprocessor
)
from imaging.segmentations import (
    run_dicomseg_validator,
    run_dicomseg_reorient_preprocessor,
    run_niftiseg_validator,
    run_niftiseg_reorient_preprocessor,
    run_nrrdseg_validator,
    run_nrrdseg_reorient_preprocessor,
    run_segimage_validator,
    run_segimage_preprocessor
)

from genomics import (
    run_genomics_structure_builder
)



LOG_FILENAME = "main_validation.log"

def cleanup_temp_files(
    temp_input_files=None,
    output_patterns=None,
    dry_run=False
):
    """
    Deletes temporary files:
    1. From input_dir (explicit file list)
    2. From Image_data_output (recursively searches patient/series folders using glob patterns)

    Args:
        temp_input_files (list): List of exact filenames to delete from input_dir.
        output_patterns (list): Glob-style patterns like "metadata*.json".
        dry_run (bool): If True, lists files instead of deleting.
    """
    input_dir = os.getenv("INPUT_DIR") 
    study_path = os.getenv("ROOT_NAME") 
    image_output_dir = os.path.join(study_path, "IMAGES")

    temp_input_files = temp_input_files or []
    output_patterns = output_patterns or ["metadata*.json"]

    # --- 1. Input Dir Cleanup ---
    for filename in temp_input_files:
        file_path = os.path.join(input_dir, filename)
        if os.path.isfile(file_path):
            if dry_run:
                print(f"[Dry Run] Would remove input file: {file_path}")
            else:
                try:
                    os.remove(file_path)
                    print(f"Removed input file: {file_path}")
                except Exception as e:
                    print(f"Error removing input file '{file_path}': {e}")

    # --- 2. Output Dir Cleanup: Match patterns in nested series folders ---
    for root, _, files in os.walk(image_output_dir):
        for file in files:
            for pattern in output_patterns:
                if fnmatch.fnmatch(file, pattern):
                    file_path = os.path.join(root, file)
                    if dry_run:
                        print(f"[Dry Run] Would remove output file: {file_path}")
                    else:
                        try:
                            os.remove(file_path)
                            print(f"Removed output file: {file_path}")
                        except Exception as e:
                            print(f"Error removing output file '{file_path}': {e}")


# Main execution flow
if __name__ == "__main__":
    try:
        # Load and validate essential configuration files: local config, protocol and mapping file
        try:
            local_config = run_config_load_and_check()
        except Exception as e:
            log_error(".", "run_config_load_and_check", e, LOG_FILENAME)  # Log to current dir (".")
            raise Exception(f"Error in 'run_config_load_and_check': {e}")

        # Define the input data directory path  
        input_dir = os.getenv("INPUT_DIR") 

        # Clear log at the start of validation
        clear_log_file(input_dir, LOG_FILENAME)

        try:
            protocol = run_protocol_load_and_check(local_config)
        except Exception as e:
            raise Exception(f"Error in 'run_protocol_load_and_check': {e}")
           
        try:
            mapping_file = run_mappingfile_load_and_check()
        except Exception as e:
            log_error(input_dir, "run_mappingfile_load_and_check", e, LOG_FILENAME)
            raise Exception(f"Error in 'run_mappingfile_load_and_check': {e}")
        
        if "clinical data" in protocol:
            try:
                run_variable_mapping_creation(protocol)
            except Exception as e:
                log_error(input_dir, "run_variable_mapping_creation", e, LOG_FILENAME)
                raise Exception(f"Error in 'run_variable_mapping_creation': {e}")
        else:
            print("Skipping variable mapping creation: 'clinical data' not found in protocol.")

        try:
            restart_manager = RestartManager(local_config)
        except Exception as e:
            log_error(input_dir, "RestartManager_init", e, LOG_FILENAME)
            raise Exception(f"Error initializing RestartManager: {e}")
        
        try:
            check_phase = restart_manager.get_check_phase()
            print("check_phase:", check_phase)
        except Exception as e:
            log_error(input_dir, "get_check_phase", e, LOG_FILENAME)
            raise Exception(f"Error in 'get_check_phase': {e}")

        # If there were no errors, call the input check and patient mapping function
        try:
            run_input_checks_patient_mapping(protocol, local_config, check_phase, mapping_file)
        except Exception as e:
            raise Exception(f"Error in 'run_input_checks_patient_mapping': {e}")

        if "clinical data" in protocol:
            # Run the clinical data check process
            try:
                simplified_df, data, num_dates = run_clinical_data_check(protocol, local_config, mapping_file, check_phase)
            except Exception as e:
                raise Exception(f"Error in 'run_clinical_data_check': {e}")
        
            # run the clinical data pre-processing workflow
            try:
                run_clinical_data_preprocessing(protocol, local_config, simplified_df, num_dates, mapping_file, check_phase)
            except Exception as e:
                raise Exception(f"Error in 'run_clinical_data_preprocessing': {e}")
            
            print("Clinical data pipeline completed successfully.")
        else:
            print("Skipping clinical data validation and preprocessing: 'clinical data' not included in protocol.")

        tag_list = None  # Initialize once, load only if needed

        if any(key.startswith("series") for key in protocol):
            for series_group_name, series_config in protocol.items():
                if not series_group_name.startswith("series"):
                    continue  # Skip non-series entries

                print(f"Processing {series_group_name}...")

                try:
                    # General structure and label validation
                    run_imagestructure_and_label_validator(
                        protocol,
                        local_config,
                        mapping_file,
                        series_group_name
                    )
                except Exception as e:
                    raise Exception(f"Error in run_imagestructure_and_label_validator for {series_group_name}: {e}")

                # -----------------------------
                # IMAGE VALIDATION + PREPROCESSING
                # -----------------------------
                print(f"--- Starting image pipeline for {series_group_name} ---")
                try:
                    if "image" in series_config:
                        image_format = series_config["image"]["image file format"]["selected"]
                        
                        if image_format == ".dcm":
                            if tag_list is None:
                                tag_list = run_taglist_load()
                            # IMAGE VALIDATION
                            run_dicom_validator(protocol, local_config, tag_list, mapping_file, series_group_name)
                            # IMAGE PREPROCESSING
                            run_dicom_reorient_preprocessor(protocol, local_config, mapping_file, series_group_name)
                        elif image_format in [".nii", ".nii.gz"]:
                            # IMAGE VALIDATION
                            run_nifti_validator(protocol, local_config, mapping_file, series_group_name)
                            # IMAGE PREPROCESSING
                            run_nifti_reorient_preprocessor(protocol, local_config, mapping_file, series_group_name)
                        elif image_format == ".nrrd":
                            # IMAGE VALIDATION
                            run_nrrd_validator(protocol, local_config, mapping_file, series_group_name)
                            # IMAGE PREPROCESSING
                            run_nrrd_reorient_preprocessor(protocol, local_config, mapping_file, series_group_name)
                        elif image_format in [".png", ".jpg", ".tiff"]:
                            # IMAGE VALIDATION
                            run_image_validator(protocol, local_config, mapping_file, series_group_name)
                            # IMAGE PREPROCESSING
                            run_image_preprocessor(protocol, local_config, mapping_file, series_group_name)
                        else:
                            print(f"Unsupported image format: {image_format} in {series_group_name}")
                except Exception as e:
                    raise Exception(f"Error in image validation or processing for {series_group_name}: {e}")
                
                print(f"--- Completed image pipeline for {series_group_name} ---")

                # -----------------------------
                # SEGMENTATION VALIDATION + PREPROCESSING
                # -----------------------------
                try:
                    if "segmentation" in series_config:
                        seg_format = series_config["segmentation"]["segmentation file format"]["selected"]

                        if seg_format == ".dcm":
                            if tag_list is None:
                                tag_list = run_taglist_load()
                            # SEGMENTATION VALIDATION
                            run_dicomseg_validator(protocol, local_config, tag_list, mapping_file, series_group_name)
                            # SEGMENTATION PREPROCESSING
                            run_dicomseg_reorient_preprocessor(protocol, local_config, mapping_file, series_group_name)
                        elif seg_format in [".nii", ".nii.gz"]:
                            # SEGMENTATION VALIDATION
                            run_niftiseg_validator(protocol, local_config, mapping_file, series_group_name)
                            # SEGMENTATION PREPROCESSING
                            run_niftiseg_reorient_preprocessor(protocol, local_config, mapping_file, series_group_name)
                        elif seg_format == ".nrrd":
                            # SEGMENTATION VALIDATION
                            run_nrrdseg_validator(protocol, local_config, mapping_file, series_group_name)
                            # SEGMENTATION PREPROCESSING
                            run_nrrdseg_reorient_preprocessor(protocol, local_config, mapping_file, series_group_name)
                        elif seg_format in [".png", ".jpg", ".tiff"]:
                            # SEGMENTATION VALIDATION
                            run_segimage_validator(protocol, local_config, mapping_file, series_group_name)
                            # SEGMENTATION PREPROCESSING
                            run_segimage_preprocessor(protocol, local_config, mapping_file, series_group_name)
                        else:
                            print(f"Unsupported segmentation format: {seg_format} in {series_group_name}")
                except Exception as e:
                    raise Exception(f"Error in segmentation validation or processing for {series_group_name}: {e}")
        else:
            print("Skipping image/segmentation: no series found in protocol.")


        if "genomic data" in protocol:
            # run the genomic data structure builder workflow
            try:
                run_genomics_structure_builder(protocol, local_config)
            except Exception as e:
                raise Exception(f"Error in 'run_genomics_structure_builder': {e}")
            
            print("Genomic data pipeline completed successfully.")
        else:
            print("Skipping genomic data structure building: 'genomic data' not included in protocol.")


        try:
            # Only run cleanup if everything else above has succeeded
            cleanup_temp_files(
                temp_input_files=["clinical_validation_state.json", "image_validation_state.json", "input_validation_state.json", "multi_index_header.csv", "single_header_data.csv", "validation_progress_by_series.json"],
                output_patterns=["metadata*.json"],
                dry_run=False  # Set to True if you want a safety check first
            )
        except Exception as e:
            log_error(input_dir, "cleanup_temp_files", e, LOG_FILENAME)
            print(f"An unexpected error occurred during cleanup_temp_files. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
            raise  

    except Exception as e:
        print(f"An error occurred in the main: {e}")


