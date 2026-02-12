import numpy as np
import os
import pydicom
from datetime import datetime
import json
from collections import defaultdict
from utils import save_state, load_state, make_json_serializable, clear_log_file, log_error
from ...helpers import (
    generate_check_file_image_data,
    retrieve_existing_warnings_image_data,
    extract_tag_names_and_values
)

LOG_FILENAME = "dicomseg_validation_error.log"

class DicomSegValidator:

    def __init__(self, protocol, tag_list, local_config, mapping_file, num_image_patients, num_slices_group, total_series, total_slices, series_group_name):
        self.protocol = protocol
        self.tag_list = tag_list
        self.local_config = local_config
        self.mapping_file = mapping_file 
        self.num_image_patients = num_image_patients
        self.num_slices_group = num_slices_group
        self.total_series = total_series
        self.total_slices = total_slices
        self.series_group_name = series_group_name
        self.input_dir = os.getenv("INPUT_DIR")
        self.host_input_dir = os.getenv("HOST_BASE_DATA_DIR")
        self.images_dir = os.path.join(self.input_dir, "IMAGES")
        self.state_file = os.path.join(self.input_dir, "image_validation_state.json")
        self.series_progress_file = os.path.join(self.input_dir, "validation_progress_by_series.json")
        self.study_path = os.getenv("ROOT_NAME")
        self.output_directory_report = os.path.join(self.input_dir, "Reports", f"Image Data Reports {self.series_group_name}")
        self.host_output_directory_report = os.path.join(self.host_input_dir, "Reports", f"Image Data Reports {self.series_group_name}")
        self.output_directory_checkfile = os.path.join(self.study_path, "CHECKFILE")
        os.makedirs(self.output_directory_report, exist_ok=True)  
        os.makedirs(self.output_directory_checkfile, exist_ok=True)


    def check_anonymization_dicom_seg(self): #OK! # NOTE: num_slices_group, total_series, total_slices come from the image pipeline
        """
        Checks DICOM SEG files for anonymization and generates a report and check file.
        """
        phase_number = 40  
        phase_data = self.mapping_file.get("check_phase", {}).get(str(phase_number), {})
        error_descriptions = phase_data.get("errors", {})
        phase_name = phase_data.get("name", f"Phase {phase_number}")  
        
        # Format check and report names dynamically
        check_name = phase_name.lower().replace(" ", "_") 
        report_name = f"3.040.{check_name}_report"

        series_group = self.series_group_name
        series_group_mapping = self.local_config["Local config"]["radiological data"]["series_mapping"]
        series_group_subfolder = series_group_mapping.get(series_group) #input name of the series
        
        # Generate formatted date-time strings
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Retrieve existing warnings with descriptions from the check file
        check_file_path = os.path.join(self.output_directory_checkfile, "image_data_check_file.json")
        existing_warnings = retrieve_existing_warnings_image_data(check_file_path)

        report_content = ""
        non_anonymous_tags_found = False
        non_anonymized_tags_info = {}
        num_non_anonymous_files = 0
        success_anonymization = ""
        error_counts = {"E4001": None}

        # List all patient folders
        patient_folders = [f for f in os.listdir(self.images_dir) if os.path.isdir(os.path.join(self.images_dir, f)) and f != "Reports"]
    
        # Iterate through patient folders
        for patient_folder in patient_folders:
            patient_folder_path = os.path.join(self.images_dir, patient_folder)
            
            if os.path.isdir(patient_folder_path):
                series_folder = os.path.normpath(series_group_subfolder).split(os.sep)[-1].strip()
                series_folder_path = os.path.join(patient_folder_path, series_folder)
                    
                if os.path.isdir(series_folder_path):
                    # Look for a unique folder inside the series folder
                    subfolders = [sf for sf in os.listdir(series_folder_path) if os.path.isdir(os.path.join(series_folder_path, sf))]
                    segmentation_folder_path = os.path.join(series_folder_path, subfolders[0])

                    # Process DICOM segmentation files inside this folder
                    for filename in os.listdir(segmentation_folder_path):
                        if filename.endswith(".dcm"):
                            file_path = os.path.join(segmentation_folder_path, filename)
                            ds = pydicom.dcmread(file_path)

                            tag_names_and_values = extract_tag_names_and_values(ds, self.tag_list)

                            if any(value not in ('', None) for value in tag_names_and_values.values()):
                                non_anonymous_tags_found = True
                                non_anonymized_tags_info[(patient_folder, series_folder, filename)] = tag_names_and_values
                                num_non_anonymous_files += 1

                            # Check "Patient Identity Removed" (DICOM tag (0x0012, 0x0062))
                            patient_identity = ds.get((0x0012, 0x0062), None)
                            if patient_identity is not None and patient_identity.value != 'YES':
                                non_anonymous_tags_found = True
                                tag_names_and_values["Patient Identity Removed"] = patient_identity.value
                                    
        # Update error count if non-anonymized tags are found
        if non_anonymous_tags_found:
            error_counts["E4001"] = num_non_anonymous_files
        else:
            error_counts["E4001"] = None  # Ensure None is set when no errors

        # Generate the check file
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

        # Generate the report
        if non_anonymous_tags_found:
            report_content += f"Report generated on: {current_datetime}\n\n"
            report_content += f"{phase_name} report:\n\n"
            report_content += f"Error E4001: {error_descriptions.get('E4001', None)}\n"
            report_content += "- Details:\n"
            for (patient_folder, series_folder, filename), tag_values in non_anonymized_tags_info.items():
                non_anonymous_tags = {k: v for k, v in tag_values.items() if v != ''}
                report_content += f"  - Patient: {patient_folder}, Series: {series_folder}, File: {filename}\n"
                report_content += f"     Non-anonymized tags: {non_anonymous_tags}\n\n"

            report_content += f"Total Errors: {num_non_anonymous_files}\n"
            
            # Save the report
            report_name = f"{report_name}.txt" 
            report_path = os.path.join(self.output_directory_report, report_name)
            host_report_path = os.path.join(self.host_output_directory_report, report_name)

            with open(report_path, "w") as report_file:
                report_file.write(report_content)
    
            raise ValueError(
                f"Anonymization check failed: one or more DICOM files contain non-anonymized tags. "
                f"See the detailed report at: {host_report_path}"
            ) #!! Path del container, deve essere quello locale.ok
        else:
            success_anonymization = "All specified tags have been anonymized in all DICOM segmentation files."
            return success_anonymization
        

    def check_num_slices_dicom_seg(self): #OK! #41
        
        phase_number = 41  
        phase_data = self.mapping_file.get("check_phase", {}).get(str(phase_number), {})
        error_descriptions = phase_data.get("errors", {})
        phase_name = phase_data.get("name", f"Phase {phase_number}")  
        
        # Format check and report names dynamically
        check_name = phase_name.lower().replace(" ", "_") 
        report_name = f"3.041.{check_name}_report"  
        # Generate formatted date-time strings
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        series_group = self.series_group_name
        series_group_mapping = self.local_config["Local config"]["radiological data"]["series_mapping"]
        series_group_subfolder = series_group_mapping.get(series_group) #input name of the series
        
        # Retrieve existing warnings
        check_file_path = os.path.join(self.output_directory_checkfile, "image_data_check_file.json")
        existing_warnings = retrieve_existing_warnings_image_data(check_file_path)
        
        report_content = ""
        discrepancy_info = ""
        num_errors = 0
        has_error = False
        error_counts = {"E4101": None}

        # List all patient folders
        patient_folders = [
            f for f in os.listdir(self.images_dir)
            if os.path.isdir(os.path.join(self.images_dir, f)) and f != "Reports"
        ]
    
        for patient_folder in patient_folders:
            patient_folder_path = os.path.join(self.images_dir, patient_folder)
            if os.path.isdir(patient_folder_path): 
                series_folder = os.path.normpath(series_group_subfolder).split(os.sep)[-1].strip()
                series_folder_path = os.path.join(patient_folder_path, series_folder)
                
                if os.path.isdir(series_folder_path):
                    # Get the segmentation file and volume DICOM files
                    subfolders = [
                        sf for sf in os.listdir(series_folder_path)
                        if os.path.isdir(os.path.join(series_folder_path, sf))
                    ]
                    segmentation_folder_path = os.path.join(series_folder_path, subfolders[0])
                    segmentation_file = next(
                        (f for f in os.listdir(segmentation_folder_path) if f.endswith(".dcm")),
                        None
                    )

                    segmentation_path = os.path.join(segmentation_folder_path, segmentation_file)
                    ds = pydicom.dcmread(segmentation_path)

                    # Initialize structure data
                    structure_data = {}
                    segment_details = {}

                    # Extract segment details from SegmentSequence
                    for segment in ds.SegmentSequence:
                        segment_number = segment.SegmentNumber
                        anatomical_structure = segment.SegmentLabel if hasattr(segment, "SegmentLabel") else "Unknown Structure"
                        segment_details[segment_number] = anatomical_structure
                    
                    for i, frame in enumerate(ds.PerFrameFunctionalGroupsSequence):
                        # Get segment number and anatomical structure
                        segment_number = frame.SegmentIdentificationSequence[0].ReferencedSegmentNumber
                        anatomical_structure = segment_details.get(segment_number, "Unknown Structure")
                
                        # Group slices by segment number and structure
                        if (segment_number, anatomical_structure) not in structure_data:
                            structure_data[(segment_number, anatomical_structure)] = []
                
                        structure_data[(segment_number, anatomical_structure)].append(i)

                    for (seg_number, anatomical_structure), slice_data in structure_data.items():
                        
                        segmentation_tensor = ds.pixel_array[slice_data, :, :]
                    
                        # Validate the number of slices for this segment
                        num_segment_slices = segmentation_tensor.shape[0]

                        # Count the number of slices in the corresponding volume
                        volume_slices = [
                            f for f in os.listdir(series_folder_path) if f.endswith(".dcm")
                        ]
                        num_volume_slices = len(volume_slices)

                        # Compare the number of slices
                        if num_segment_slices != num_volume_slices:
                            has_error = True
                            num_errors += 1
                            error_counts["E4101"] = 1 if error_counts["E4101"] is None else error_counts["E4101"] + 1
                            discrepancy_info += (
                                f"  - Discrepancy found for patient '{patient_folder}', "
                                f"series '{series_folder}', segment '{anatomical_structure}' "
                                f"(segment number {seg_number}).\n"
                                f"    Segmentation slices: {num_segment_slices}, "
                                f"Volume slices: {num_volume_slices}\n\n"
                            )

        # Generate the report and check file
        report_name = f"{report_name}.txt"
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


        if has_error:
            report_filename = os.path.join(self.output_directory_report, report_name)
            host_report_filename = os.path.join(self.host_output_directory_report, report_name)
            report_content += f"Report generated on: {current_datetime}\n\n"
            report_content += f"{phase_name} report:\n\n"
            report_content += f"Error E4101: {error_descriptions.get('E4101', None)}\n"
            report_content += f"- Details:\n"
            report_content += discrepancy_info
            report_content += f"Total Errors: {num_errors}"
            with open(report_filename, "w") as report_file:
                report_file.write(report_content)
            raise Exception(
                f"Discrepancies found in segmentation vs. volume slice count for one or more segments. "
                f"See the detailed report at: {host_report_filename}") #!! Path del container, deve essere quello locale.ok
        else: 
            success_numslices_dicomseg = "All segmentation files match the slice count of their corresponding volumes."
            return success_numslices_dicomseg
        

    def check_slice_position_seg_volume(self): 
        
        phase_number = 42  
        phase_data = self.mapping_file.get("check_phase", {}).get(str(phase_number), {})
        error_descriptions = phase_data.get("errors", {})
        phase_name = phase_data.get("name", f"Phase {phase_number}")  
        
        # Format check and report names dynamically
        check_name = phase_name.lower().replace(" ", "_") 
        report_name = f"3.042.{check_name}_report"
        
        # Generate formatted date-time strings
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        series_group = self.series_group_name
        series_group_mapping = self.local_config["Local config"]["radiological data"]["series_mapping"]
        series_group_subfolder = series_group_mapping.get(series_group) #input name of the series
    
        report_content = ""
        discrepancy_info = ""
        num_errors = 0
        has_error = False
        first_discrepancy = True
        error_counts = {"E4201": None}

        # Retrieve existing warnings
        check_file_path = os.path.join(self.output_directory_checkfile, "image_data_check_file.json")
        existing_warnings = retrieve_existing_warnings_image_data(check_file_path)
            
        # List all patient folders
        patient_folders = [
            f for f in os.listdir(self.images_dir)
            if os.path.isdir(os.path.join(self.images_dir, f)) and f != "Reports"
        ]
        
        for patient_folder in patient_folders:
            patient_folder_path = os.path.join(self.images_dir, patient_folder)
            if os.path.isdir(patient_folder_path):
                series_folder = os.path.normpath(series_group_subfolder).split(os.sep)[-1].strip()
                series_folder_path = os.path.join(patient_folder_path, series_folder)
            
                if os.path.isdir(series_folder_path):
                    # Get the segmentation file
                    subfolders = [
                        sf for sf in os.listdir(series_folder_path)
                        if os.path.isdir(os.path.join(series_folder_path, sf))
                    ]
                    segmentation_folder_path = os.path.join(series_folder_path, subfolders[0])
                    segmentation_file = next(
                        (f for f in os.listdir(segmentation_folder_path) if f.endswith(".dcm")),
                        None
                    )

                    segmentation_path = os.path.join(segmentation_folder_path, segmentation_file)
                    ds = pydicom.dcmread(segmentation_path)

                    # Extract segment details from SegmentSequence
                    segment_details = {}
                    for segment in ds.SegmentSequence:
                        segment_number = segment.SegmentNumber
                        anatomical_structure = segment.SegmentLabel if hasattr(segment, "SegmentLabel") else "Unknown Structure"
                        segment_details[segment_number] = anatomical_structure

                    # Extract Z-positions from the segmentation
                    segmentation_z_positions = {}
                    for i, frame in enumerate(ds.PerFrameFunctionalGroupsSequence):
                        segment_number = frame.SegmentIdentificationSequence[0].ReferencedSegmentNumber
                        z_position = frame.PlanePositionSequence[0].ImagePositionPatient[2]  # Z position is the third coordinate in ImagePositionPatient
                        if segment_number not in segmentation_z_positions:
                            segmentation_z_positions[segment_number] = []
                        segmentation_z_positions[segment_number].append(z_position)
                        #print("segmentation_z_positions:", segmentation_z_positions)


                    # Extract Z-positions from the corresponding volume slices
                    volume_files = [
                        f for f in os.listdir(series_folder_path) if f.endswith(".dcm")
                    ]
                    volume_z_positions = []
                    for volume_file in volume_files:
                        volume_path = os.path.join(series_folder_path, volume_file)
                        volume_ds = pydicom.dcmread(volume_path)
                        z_position = volume_ds.ImagePositionPatient[2]
                        volume_z_positions.append(z_position)

                    # Sort the Z-positions in increasing order (if needed)
                    if volume_z_positions != sorted(volume_z_positions):
                        volume_z_positions = sorted(volume_z_positions)
                        #print("volume_z_positions:", volume_z_positions)
                    precision = 5
                    epsilon = 0.0001
                        
                    for segment_number, z_positions in segmentation_z_positions.items():
                        # Sort segmentation Z-positions for this segment
                        if z_positions != sorted(z_positions):
                            z_positions = sorted(z_positions)
                            
                        rounded_volume_z_positions = [round(z, precision) for z in volume_z_positions]
                        rounded_z_position = [round(z, precision) for z in z_positions]

                        # Check Z-position values with tolerance
                        for i, (seg_z, vol_z) in enumerate(zip(rounded_z_position, rounded_volume_z_positions)):
                            if abs(seg_z - vol_z) > epsilon:  # Compare with tolerance
                                # Find the index of the mismatch
                                mismatch_index = rounded_z_position.index(seg_z)
                                
                                has_error = True
                                num_errors += 1
                                error_counts["E4201"] = 1 if error_counts["E4201"] is None else error_counts["E4201"] + 1
                                
                                anatomical_structure = segment_details.get(segment_number, "Unknown Structure")

                                # Add a newline only if it's not the first discrepancy
                                if not first_discrepancy:
                                    discrepancy_info += "\n" 
                                
                                discrepancy_info += (
                                    f"  - Discrepancy found for patient '{patient_folder}', "
                                    f"    series '{series_folder}', segment '{anatomical_structure}' "
                                    f"    (segment number {segment_number}).\n"
                                    f"     Mismatch at position {mismatch_index + 1}: " # index of dicom in increasing order 
                                    f"     Segmentation Z = {seg_z}, Volume Z = {vol_z}\n"
                                )

                                # Set the flag to False after the first discrepancy
                                first_discrepancy = False

        # Generate the report and check file
        report_name = f"{report_name}.txt"
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
    
        if has_error:
            report_filename = os.path.join(self.output_directory_report, report_name)
            host_report_filename = os.path.join(self.host_output_directory_report, report_name)

            report_content = f"Report generated on: {current_datetime}\n\n"
            report_content += f"{phase_name} report:\n\n"
            report_content += f"Error E4201: {error_descriptions.get('E4201', None)}\n"
            report_content += f"- Details:\n"
            report_content += discrepancy_info
            report_content += f"\nTotal Errors: {num_errors}"

            with open(report_filename, "w") as report_file:
                report_file.write(report_content)
            raise Exception(
                f"Z-position mismatches detected between segmentation and volume slices in one or more segments. "
                f"See the detailed report at: {host_report_filename}") #!! Path del container, deve essere quello locale.ok
        else: 
            success_zpositions = "All segmentation Z-positions match the corresponding volume Z-positions."
            return success_zpositions
        
    
    def validate_label_conformity(self): 
        """
        Validate the conformity of DICOM SEG file segment labels against the protocol.
        Flag cases with duplicate labels. Generates a report of any discrepancies found.
        """

        phase_number = 44  
        phase_data = self.mapping_file.get("check_phase", {}).get(str(phase_number), {})
        error_descriptions = phase_data.get("errors", {})
        phase_name = phase_data.get("name", f"Phase {phase_number}")  
        
        # Format check and report names dynamically
        check_name = phase_name.lower().replace(" ", "_") 
        report_name = f"3.044.{check_name}_report"  
        
        # Generate formatted date-time strings for report names
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        series_group = self.series_group_name
        series_group_mapping = self.local_config["Local config"]["radiological data"]["series_mapping"]
        series_group_subfolder = series_group_mapping.get(series_group) #input name of the series
        seg_group_config = self.protocol.get(series_group, {}).get("segmentation", {}) #original self.protocol["segmentation"]

        # Initialize report content and error tracking
        discrepancy_info = ""
        report_content = ""
        num_errors = 0
        has_error = False
        duplicate_label_flag = False
        duplicate_labels = defaultdict(list)  # To track labels with multiple seg 
        label_multiplicity = defaultdict(list)  # To track multiplicity across series
        error_counts = {"E4401": None}

        # Retrieve existing warnings
        check_file_path = os.path.join(self.output_directory_checkfile, "image_data_check_file.json")
        existing_warnings = retrieve_existing_warnings_image_data(check_file_path)
        
        protocol_segments_number = seg_group_config["segments_number"]
        protocol_segments_labels = seg_group_config["segments_labels"]
        protocol_segmentation_code = seg_group_config["segmentation_code"]

        # Construct a dictionary from the protocol segments, excluding background (0)
        protocol_dict = {}
        for i in range(1, protocol_segments_number + 1):  # Start from 1 to exclude background (0)
            seg_code = protocol_segmentation_code[i]
            seg_label = protocol_segments_labels[i]
            protocol_dict[seg_code] = seg_label  # Mapping code to label

        # List all patient folders
        patient_folders = [
            f for f in os.listdir(self.images_dir)
            if os.path.isdir(os.path.join(self.images_dir, f)) and f != "Reports"
        ]
        
        for patient_folder in patient_folders:
            patient_folder_path = os.path.join(self.images_dir, patient_folder)
            if os.path.isdir(patient_folder_path):
                series_folder = os.path.normpath(series_group_subfolder).split(os.sep)[-1].strip()
                series_folder_path = os.path.join(patient_folder_path, series_folder)
                
                if os.path.isdir(series_folder_path):
                    # Get the segmentation file (assuming .dcm files)
                    subfolders = [
                        sf for sf in os.listdir(series_folder_path)
                        if os.path.isdir(os.path.join(series_folder_path, sf))
                    ]
                    segmentation_folder_path = os.path.join(series_folder_path, subfolders[0])
                    segmentation_file = next(
                        (f for f in os.listdir(segmentation_folder_path) if f.endswith(".dcm")),
                        None
                    )
    
                    segmentation_path = os.path.join(segmentation_folder_path, segmentation_file)
                    # Read DICOM file
                    ds = pydicom.dcmread(segmentation_path)

                    # Extract segment details from the DICOM SEG file
                    segment_details = defaultdict(list)
                    for segment in ds.SegmentSequence:
                        segment_number = segment.SegmentNumber
                        anatomical_structure = segment.SegmentLabel if hasattr(segment, "SegmentLabel") else "Unknown Structure"
                        segment_details[anatomical_structure].append(segment_number)

                    # Track label multiplicity for each anatomical structure
                    for anatomical_structure, segment_numbers in segment_details.items():
                        label_multiplicity[anatomical_structure].append(set(segment_numbers))

                    # Check for duplicate labels
                    for anatomical_structure, segment_numbers in segment_details.items():
                        if len(segment_numbers) > 1:
                            duplicate_label_flag = True
                            duplicate_labels[anatomical_structure].append(set(segment_numbers))
                        
                            
                            # Check if one of the segment codes matches the protocol
                            matching_codes = [
                                code for code in segment_numbers if protocol_dict.get(code) == anatomical_structure
                            ]
                            if not matching_codes:
                                expected_label = next(
                                    (label for code, label in protocol_dict.items() if code in segment_numbers),
                                    None
                                )
                                expected_code = next(
                                    (code for code, label in protocol_dict.items() if label == anatomical_structure),
                                    None
                                )
                                if expected_code is None and expected_label is None:
                                    # Find expected label(s) and code(s) that exist in the protocol but are missing in the data
                                    missing_labels = [label for label in protocol_segments_labels if label not in segment_details and label.lower() != "background"]
                                    missing_codes = [code for code in protocol_segmentation_code if code not in segment_numbers and code != 0]
                                    discrepancy_info += (
                                        f"  - patient '{patient_folder}', series '{series_folder}', DICOM SEG file '{segmentation_file}':"
                                        f" Multiple segments found for '{anatomical_structure}': {segment_numbers}."
                                        f" Both segment label and code do not match the protocol.\n"
                                        f"    Expected labels (from protocol, but missing in data): {missing_labels}.\n"
                                        f"    Expected codes (from protocol, but missing in data): {missing_codes}.\n"
                                        f"    Found label: '{anatomical_structure}', found code(s): {segment_numbers}.\n"
                                    )
                                elif expected_code is None:
                                    discrepancy_info += (
                                        f"  - patient '{patient_folder}', series '{series_folder}', DICOM SEG file '{segmentation_file}': "
                                        f" Multiple segments found for '{anatomical_structure}': {segment_numbers}. "
                                        f" No matching protocol label, the expected label is: {expected_label}.\n"
                                    )
                                elif expected_label is None: 
                                    discrepancy_info += (
                                        f"  - patient '{patient_folder}', series '{series_folder}', DICOM SEG file '{segmentation_file}': "
                                        f" Multiple segments found for '{anatomical_structure}': {segment_numbers}. "
                                        f" No matching protocol code, the expected code is: {expected_code}.\n"
                                    )
                                num_errors += 1
                                has_error = True

                    # Exclude segment codes with duplicate labels
                    unique_segment_details = {
                        anatomical_structure: segment_numbers
                        for anatomical_structure, segment_numbers in segment_details.items()
                        if len(segment_numbers) == 1
                    }

                    # Now check the labels and codes against the protocol
                    for anatomical_structure, segment_numbers in unique_segment_details.items():
                        for segment_number in segment_numbers:
                            # Compare segment code and label
                            if segment_number in protocol_dict:
                                expected_label = protocol_dict[segment_number]
                                if anatomical_structure != expected_label:
                                    discrepancy_info += (
                                        f"  - patient '{patient_folder}', series '{series_folder}', DICOM SEG file '{segmentation_file}': Segment {segment_number} label mismatch. Expected '{expected_label}', found '{anatomical_structure}'.\n"
                                    )
                                    num_errors += 1
                                    has_error = True
                            else:
                                # If the segment number is not in the protocol (unexpected segment)
                                discrepancy_info += (
                                    f"  - patient '{patient_folder}', series '{series_folder}', DICOM SEG file '{segmentation_file}': Unexpected segment code {segment_number}.\n"
                                )
                                num_errors += 1
                                has_error = True

        error_counts = {"E4401": num_errors if num_errors > 0 else None}               
        report_name = f"{report_name}.txt"
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

        # Return a status message
        if has_error:
            report_filename = os.path.join(self.output_directory_report, report_name)
            host_report_filename = os.path.join(self.host_output_directory_report, report_name)
            report_content = f"Report generated on: {current_datetime}\n\n"
            report_content += f"{phase_name} report:\n\n"
            report_content += f"Error E4401: {error_descriptions.get('E4401', None)}\n"
            report_content += f"- Details:\n"
            report_content += discrepancy_info
            report_content += f"\nTotal Errors: {num_errors}"
    
            with open(report_filename, "w") as report_file:
                report_file.write(report_content)
            raise Exception(
                f"Segment label conformity check failed. "
                f"See the detailed report at: {host_report_filename}") #!! Path del container, deve essere quello locale.ok
        else:
            # Compose the final return message
            label_conformity_message = "All segment labels are conforming to the protocol."
            
            # Check if any duplicate labels were found
            if duplicate_label_flag:
                label_conformity_message += " However, labels with multiple segments were found. Details:\n"
                for anatomical_structure, segment_codes in duplicate_labels.items():
                    # Check if all series have identical segment codes
                    unique_codes = {frozenset(codes) for codes in segment_codes}
                    if len(unique_codes) == 1:
                        label_conformity_message += (
                            f"  - Label '{anatomical_structure}' has the following multiple segment codes: {list(unique_codes.pop())}. "
                        )
                        # Add note about unification
                        label_conformity_message += "Multiple segments for the same label will be unified."
                    else:
                        label_conformity_message += (
                            f"  - Label '{anatomical_structure}' has multiple segment codes with different multiplicity across series. "
                        )
                        # Add note about unification
                        label_conformity_message += "Multiple segments for the same label will be unified."
    

            return label_conformity_message, duplicate_label_flag, duplicate_labels, protocol_dict
        

    def extract_frame_to_segment_map(self, ds):
        """
        Extracts a list mapping each frame index to the corresponding segment number.
        
        Parameters:
            ds: pydicom Dataset for the DICOM SEG file
        
        Returns:
            frame_to_segment: list[int], length = number of frames
        """
        num_frames = int(ds.NumberOfFrames)
        frame_to_segment = []

        for i in range(num_frames):
            segment_number = ds.PerFrameFunctionalGroupsSequence[i] \
                                .SegmentIdentificationSequence[0] \
                                .ReferencedSegmentNumber
            frame_to_segment.append(segment_number)
            
        return frame_to_segment
    

    def extract_segment_volumes(self, ds):
        """
        Extract segment volumes and labels from DICOM SEG dataset.
        
        Parameters:
            ds: pydicom Dataset
        
        Returns:
            segment_volumes: dict {segment_number: 3D numpy array mask}
            segment_labels: dict {segment_number: segment_label}
        """
        segments = ds.SegmentSequence
        segment_labels = {}
        for i, segment in enumerate(segments):
            segment_labels[i+1] = segment.SegmentLabel  # segment numbers start at 1
            
        # Get the pixel data as numpy array (num_frames, height, width)
        pixel_data = ds.pixel_array
        
        # Extract mapping of each frame to segment number
        frame_to_segment = self.extract_frame_to_segment_map(ds)
        
        # Build segment volumes using your function
        segment_volumes = self.build_segment_volumes(ds, pixel_data, frame_to_segment)
        
        return segment_volumes, segment_labels
    

    def build_segment_volumes(self, ds_mod, pixel_data, frame_to_segment):
        """
        Build a dictionary mapping segment number to a 3D numpy volume 
        constructed directly from pixel_data slices.
        
        Parameters:
        - ds_mod: DICOM dataset with PerFrameFunctionalGroupsSequence
        - pixel_data: numpy array (num_slices, height, width) with binary masks
        - frame_to_segment: list mapping frame index to segment number
        
        Returns:
        - segment_volumes: dict {segment_number: 3D numpy array (slices, H, W)}
        """
        # Group frame indices by segment
        segment_frames = {}
        for i, seg_num in enumerate(frame_to_segment):
            if seg_num not in segment_frames:
                segment_frames[seg_num] = []
            segment_frames[seg_num].append(i)
    
        # Build volumes from pixel_data slices, sorted by frame index
        segment_volumes = {}
        for seg_num, frames_indices in segment_frames.items():
            frames_indices_sorted = sorted(frames_indices)
            volume = np.stack([pixel_data[idx] for idx in frames_indices_sorted], axis=0)
            segment_volumes[seg_num] = volume
    
        return segment_volumes
    

    def check_pixel_overlap_single_label_seg(self):
        """
        Checks for overlapping pixels in single DICOM-SEG files containing multiple labels (segments).
        
        If overlaps are found, generates a report and raises an exception.
        
        Returns:
            str: Status message if no overlaps are found.
        
        Raises:
            Exception: If overlapping voxels between labels are detected.
        """
        phase_number = 46 
        phase_data = self.mapping_file.get("check_phase", {}).get(str(phase_number), {})
        error_descriptions = phase_data.get("errors", {})
        phase_name = phase_data.get("name", f"Phase {phase_number}")  
        
        # Format check and report names dynamically
        check_name = phase_name.lower().replace(" ", "_") 
        report_name = f"3.046.{check_name}_report"

        series_group = self.series_group_name
        series_group_mapping = self.local_config["Local config"]["radiological data"]["series_mapping"]
        series_group_subfolder = series_group_mapping.get(series_group) 
        seg_group_config = self.protocol.get(series_group, {}).get("segmentation", {})

        seg_input_format = seg_group_config["segmentation_input_format"]["selected"]
        segments_number = seg_group_config["segments_number"]

        multiple_segmentation_flag = seg_input_format == "single-mask" and segments_number > 1
        
        # Only proceed if condition is met
        if not multiple_segmentation_flag:
            overlapping_pixel_message = "Overlapping pixels check is not needed. Segmentations are not single-mask with multiple segments."
            return overlapping_pixel_message

        check_file_path = os.path.join(self.output_directory_checkfile, "image_data_check_file.json")
        existing_warnings = retrieve_existing_warnings_image_data(check_file_path)

        report_info = ""
        error_counts = {"E4601": None}
        num_errors = 0
        overlap_detected = False

        patient_folders = [
            f for f in os.listdir(self.images_dir)
            if os.path.isdir(os.path.join(self.images_dir, f)) and f != "Reports"
        ]
        
        overlapping_segments = set()
        for patient_folder in patient_folders:
            patient_folder_path = os.path.join(self.images_dir, patient_folder)
            if os.path.isdir(patient_folder_path):
                series_folder = os.path.normpath(series_group_subfolder).split(os.sep)[-1].strip()
                series_folder_path = os.path.join(patient_folder_path, series_folder)

                if os.path.isdir(series_folder_path):
                    # Get the segmentation file (assuming .dcm files)
                    subfolders = [
                        sf for sf in os.listdir(series_folder_path)
                        if os.path.isdir(os.path.join(series_folder_path, sf))
                    ]
                    segmentation_folder_path = os.path.join(series_folder_path, subfolders[0])
                    segmentation_file = next(
                        (f for f in os.listdir(segmentation_folder_path) if f.endswith(".dcm")),
                        None
                    )
                    seg_path = os.path.join(segmentation_folder_path, segmentation_file)

                    try:
                        ds = pydicom.dcmread(seg_path)
                    except Exception as e:
                        print(f"Skipping file {seg_path}: not a valid DICOM SEG - {str(e)}")
                        continue
    
                    try:
                        segment_volumes, segment_labels = self.extract_segment_volumes(ds)
                    except Exception as e:
                        print(f"Error extracting segments from {seg_path}: {str(e)}")
                        continue

                    seg_nums = list(segment_volumes.keys())
                    # Check pairwise overlap between segment masks
                    for i, seg_num1 in enumerate(seg_nums):
                        vol1 = segment_volumes[seg_num1]
                        for seg_num2 in seg_nums[i+1:]:
                            vol2 = segment_volumes[seg_num2]
    
                            if vol1.shape != vol2.shape:
                                # Shapes differ - skip
                                continue
    
                            overlap = np.logical_and(vol1 > 0, vol2 > 0)
                            if np.any(overlap):
                                overlap_detected = True
                                overlap_positions = np.argwhere(overlap)
                                label1 = segment_labels.get(seg_num1, f"Segment {seg_num1}")
                                label2 = segment_labels.get(seg_num2, f"Segment {seg_num2}")
                                report_info += f"\n   Overlapping segments found in: {patient_folder}/{series_folder}/{segmentation_file}\n"
                                report_info += f"   - {label1} (#{seg_num1}) and {label2} (#{seg_num2}) segmentations\n"
                                report_info += f"   - Number of overlapping voxels: {len(overlap_positions)}\n"
                                report_info += f"   - Overlapping voxel locations (Slice [1-based], Y, X):\n"
                                for z, y, x in overlap_positions:
                                    report_info += f"      - Slice: {z+1}, Y: {y}, X: {x}\n"
                                report_info += "-" * 80 + "\n"

                                overlapping_segments.add((patient_folder, seg_num1))
                                overlapping_segments.add((patient_folder, seg_num2))
        
        if overlap_detected:
            num_errors = len(overlapping_segments)
        else:
            num_errors = 0
            
        error_counts["E4601"] = num_errors if num_errors > 0 else None
    
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
        # Write check file
        generate_check_file_image_data(
            check_name=phase_name,
            phase_number=phase_number,
            series_group_name=self.series_group_name,
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
    
        # Write report
        report_filename = os.path.join(self.output_directory_report, f"{report_name}.txt")
        host_report_filename = os.path.join(self.host_output_directory_report, f"{report_name}.txt")
        if overlap_detected:
            report_content = f"Report generated on: {current_datetime}\n\n"
            report_content += f"{phase_name} report:\n\n"
            report_content += f"Error E4601: {error_descriptions.get('E4601', '')}\n"
            report_content += f"- Details:\n"
            report_content += report_info
            
            report_content += f"\nTotal Errors: {num_errors}"
    
            with open(report_filename, "w") as f:
                f.write(report_content)
    
            raise Exception(
                f"Overlapping segmentations found in DICOM SEG. "
                f"See the detailed report at: {host_report_filename}") #!! Path del container, deve essere quello locale.ok

        overlapping_pixel_message = "No overlapping pixels found between segmentation masks within the same series."
        return overlapping_pixel_message
    

    def generate_DicomSegValidation_final_report(self, success_anonymization, success_numslices_dicomseg, success_zpositions, label_conformity_message, overlapping_pixel_message): 
        """
        Generate a comprehensive final report for the DicomSegCheckRename process.
        
        Parameters:
        - success_anonymization: Message about anonymization success.
        - success_numslices_dicomseg: Message about number of slices validation.
        - success_zpositions: Message about Z-position check success.
        - label_conformity_message: Message about label conformity check. 
        """
        
        # Get the current datetime
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")

        # Construct the final report
        final_report_lines = [
            f"Report generated on: {formatted_datetime}",
            "",
            "Final report on DICOM segmentation conformity checks:",
            "",
            "Anonymization status:",
            f"- {success_anonymization}",
            "",
            "Number of slices check:",
            f"- {success_numslices_dicomseg}",
            "",
            "Z-position consistency check:",
            f"- {success_zpositions}",
            "",
            "Label conformity check:",
            f"- {label_conformity_message}",
            "",
            "Overlapping pixel check:",
            f"- {overlapping_pixel_message}",
            "",
            "Summary:",
            "All checks completed."
        ]
    
        # Join all lines into a single string
        final_report = "\n".join(final_report_lines)
        
        # Define the report filename
        report_filename = os.path.join(self.output_directory_report, "3.DicomSegValidation_final_report.txt")
        
        # Write the final report to a file
        with open(report_filename, "w") as report_file:
            report_file.write(final_report)
        
        # Print a message indicating that the final report has been generated
        print("3.DicomSegValidation final report has been generated.")


def run_dicomseg_validator(protocol, local_config, tag_list, mapping_file, series_group_name): 
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
    
    dicomseg_validator = DicomSegValidator(protocol, tag_list, local_config, mapping_file, num_image_patients, num_slices_group, total_series, total_slices, series_group_name)

    series_progress_state =  load_state(dicomseg_validator.series_progress_file)

    # Access or create the per-series progress dictionary
    series_state = series_progress_state.setdefault(series_group_name, {})
    last_phase_done = series_state.get("last_successful_phase", 0)

    if last_phase_done < 40:
        print("Running check_anonymization_dicom_seg function...")  # check phase 40
        try:
            image_state["success_anonymization_dicomseg"] = dicomseg_validator.check_anonymization_dicom_seg()
            save_state(image_state, dicomseg_validator.state_file)  # Save updated state

            # Update series progress state
            series_state["last_successful_phase"] = 40
            series_progress_state[series_group_name] = series_state
            save_state(series_progress_state, dicomseg_validator.series_progress_file)
        except Exception as e:
            log_error(input_dir, "check_anonymization_dicom_seg", e, LOG_FILENAME)
            print(f"An unexpected error occurred during check_anonymization_dicom_seg. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
            raise 

    if last_phase_done < 41:
        print("Running check_num_slices_dicom_seg function...") # check phase 41
        try:
            image_state["success_numslices_dicomseg"] = dicomseg_validator.check_num_slices_dicom_seg()
            save_state(image_state, dicomseg_validator.state_file)

            # Update series progress state
            series_state["last_successful_phase"] = 41
            series_progress_state[series_group_name] = series_state
            save_state(series_progress_state, dicomseg_validator.series_progress_file)
        except Exception as e:
            log_error(input_dir, "check_num_slices_dicom_seg", e, LOG_FILENAME)
            print(f"An unexpected error occurred during check_num_slices_dicom_seg. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
            raise 

    if last_phase_done < 42:
        print("Running check_slice_position_seg_volume function...") # check phase 42
        try:
            image_state["success_zpositions"] = dicomseg_validator.check_slice_position_seg_volume()
            save_state(image_state, dicomseg_validator.state_file)

            # Update series progress state
            series_state["last_successful_phase"] = 42
            series_progress_state[series_group_name] = series_state
            save_state(series_progress_state, dicomseg_validator.series_progress_file)
        except Exception as e:
            log_error(input_dir, "check_slice_position_seg_volume", e, LOG_FILENAME)
            print(f"An unexpected error occurred during check_slice_position_seg_volume. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
            raise 

        # NOTE: Phase 43 intentionally skipped

    if last_phase_done < 44:
        print("Running validate_label_conformity function...") # check phase 44
        try:
            image_state["label_conformity_message"], image_state["duplicate_label_flag"], image_state["duplicate_labels"], image_state["protocol_dict"] = dicomseg_validator.validate_label_conformity()
            image_state["duplicate_labels"] = make_json_serializable(image_state["duplicate_labels"])
            save_state(image_state, dicomseg_validator.state_file)

            # Update series progress state
            series_state["last_successful_phase"] = 44
            series_progress_state[series_group_name] = series_state
            save_state(series_progress_state, dicomseg_validator.series_progress_file)
        except Exception as e:
            log_error(input_dir, "validate_label_conformity", e, LOG_FILENAME)
            print(f"An unexpected error occurred during validate_label_conformity. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
            raise  

        # NOTE: Phase 45 intentionally skipped

    if last_phase_done < 46:
        print("Running check_pixel_overlap_single_label_seg function...") # check phase 46
        try:
            image_state["overlapping_pixel_message"] = dicomseg_validator.check_pixel_overlap_single_label_seg()
            save_state(image_state, dicomseg_validator.state_file)

            # Update series progress state
            series_state["last_successful_phase"] = 46
            series_progress_state[series_group_name] = series_state
            save_state(series_progress_state, dicomseg_validator.series_progress_file)
        except Exception as e:
            log_error(input_dir, "check_pixel_overlap_single_label_seg", e, LOG_FILENAME)
            print(f"An unexpected error occurred during check_pixel_overlap_single_label_seg. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
            raise  

        print("Running generate_DicomSegValidation_final_report function...")
        try:
            dicomseg_validator.generate_DicomSegValidation_final_report(
                    image_state.get("success_anonymization_dicomseg"),
                    image_state.get("success_numslices_dicomseg"),
                    image_state.get("success_zpositions"),
                    image_state.get("label_conformity_message"),
                    image_state.get("overlapping_pixel_message")
                )
        except Exception as e:
            log_error(input_dir, "generate_DicomSegValidation_final_report", e, LOG_FILENAME)
            print(f"An unexpected error occurred during generate_DicomSegValidation_final_report. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
            raise  




        
    

    






    
