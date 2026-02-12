import pandas as pd
import os 
import json
from datetime import datetime
import shutil
import csv
from utils import clear_log_file, log_error, normalize_path

LOG_FILENAME = "genomics_structure_builder_error.log"

class GenomicsStructureBuilder:

    def __init__(self, protocol, local_config):
        self.protocol = protocol
        self.local_config = local_config
        self.study_path = os.getenv("ROOT_NAME")
        self.host_study_path = os.getenv("HOST_ROOT_DIR")
        self.input_dir = os.getenv("INPUT_DIR")
        self.host_input_dir = os.getenv("HOST_BASE_DATA_DIR")
        self.genomics_dir = os.path.join(self.input_dir, "GENOMICS")
        self.host_genomics_dir = os.path.join(self.host_input_dir, "GENOMICS")
        self.genomics_mapping_filename = self.local_config["Local config"]["genomic data"]["patient-sample mapping file"]
        self.patients_list_filename = self.local_config["Local config"]["patients list"]
        self.patients_list_path = os.path.join(self.input_dir, self.patients_list_filename)
        self.host_patients_list_path = os.path.join(self.host_input_dir, self.patients_list_filename)
        self.output_directory_report = os.path.join(self.input_dir, "Reports", "Genomic Data Report")
        self.host_output_directory_report = os.path.join(self.host_input_dir, "Reports", "Genomic Data Report")
        self.output_directory_data = os.path.join(self.study_path, "GENOMICS")
        self.host_output_directory_data = os.path.join(self.host_study_path, "GENOMICS")
        os.makedirs(self.output_directory_report, exist_ok=True)

    
    def create_patient_folder_structure(self):
        """
        Creates a standardized folder structure for genomics data:
            root/GENOMICS/patientXXXXX_ABCDE/sample001_ABCDE/
        
        Based on:
        - patients.csv (mapping of old ID → new ID)
        - input/GENOMICS/mapping.csv (patient_id → sample_id)
        
        Also produces sample_mapping.csv inside self.genomics_dir.
        """
        # ---------------------------------------------------------
        # 1. Load patients.csv (already standardized)
        # --------------------------------------------------------- 
    
        # Detect delimiter
        with open(self.patients_list_path, "r") as f:
            dialect = csv.Sniffer().sniff(f.read(1024))
            sep = dialect.delimiter
    
        patients_df = pd.read_csv(self.patients_list_path, sep=sep)
    
        # patients.csv columns:
        # Patient ID | Clinical Data Available | Image Data Available | Genomic Data Available | New Patient ID
        patient_to_newid = dict(zip(
            patients_df["Patient ID"].astype(str),
            patients_df["New Patient ID"].astype(str)
        ))
        
        # ---------------------------------------------------------
        # 2. Load mapping.csv (raw genomics input)
        # ---------------------------------------------------------
        mapping_path = os.path.join(self.genomics_dir, self.genomics_mapping_filename)
        mapping_df = pd.read_csv(mapping_path)
        
        # Expecting: patient_id, sample_id, sample_data, batch_id, label

        # ---------------------------------------------------------
        # 3. Prepare sample mapping output
        # ---------------------------------------------------------
        sample_mapping_records = []

        # ---------------------------------------------------------
        # 4. Create folder structure per patient
        # ---------------------------------------------------------
        for _, row in mapping_df.iterrows():
            raw_patient = str(row["patient_id"])
            raw_sample = str(row["sample_id"])
    
            if raw_patient not in patient_to_newid:
                raise RuntimeError(
                    f"Genomics mapping contains patient '{raw_patient}' "
                    f"which is missing from patients.csv."
                )
    
            new_patient = patient_to_newid[raw_patient]
            five_letters = new_patient[-5:]
    
            # Build new sample folder name
            new_sample_name = f"sample001_{five_letters}"
    
            # Create output folders
            patient_out_dir = os.path.join(self.output_directory_data, new_patient)
            sample_out_dir = os.path.join(patient_out_dir, new_sample_name)
    
            os.makedirs(sample_out_dir, exist_ok=True)
    
            # Save mapping entry (for sample_mapping.csv) (OS-independent path separator)
            sample_mapping_records.append((os.path.join(raw_patient, raw_sample), new_sample_name))

        # ---------------------------------------------------------
        # 5. Save sample_mapping.csv
        # ---------------------------------------------------------
        sample_mapping_df = pd.DataFrame(
            sample_mapping_records,
            columns=["Original Sample Name", "New Sample Name"]
        )
    
        sample_mapping_path = os.path.join(self.genomics_dir, "sample_mapping.csv")
        host_sample_mapping_path = os.path.join(self.host_genomics_dir, "sample_mapping.csv")

        sample_mapping_df.to_csv(sample_mapping_path, index=False)
    
        # ---------------------------------------------------------
        # 6. Final message
        # ---------------------------------------------------------
        gen_structure_creation_msg = (
            "Genomics folder structure created successfully:\n"
            f"  1. Patient and sample folders created inside: {self.host_output_directory_data}\n" # Metto path locali!ok
            f"  2. Sample mapping file saved at: {host_sample_mapping_path}" # Metto path locali!ok
        )
    
        return gen_structure_creation_msg
    

    def organize_and_rename_files(self):
        """
        Read raw genomics input files, rename them using standardized rules,
        and place them into the corresponding patient/sample folders.
        """ 
        # ---------------------------------------------------------
        # 1. Load patients.csv
        # ---------------------------------------------------------
        patients_path = os.path.join(
            self.input_dir,
            self.local_config["Local config"]["patients list"]
        )
        patients_df = pd.read_csv(patients_path)
    
        patient_to_newid = dict(zip(
            patients_df["Patient ID"].astype(str),
            patients_df["New Patient ID"].astype(str)
        ))
    
        # ---------------------------------------------------------
        # 2. Load mapping.csv (raw genomics map)
        # ---------------------------------------------------------
        mapping_path = os.path.join(self.genomics_dir, self.genomics_mapping_filename)
        mapping_df = pd.read_csv(mapping_path)
    
        # ---------------------------------------------------------
        # 3. Load sample_mapping.csv
        # ---------------------------------------------------------
        sample_mapping_path = os.path.join(self.genomics_dir, "sample_mapping.csv")
        sample_mapping_df = pd.read_csv(sample_mapping_path)
    
        original_to_new_sample = {
            normalize_path(orig): new
            for orig, new in zip(
                sample_mapping_df["Original Sample Name"].astype(str),
                sample_mapping_df["New Sample Name"].astype(str)
            )
        }
        
        # ---------------------------------------------------------
        # 4. Read experiment_metadata.json once
        # ---------------------------------------------------------
        experiment_metadata_path = os.path.join(self.genomics_dir, "experiment_metadata.json")
        with open(experiment_metadata_path, "r") as f:
            experiment_metadata = json.load(f)
            
        # ---------------------------------------------------------
        # 5. Process each sample
        # ---------------------------------------------------------
        patients_processed = set()
        num_samples_processed = 0

        for _, row in mapping_df.iterrows():
            raw_patient = str(row["patient_id"])
            raw_sample = str(row["sample_id"])
            label = row["label"]
    
            original_key = normalize_path(os.path.join(raw_patient, raw_sample))
            if original_key not in original_to_new_sample:
                raise RuntimeError(
                    f"Sample '{original_key}' missing from sample_mapping.csv"
                )
    
            new_sample = original_to_new_sample[original_key]
            new_patient = patient_to_newid[raw_patient]
            
            patients_processed.add(new_patient)
            num_samples_processed += 1

            code = new_patient[-5:]                # PFEDM
            sample_number = new_sample.split("_")[0]  # sample001
    
            # Output directory
            sample_out_dir = os.path.join(
                self.output_directory_data, new_patient, new_sample
            )
    
            # -----------------------------------------------------
            # 5.1 Experiment metadata
            # -----------------------------------------------------
            metadata_out = os.path.join(
                sample_out_dir,
                f"experiment_metadata_{code}_{sample_number}.json"
            )
            with open(metadata_out, "w") as f:
                json.dump(experiment_metadata, f, indent=2)
    
            # -----------------------------------------------------
            # 5.2 Genomic JSON
            # -----------------------------------------------------
            raw_json_name = f"{raw_sample}_vcf2matrix.json"
            raw_json_path = os.path.join(self.genomics_dir, raw_json_name)
    
            genomic_out = os.path.join(
                sample_out_dir,
                f"genomic_data_{code}_{sample_number}.json"
            )
    
            shutil.copyfile(raw_json_path, genomic_out)
    
            # -----------------------------------------------------
            # 5.3 Mapping CSV (single-row)
            # -----------------------------------------------------
            mapping_out_df = row.drop(labels=["label"]).copy()
            mapping_out_df["patient_id"] = new_patient
            mapping_out_df["sample_id"] = new_sample
    
            mapping_out_path = os.path.join(
                sample_out_dir,
                f"mapping_{code}_{sample_number}.csv"
            )
    
            pd.DataFrame([mapping_out_df]).to_csv(mapping_out_path, index=False)
    
            # -----------------------------------------------------
            # 5.4 Patient label file
            # -----------------------------------------------------
            label_out_path = os.path.join(
                sample_out_dir,
                f"patient_label_{code}_{sample_number}.txt"
            )
    
            with open(label_out_path, "w") as f:
                f.write(str(label))
        
        # ---------------------------------------------------------
        # 6. Final message
        # ---------------------------------------------------------
        organize_and_rename_msg = (
            "Genomic data organization completed successfully:\n"
            f"  - Patients processed: {len(patients_processed)}\n"
            f"  - Samples processed: {num_samples_processed}\n"
            f"  - Files created per sample:\n"
            f"      • experiment metadata (JSON)\n"
            f"      • genomic data (JSON)\n"
            f"      • sample-specific mapping (CSV)\n"
            f"      • patient label (TXT)\n"
            f"  - Output directory: {self.host_output_directory_data}" # metto path locale!ok
        )
    
        return organize_and_rename_msg
    

    def generate_GenomicsStructureBuilder_final_report(self, gen_structure_creation_msg, organize_and_rename_msg):
        """
        Generate a final summary report for genomics folder structure and file organization
        """
        # Define the report file path
        report_file_name = "1.GenomicsStructureBuilder_final_report.txt"
        report_file_path = os.path.join(self.output_directory_report, report_file_name)
        host_report_file_path = os.path.join(self.host_output_directory_report, report_file_name)
        
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
        
        # Open the file in write mode
        with open(report_file_path, 'w') as file:
            # Write the first report
            file.write(f"Report generated on: {formatted_datetime}\n\n")
            file.write("Genomic data structure and file organization report:\n\n")

            # 1. Folder structure creation
            file.write("Folder Structure Creation:\n")
            file.write("- " + gen_structure_creation_msg + "\n\n")

            # 2. Files organization and renaming
            file.write("File Organization And Renaming:\n")
            file.write("- " + organize_and_rename_msg + "\n\n")
            
        print(f"Final report written to {host_report_file_path}.") #path locale


def run_genomics_structure_builder(protocol, local_config):

    # Define the input directory path  
    input_dir = os.getenv("INPUT_DIR")

    # Clear log at the start of validation
    clear_log_file(input_dir, LOG_FILENAME)

    print("Running GenomicsStructureBuilder...")
    # Create an instance of GenomicsStructureBuilder
    genomics_structure_builder = GenomicsStructureBuilder(protocol, local_config)

    print("Running create_patient_folder_structure function...")
    try:
        gen_structure_creation_msg = genomics_structure_builder.create_patient_folder_structure() 
    except Exception as e:
        log_error(input_dir, "create_patient_folder_structure", e, LOG_FILENAME)
        print(f"An unexpected error occurred during create_patient_folder_structure. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
        raise

    print("Running organize_and_rename_files function...")
    try:
        organize_and_rename_msg = genomics_structure_builder.organize_and_rename_files() 
    except Exception as e:
        log_error(input_dir, "organize_and_rename_files", e, LOG_FILENAME)
        print(f"An unexpected error occurred during organize_and_rename_files. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
        raise

    print("Running generate_GenomicsStructureBuilder_final_report function...")
    try:
        genomics_structure_builder.generate_GenomicsStructureBuilder_final_report(gen_structure_creation_msg, organize_and_rename_msg) 
    except Exception as e:
        log_error(input_dir, "generate_GenomicsStructureBuilder_final_report", e, LOG_FILENAME)
        print(f"An unexpected error occurred during generate_GenomicsStructureBuilder_final_report. Details were written to {os.path.join(input_dir, LOG_FILENAME)}.")
        raise

    print("Genomics structure building completed successfully.")








    
