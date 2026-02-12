import sys
import os
# Redirect stdout and stderr to a log file
print(f"Redirect output and errors to /home/output/logfile.log") #host directory

log_file = open(f"/home/output/logfile.log", 'w')

sys.stdout = log_file
sys.stderr = log_file

import glob
import hashlib
import json
from datetime import datetime
import csv

import numpy as np
import pandas as pd
import math
import psycopg2
from psycopg2.extras import execute_values

print("Starting ETL process...")
################################################
# Configure database connection
config = dict(dbname=os.getenv('POSTGRES_DB'), 
            user=os.getenv('POSTGRES_USER'),
            password=os.getenv('POSTGRES_PASSWORD'),
            host=os.getenv('POSTGRES_HOST'),
            #port=os.getenv('POSTGRES_PORT'))
            port=5432)
# Configure directories  
tensor_local_dir = os.getenv('IMAGES_DIR')           
tensors_base_dir = f"/home/images"
genomic_base_dir = f"/home/genomics"
data_csv  = f"/home/clinical_data/preprocessed_total_data.csv"
#################################################
# OMOP 
#################################################
# Get the current date and time
CURRENT_YEAR = datetime.today().year 
TODAY_date = datetime.today().strftime('%Y-%m-%d')
TODAY_datetime = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)

# OMOP 5.4 convention
NULL_CONCEPT_ID = 0	
NULL_SOURCE_CONCEPT_ID = 0	
NULL_SOURCE_VALUE = "null" 
DEFAULT_SOURCE_VALUE = None
#PERSON_SOURCE_VALUE = "value as in source"
GENDER_SOURCE_CONCEPT_ID = 0 

# Queries for inserting data into the database
patients_query = """
                    INSERT INTO person (person_id, gender_concept_id, year_of_birth, race_concept_id, ethnicity_concept_id,
                    person_source_value, gender_source_value, gender_source_concept_id, race_source_value, race_source_concept_id,
                    ethnicity_source_value, ethnicity_source_concept_id) VALUES %s ;
                    """ 
single_patient_query = """
                        INSERT INTO person (person_id, gender_concept_id, year_of_birth, race_concept_id, ethnicity_concept_id,
                        person_source_value, gender_source_value, gender_source_concept_id, race_source_value, race_source_concept_id,
                        ethnicity_source_value, ethnicity_source_concept_id) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
                        """ 
visit_occurrence_query = """ INSERT INTO visit_occurrence (person_id, visit_concept_id, visit_start_datetime, visit_end_datetime, visit_type_concept_id) 
                    VALUES %s RETURNING visit_occurrence_id;"""
                    
single_visit_query = """ INSERT INTO visit_occurrence (person_id, visit_concept_id, visit_start_datetime, visit_end_datetime, visit_type_concept_id) 
                    VALUES (%s,%s,%s,%s,%s) RETURNING visit_occurrence_id; """
                    
observation_query =""" 
                    INSERT INTO observation (person_id, observation_concept_id, observation_datetime, observation_type_concept_id,visit_occurrence_id,
                    value_as_number, value_as_string,value_as_concept_id, qualifier_concept_id, unit_concept_id, observation_source_value, 
                    observation_source_concept_id,unit_source_value, qualifier_source_value) VALUES %s;
                    """ # value_source_value,
measurement_query = """ 
                    INSERT INTO measurement (person_id, measurement_concept_id, measurement_datetime, measurement_type_concept_id,visit_occurrence_id, 
                    value_as_number,value_as_concept_id,unit_concept_id, measurement_source_value, 
                    measurement_source_concept_id,value_source_value, unit_source_value) VALUES %s; 
                    """ #qualifier_concept_id, qualifier_source_value
procedure_occur_query = """
                            INSERT INTO procedure_occurrence (person_id, procedure_concept_id,visit_occurrence_id) VALUES (%s, %s, %s)
                            RETURNING procedure_occurrence_id
                        """
                        
image_occurrence_query = """
        INSERT INTO image_occurrence(
        procedure_occurrence_id,
        filename,
        seg_mask, 
        feature_vector,
        patient_label,
        highlights, 
        note)
        VALUES(%s, %s, %s, %s, %s, %s, %s)
        """

genomic_result_query = """
INSERT INTO genomic_result (
    procedure_occurrence_id,
    sample_date,
    VCF2matrix,
    VCF2matrix_version
)
VALUES (%s, %s, %s, %s)
RETURNING genomic_result_id;
"""

# Query to check if concept_name exists in concept table
concept_exists_query = """
SELECT concept_id FROM concept WHERE concept_name = %s;
"""

# Query to insert a new concept
insert_concept_query = """
INSERT INTO concept (
    concept_id,
    concept_name,
    domain_id,
    vocabulary_id,
    concept_class_id,
    standard_concept,
    concept_code,
    valid_start_date,
    valid_end_date,
    invalid_reason
)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
"""

# Query to insert into condition_occurrence
condition_occurrence_query = """
INSERT INTO condition_occurrence (
    condition_occurrence_id,
    person_id,
    condition_concept_id,
    condition_start_datetime,
    condition_type_concept_id,
    condition_status_concept_id,
    condition_source_concept_id
)
VALUES (%s, %s, %s, %s, %s, %s, %s)
RETURNING condition_occurrence_id;
"""

# Mapping of disease labels to concept_ids
LABEL_CONCEPT_ID_MAP = {
    "disease_1": 100,
    "disease_2": 101
}

insert_vcf2matrix_version = """ INSERT INTO VCF2matrix_versions ( VCF2matrix_version, VCF2matrix_config, VCF2matrix_bed ) VALUES (%s, %s, %s); """

        
# Value mappings (CLINICAL PROTOCOL)
person_mapping = {"person_id": None, 
                "gender_concept_id": None ,
                "year_of_birth": CURRENT_YEAR, #CURRENT_YEAR - p_age, 
                "race_concept_id": NULL_CONCEPT_ID,
                "ethnicity_concept_id": NULL_CONCEPT_ID,
                "person_source_value": DEFAULT_SOURCE_VALUE , 
                "gender_source_value": DEFAULT_SOURCE_VALUE, 
                "gender_source_concept_id": NULL_SOURCE_CONCEPT_ID,
                "race_source_value": NULL_SOURCE_VALUE,
                "race_source_concept_id": NULL_CONCEPT_ID,
                "ethnicity_source_value": NULL_SOURCE_VALUE, 
                "ethnicity_source_concept_id": NULL_SOURCE_CONCEPT_ID} 

visit_concept_id = 32026 # laboratory visit (clinical tables)
visit_concept_id = 0 # to define #raccolta immagini
visit_type_concept_id = 32817 #EHR
observation_type_concept_id = 32817 #EHR
measurement_type_concept_id = 32817 #EHR

visit_occurrence_mapping = {"person_id":None, 
                            "visit_concept_id": 0, 
                            "visit_start_datetime": TODAY_datetime,
                            "visit_end_datetime": TODAY_datetime,
                            "visit_type_concept_id": 32817
}

procedure_occurrence_mapping = {"person_id": None, 
                                "procedure_concept_id": None,
                                "visit_occurrence_id": None } 

observation_mapping = { 
                    "person_id": None, #NOT NULL
                    "observation_concept_id": None, #NOT NULL
                    "observation_datetime": TODAY_datetime, #NOT NULL
                    "observation_type_concept_id" :32817, # EHR
                    "visit_occurrence_id":None, 
                    "value_as_number": DEFAULT_SOURCE_VALUE,  #
                    "value_as_string": DEFAULT_SOURCE_VALUE, #
                    "value_as_concept_id": DEFAULT_SOURCE_VALUE,  #
                    "qualifier_concept_id":NULL_CONCEPT_ID, 
                    "unit_concept_id":NULL_CONCEPT_ID,
                    "observation_source_value": DEFAULT_SOURCE_VALUE, 
                    "observation_source_concept_id": NULL_CONCEPT_ID,
                    "unit_source_value": DEFAULT_SOURCE_VALUE, 
                    "qualifier_source_value": DEFAULT_SOURCE_VALUE
                    }  

measurement_mapping = {
                    "person_id": None, #NOT NULL
                    "measurement_concept_id": None, #NOT NULL
                    "measurement_datetime": TODAY_datetime, 
                    "measurement_type_concept_id" :32817, #EHR
                    "visit_occurrence_id": None, 
                    "value_as_number": DEFAULT_SOURCE_VALUE, #
                    "value_as_concept_id": DEFAULT_SOURCE_VALUE, #
                    "unit_concept_id":NULL_CONCEPT_ID,
                    "measurement_source_value": DEFAULT_SOURCE_VALUE, 
                    "measurement_source_concept_id": NULL_CONCEPT_ID,
                    "value_source_value": DEFAULT_SOURCE_VALUE,
                    "unit_source_value": DEFAULT_SOURCE_VALUE
                    } 


def remove_constrain(table_name, column_name):
    query= """
        ALTER TABLE {} ALTER COLUMN {} DROP NOT NULL
        """.format(table_name, column_name)
    return query
    
def person_id_exists(sample_id, cursor):
    cursor.execute("SELECT person_id FROM person WHERE person_source_value = %s", (sample_id,)) 
    fn = cursor.fetchone()
    print(f"Query result for {sample_id}: {fn}") #debug
    if fn:
        return fn[0]  # if true return the value
    return None

def series_name_exists(patient_series_name):
    try:
        # Connect to the PostgreSQL database (replace the parameters with your database credentials)
        connection = psycopg2.connect(**config)
        cursor = connection.cursor()
        
        cursor.execute(
            "SELECT * FROM image_occurrence WHERE filename LIKE %s", (f"%{patient_series_name}%",))
        series_list = cursor.fetchall()

        connection.commit()
        cursor.close()
        connection.close()
        
        return series_list
    except psycopg2.Error as e:
        print("series_name_exists ERROR")
        print("Error connecting to the database:", e)
        return []

def gen_sample_name_exists(gen_samples_name):
    try:
        # Connect to the PostgreSQL database (replace the parameters with your database credentials)
        connection = psycopg2.connect(**config)
        cursor = connection.cursor()
        
        cursor.execute(
            "SELECT * FROM genomic_result WHERE VCF2matrix LIKE %s", (f"%{gen_samples_name}%",))
        series_list = cursor.fetchall()

        connection.commit()
        cursor.close()
        connection.close()
        
        return series_list

    except psycopg2.Error as e:
        print("Error connecting to the database or creating tables:", e)

def remove_none_values(mapping):
    return {k: v for k, v in mapping.items() if v is not None}
def convert_nan_to_none(mapping):
    return {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in mapping.items()}


def string_to_int(string):
    hash_object = hashlib.sha256(string.encode())
    hash_hex = hash_object.hexdigest()
    return int(hash_hex[:16], 16) % (10**10)
    
def update_PERSON_TABLE(data_patients,map_):
    p_id_list = []
    data = []
    try:
        # Connect to the PostgreSQL database (replace the parameters with your database credentials)
        connection = psycopg2.connect(**config)
        # Create a cursor to interact with the database
        cursor = connection.cursor()
        cursor.execute("SELECT MAX(person_id) FROM person")
        max_id = cursor.fetchone()[0]
        
        for index, row in data_patients.iterrows():
            record = person_mapping.copy()
            for el in map_:
                record[el[0]] = row.loc[el[1]] 
                
            p_id = person_id_exists(record["person_source_value"], cursor)
            if p_id:
                p_id_list.append(p_id)
                print("Patient with ID {} already exists. Encoded as {}".format(record["person_source_value"],p_id)) 
            else: 
                cursor.execute("SELECT MAX(person_id) FROM person")                
                max_id = cursor.fetchone()[0]  
                if max_id is None:
                    max_id = 0   # initial default value

                record["person_id"] = max_id + index +1
                p_id_list.append(record["person_id"])
                data.append(tuple(record.values())) # add person to the database 
        
        if data:  
            print("Adding {} rows to person table".format(len(data)))
            print()
            execute_values(cursor, patients_query, data) # fetch=True)
        else:
            print("No data to be inserted.")

        connection.commit()
        cursor.close()
        connection.close()

    except psycopg2.Error as e:
        print("PERSON TABLE: Error connecting to the database or creating tables:", e)
        if len(data)==0:
            print("No persons to upload")
            
    return p_id_list


def add_vcf2matrix_version(vcf2matrix_version, vcf2matrix_config, vcf2matrix_bed):
    try:
        connection = psycopg2.connect(**config)
        cursor = connection.cursor()
        cursor.execute(insert_vcf2matrix_version, (vcf2matrix_version, vcf2matrix_config, vcf2matrix_bed))
        connection.commit()
        cursor.close()
        connection.close()
    except psycopg2.Error as e:
        print("VCF2MATRIX VERSION TABLE: Error connecting to the database or creating tables:", e)
        
    return vcf2matrix_version
        
def update_ONE_PERSON(mapping):
    try:
        # Connect to the PostgreSQL database (replace the parameters with your database credentials)
        connection = psycopg2.connect(**config)
        cursor = connection.cursor()
        record = person_mapping.copy()
        
        cursor.execute("SELECT MAX(person_id) FROM person")
        max_person_id = cursor.fetchone()[0]
        if max_person_id is None:
            max_person_id = 0  

        for el in mapping:
            record[el[0]] = el[1]
            
        p_id = person_id_exists(record["person_source_value"], cursor) 
        if p_id: # IF PERSON_ID ALREADY EXISTS 
            print("Patient with ID {} already exists. Encoded as {}".format(record["person_source_value"],p_id)) 
            record["person_id"] = p_id
        else: 
            record["person_id"] = max_person_id + 1 # PERSON_ID = MAX_ID +1
            data = tuple(record.values())
            cursor.execute(single_patient_query, data)
        
        connection.commit()
        cursor.close()
        connection.close()
        
    except psycopg2.Error as e:
        print("Error connecting to the database or creating tables:", e)
        
    return record["person_id"]

def update_ONE_VISIT(mapping):
    try:
        # Connect to the PostgreSQL database (replace the parameters with your database credentials)
        connection = psycopg2.connect(**config)
        cursor = connection.cursor()
        record = visit_occurrence_mapping.copy()
        
        #person_id e visit_concept_id # eventaulemente se si aggiungono altro da modificare con la coppia attributo-valore come in person
        record['person_id'] = mapping[0]
        record['visit_concept_id'] = mapping[1]
        data = tuple(record.values())
    
        cursor.execute(single_visit_query, data)
        visit_id = cursor.fetchone()[0]
        
        connection.commit()
        cursor.close()
        connection.close()
        
    except psycopg2.Error as e:
        print("Error connecting to the database or creating tables:", e)
        
    return visit_id

def update_VISIT_OCCURRENCE(p_id_list, visit_concept_id):
    try:
        # Connect to the PostgreSQL database (replace the parameters with your database credentials)
        connection = psycopg2.connect(**config)
        # Create a cursor to interact with the database
        cursor = connection.cursor()
        
        data = []
        v_id_list = []
        record = visit_occurrence_mapping.copy()
        record['visit_concept_id'] = visit_concept_id

        for p_id in p_id_list:
            record['person_id'] = p_id
            data.append(tuple(record.values()))
            
        if data:  
            print("Adding {} rows to visit_occurrence table".format(len(data)))
            print()
            v_id_list = execute_values(cursor, visit_occurrence_query, data, fetch=True) #to return visit_occurrence_id
        else: 
            print("p_id_list vuota.")

        connection.commit()
        cursor.close()
        connection.close()

    except psycopg2.Error as e:
        print("VISIT OCCURRENCE TABLE: Error connecting to the database or creating tables:", e)
            
    return v_id_list
            
def update_OBSERVATION_TABLE(data_patients,map_):
    
        try:
            # Connect to the PostgreSQL database (replace the parameters with your database credentials)
            connection = psycopg2.connect(**config)
            # Create a cursor to interact with the database
            cursor = connection.cursor()
            attribute_concept = ["observation_concept_id","observation_source_value","unit_concept_id","unit_source_value" ]
            data = []
            for index, row in data_patients.iterrows():
                record = observation_mapping.copy()
                for col in map_:
                    for el in col:
                        if el[0] in attribute_concept:
                            record[el[0]] = el[1]
                        else: #value
                            record[el[0]] = row[el[1]]
                    # Converti il vocabolario in una tupla e aggiungila a `data`
                    record["person_id"] = row["person_id"]  #string_to_int(record["person_source_value"])
                    record["visit_occurrence_id"] = row["visit_occurrence_id"]
                    record = convert_nan_to_none(record)

                    data.append(tuple(record.values()))
            
            execute_values(cursor, observation_query, data)
            
            connection.commit()
            cursor.close()
            connection.close()
            
        except psycopg2.Error as e:
            print("OBSERVATION TABLE: Error connecting to the database or creating tables:", e)     
            
def update_MEASUREMENT_TABLE(data_patients,map_):
    
        try:
            # Connect to the PostgreSQL database (replace the parameters with your database credentials)
            connection = psycopg2.connect(**config)
            
            # Create a cursor to interact with the database
            cursor = connection.cursor()
            #cursor.execute(remove_constrain("measurement", "measurement_source_concept_id")) # drop not null AGGIUNTO ALLA STRUTTURA 

            # attribute_concept + clinical fact
            attribute_concept = ["measurement_concept_id","measurement_source_value","unit_concept_id","unit_source_value" ]
            data = []
            for index, row in data_patients.iterrows():
                record = measurement_mapping.copy()
                for col in map_:
                    for el in col:
                        if el[0] in attribute_concept:
                            record[el[0]] = el[1]
                        else:
                            record[el[0]] = row[el[1]]
                    record["person_id"] = row["person_id"] 
                    record["visit_occurrence_id"] = row["visit_occurrence_id"][0] #tupla (id,)
                    record = convert_nan_to_none(record)
                    
                    data.append(tuple(record.values()))
                    
            execute_values(cursor, measurement_query, data) # fetch=True to return the inserted rows
            
            connection.commit()
            cursor.close()
            connection.close()
            
        except psycopg2.Error as e:
            print("MEASUREMENT TABLE: Error connecting to the database or creating tables:", e)  

def update_PROCEDURE_OCCURRENCE_TABLE(patient,procedure_concept_id,visit_occurrence_id):
    try:
            connection = psycopg2.connect(**config)
            cursor = connection.cursor()
            
            #riempimento
            patient_id = patient #string_to_int(patient)
            cursor.execute(procedure_occur_query, (patient_id, procedure_concept_id, visit_occurrence_id))
            procedure_occurrence_id = cursor.fetchone()[0]
            
            connection.commit()
            cursor.close()
            connection.close()
            return procedure_occurrence_id
            
    except psycopg2.Error as e:
        print("update_PROCEDURE_OCCURRENCE_TABLE ERROR")
        print("Error connecting to the database or creating tables:", e) 

def update_DIAGNOSTIC_IMAGES_TABLE(procedure_occurrence_id,img_tensor,img_mask,feature_vector,patient_label,highlights,note):
    
    try:
            connection = psycopg2.connect(**config)
            cursor = connection.cursor()            
            cursor.execute(image_occurrence_query, (procedure_occurrence_id,img_tensor, img_mask, feature_vector, patient_label,highlights,note))
                        
            connection.commit()
            cursor.close()
            connection.close()
            
    except psycopg2.Error as e:
        print("update_DIAGNOSTIC_IMAGES_TABLE ERROR")
        print("Error connecting to the database or creating tables:", e) 


def update_GENOMIC_RESULT_TABLE(procedure_occurrence_id, mapping_dict, experiment_metadata, vcf2matrix_path, vcf2matrix_version):
    sample_date = mapping_dict.get("sample_date")
    # Convert empty string to None (NULL) for DATE column
    if sample_date == '' or sample_date is None:
        sample_date = None

    try:
        connection = psycopg2.connect(**config)
        cursor = connection.cursor()
        cursor.execute(genomic_result_query, (procedure_occurrence_id, sample_date,vcf2matrix_path, vcf2matrix_version))
        genomic_result_id = cursor.fetchone()[0]
        connection.commit()
        cursor.close()
        connection.close()
        return genomic_result_id
    except psycopg2.Error as e:
        print("update_GENOMIC_RESULT_TABLE ERROR")
        print("Error connecting to the database or creating tables:", e) 


def add_condition_occurrence(person_id, label):
    """
    Adds a condition occurrence for a person based on a disease label.
    
    1. Queries the concept table to check if label exists as concept_name
    2. If not present, inserts with concept_id 100 for "disease_1" and 101 for "disease_2"
    3. Adds a row to condition_occurrence with person_id and the concept_id of the label
    
    Args:
        person_id: The person_id to associate with the condition
        label: The disease label (e.g., "disease_1", "disease_2")
    
    Returns:
        condition_occurrence_id if successful, None otherwise
    """
    try:
        connection = psycopg2.connect(**config)
        cursor = connection.cursor()
        
        # Check if concept_name exists in concept table
        cursor.execute(concept_exists_query, (label,))
        result = cursor.fetchone()
        
        if result:
            # Concept exists, get its concept_id
            concept_id = result[0]
            print(f"Concept '{label}' already exists with concept_id {concept_id}")
        else:
            # Concept does not exist, insert it
            if label in LABEL_CONCEPT_ID_MAP:
                concept_id = LABEL_CONCEPT_ID_MAP[label]
            else:
                # For unknown labels, generate a concept_id starting from 200
                cursor.execute("SELECT COALESCE(MAX(concept_id), 199) + 1 FROM concept WHERE concept_id >= 200")
                concept_id = cursor.fetchone()[0]
                print(f"Unknown label '{label}', assigning concept_id {concept_id}")
            
            # Insert the new concept
            cursor.execute(insert_concept_query, (
                concept_id,           # concept_id
                label,                # concept_name
                "Condition",          # domain_id
                "None",               # vocabulary_id
                "Clinical Finding",   # concept_class_id
                "",                   # standard_concept (nullable, use empty string)
                label,                # concept_code
                TODAY_date,           # valid_start_date
                "2099-12-31",         # valid_end_date
                ""                    # invalid_reason (nullable, use empty string)
            ))
            print(f"Inserted concept '{label}' with concept_id {concept_id}")
        
        # Get max condition_occurrence_id and increment
        cursor.execute("SELECT COALESCE(MAX(condition_occurrence_id), 0) + 1 FROM condition_occurrence")
        condition_occurrence_id = cursor.fetchone()[0]
        
        # Insert into condition_occurrence
        cursor.execute(condition_occurrence_query, (
            condition_occurrence_id,  # condition_occurrence_id
            person_id,            # person_id
            concept_id,           # condition_concept_id
            TODAY_datetime,       # condition_start_datetime
            0,                    # condition_type_concept_id
            0,                    # condition_status_concept_id
            0                     # condition_source_concept_id
        ))
        cursor.fetchone()  # consume the RETURNING result
        
        connection.commit()
        cursor.close()
        connection.close()
        
        print(f"Added condition_occurrence for person_id {person_id} with condition_concept_id {concept_id}")
        return condition_occurrence_id
        
    except psycopg2.Error as e:
        print("add_condition_occurrence ERROR")
        print("Error connecting to the database or creating tables:", e)
        return None


def update_clinical_tables(protocol):
    
    if not os.path.exists(data_csv):
        print()
        print(f"No clinical data to upload.")
        print()
        return  
    
    # Get data
    data_patients = pd.read_csv(data_csv, header=0, sep=";", skiprows=[1])
    
    print("READING DATA FROM {}".format(data_csv))
    print("...")

    # Add temp columns to csv file for value entries 
    for var, details in protocol.items():
        if "Concept_IDs" in details:
            concept_ids = details["Concept_IDs"]
            exp_data = data_patients[var]
            if "synonymous_values" in details:  
                synonyms = details["synonymous_values"]
                exp_data = data_patients[var].replace({k: v for v, synonyms in synonyms.items() for k in synonyms})
            value_to_concept = dict(zip(details["expected_values"], concept_ids)) #Mapping from expected to concept_id
            data_patients[var+"_Concept_ID"] = exp_data.map(value_to_concept) # Create <attribute>_Concep_ID column 

        # Verifico se ci sono dati sull'età e genero una colonna date 
        if (details.get("ETL", {}).get("Voce_tabella_target") == "year_of_birth") and (details['type']=="int"):
            data_patients["year_of_birth"] = CURRENT_YEAR - data_patients[var]

    person_map = [] #una sola mappa per csv    
    observations = [] #possono essere più di una modello ATTRIBUTE-VALUE pair
    measurements = [] #possono essere più di una modello ATTRIBUTE-VALUE pair
    
    # Clinical protocol map 
    for key, value in protocol.items():
        #FATTO
        if value.get("ETL", {}).get("Tabella_target") == "PERSON":
            #nome_variabile_csv = key
            #TARGET
            if "Voce_tabella_target" in value.get("ETL", {}):
                voce_tabella_target = value.get("ETL", {}).get("Voce_tabella_target") #Value Target
                person_map.append([voce_tabella_target, key])
                if "Concept_IDs" in value: 
                    person_map[next((i for i, item in enumerate(person_map) if item[0] == voce_tabella_target), None)][1] = key + "_Concept_ID"
                if (value.get("ETL", {}).get("Voce_tabella_target") == "year_of_birth") and (value['type']=="int"):
                    person_map[next((i for i, item in enumerate(person_map) if item[0] == "year_of_birth"), None)][1] = "year_of_birth"
            #SORGENTE
            if "Voce_tabella_sorgente" in value.get("ETL", {}):
                voce_tabella_sorgente = value.get("ETL", {}).get("Voce_tabella_sorgente") #Value Source
                person_map.append([voce_tabella_sorgente, key])
        #FATTO
        elif value.get("ETL", {}).get("Tabella_target") == "OBSERVATION":
            observation_map=[ ]
            #TARGET
            if "Voce_tabella_target" in value.get("ETL", {}): #obbligatorio (try-except)
                concept_ID=value.get("ETL", {}).get("Concept_ID")
                voce_tabella_target = value.get("ETL", {}).get("Voce_tabella_target") #Value Target
                observation_map.append([voce_tabella_target, concept_ID])
                
                voce_valore_target= value.get("ETL", {}).get("Voce_valore_target")
                if "Concept_IDs" in value: #obbligatorio (try-except)
                    #value_as_Concept_ID
                    observation_map.append([voce_valore_target, key + "_Concept_ID"])
                else:
                    #value_as_number / value_as_string ??
                    observation_map.append([voce_valore_target,key])
            
            #SORGENTE
            if "Voce_tabella_sorgente" in value.get("ETL", {}):
                voce_tabella_sorgente = value.get("ETL", {}).get("Voce_tabella_sorgente") #Value Source
                observation_map.append([voce_tabella_sorgente, key]) 
            if "Voce_valore_sorgente"in value.get("ETL", {}):
                voce_valore_sorgente = value.get("ETL", {}).get("Voce_valore_sorgente")
                observation_map.append([voce_valore_sorgente, key])
            
            # UNITÀ DI MISURA
            if "Voce_unità_target" in value.get("ETL", {}):
                voce_unità_target = value.get("ETL", {}).get("Voce_unità_target")
                unit_Concept_ID = value.get("unit_Concept_ID")
                observation_map.append([voce_unità_target, unit_Concept_ID]) 
            if "Voce_unità_sorgente" in value.get("ETL", {}):
                voce_unità_sorgente = value.get("ETL", {}).get("Voce_unità_sorgente")
                unit_of_measure = value.get("unit_of_measure")
                observation_map.append([voce_unità_sorgente, unit_of_measure])
        
            observations.append(observation_map)
        #FATTO
        elif value.get("ETL", {}).get("Tabella_target") == "MEASUREMENT":
            measurement_map=[ ]
            if "Voce_tabella_target" in value.get("ETL", {}): #obbligatorio (try-except)
                concept_ID=value.get("ETL", {}).get("Concept_ID")
                voce_tabella_target = value.get("ETL", {}).get("Voce_tabella_target") #Value Target
                measurement_map.append([voce_tabella_target, concept_ID])
                
                voce_valore_target= value.get("ETL", {}).get("Voce_valore_target")
                if "Concept_IDs" in value: #obbligatorio (try-except)
                    #value_as_Concept_ID
                    measurement_map.append([voce_valore_target, key + "_Concept_ID"])
                else:
                    #value_as_number
                    measurement_map.append([voce_valore_target,key])
            
            #SORGENTE
            if "Voce_tabella_sorgente" in value.get("ETL", {}):
                voce_tabella_sorgente = value.get("ETL", {}).get("Voce_tabella_sorgente") #Value Source
                measurement_map.append([voce_tabella_sorgente, key]) 
            if "Voce_valore_sorgente"in value.get("ETL", {}):
                voce_valore_sorgente = value.get("ETL", {}).get("Voce_valore_sorgente")
                measurement_map.append([voce_valore_sorgente, key])
            
            # UNITÀ DI MISURA
            if "Voce_unità_target" in value.get("ETL", {}):
                voce_unità_target = value.get("ETL", {}).get("Voce_unità_target")
                unit_Concept_ID = value.get("unit_Concept_ID")
                measurement_map.append([voce_unità_target, unit_Concept_ID]) 
            if "Voce_unità_sorgente" in value.get("ETL", {}):
                voce_unità_sorgente = value.get("ETL", {}).get("Voce_unità_sorgente")
                unit_of_measure = value.get("unit_of_measure")
                measurement_map.append([voce_unità_sorgente, unit_of_measure])
            
            measurements.append(measurement_map)
        
    #CARICAMENTO 
    visit_concept_id = 32026 # laboratory visit (clinical tables)
    if len(person_map) == 0:
        print("No mapping for PERSON table") # empty csv 
    else:
        p_id_list = update_PERSON_TABLE(data_patients,person_map) # update the table
        visit_id_list = update_VISIT_OCCURRENCE(p_id_list, visit_concept_id) # update visit_occurrence table
        data_patients["person_id"] = p_id_list 
        data_patients["visit_occurrence_id"] = visit_id_list
        
    if len(observations) == 0:
        print("No mapping for OBSERVATION table")
    else:
        update_OBSERVATION_TABLE(data_patients,observations) # carico il paziente se non esiste
        
    if len(measurements) == 0:
        print("No mapping for MEASUREMENT table")
    else:
        update_MEASUREMENT_TABLE(data_patients,measurements) # carico il paziente se non esiste
    
def update_images(protocol, series_name):        
    # protocol reading 
    img_protocol = protocol[series_name]["image"]
    omop_ETL = protocol["image_ETL"] # accepted concept ids 
    # series data
    image_name= img_protocol["type"]["selected"] # image modality 
    procedure_concept_id = int(next((k for k, v in omop_ETL.items() if image_name in v), 0))

    patient_label = img_protocol.get("patient_label", {}).get("labels")
    comment_a = img_protocol.get("patient_label", {}).get("comment")
    
    slice_labels =  img_protocol.get("slice_labels", {}).get("labels")
    comment_b = img_protocol.get("slice_labels", {}).get("comment")
    note = ""
    if patient_label: 
        if comment_a:
            note += "Patient Labels : {}; Comment: {} ".format(patient_label,comment_a)
        else: 
            note += "Labels: {}" .format(patient_label)
    else: 
        note = "" # "null"
        
    if slice_labels: 
        if comment_b:
            note += "Slice Labels : {}; Comment: {} ".format(slice_labels,comment_b)
        else: 
            note += "Labels: {} ".format(slice_labels)
    else: 
        note = ""
    
    patient_ids = [p for p in os.listdir(tensors_base_dir) if p.startswith("patient")]
    
    tensor_paths = glob.glob(os.path.join(tensors_base_dir,"patient*", series_name+"*", "tensor*.npy"))
    mask_paths = glob.glob(os.path.join(tensors_base_dir,"patient*", series_name+"*", "seg*","seg*.npy" ))
    vector_paths = glob.glob(os.path.join(tensors_base_dir, "patient*", series_name+"*", "features*.npy"))
    patient_label_files = glob.glob(os.path.join(tensors_base_dir, "patient*", series_name+"*", "patient_label*.txt"))
    result_csv_files =glob.glob(os.path.join(tensors_base_dir, "patient*", series_name+"*", "labels*.csv"))
    
    #tensor_paths = [path.replace(tensors_base_dir, tensor_local_dir) for path in tensor_paths]
    #mask_paths = [path.replace(tensors_base_dir, tensor_local_dir) for path in mask_paths]
    #patient_label_files = [path.replace(tensors_base_dir, tensor_local_dir) for path in patient_label_files]
    #result_csv_files = [path.replace(tensors_base_dir, tensor_local_dir) for path in result_csv_files]

    for patient in patient_ids:
        # New procedure for the patient
        mapping = [["person_source_value", patient],["gender_concept_id", 0],["gender_source_value","null"], ["year_of_birth",  0]]  # Default values 
        person_id = update_ONE_PERSON(mapping)
        patient_series_name = f"{patient}/{series_name}"
        # Check if the series already exists for the patient
        if series_name_exists(patient_series_name):
            print(f"Series {series_name} already exists for patient {patient}. Skipping update.")
            continue
        
        visit_concept_id = 0 # to define #raccolta immagini 
        visit = [person_id, visit_concept_id] 
        visit_id = update_ONE_VISIT(visit)
        
        temp_path = [t for t in tensor_paths if f"{patient}/" in t]
        temp_mask =  [m for m in mask_paths if f"{patient}/" in m]
        temp_vector = [v for v in vector_paths if f"{patient}/" in v] #[v.replace("tensor","features") for v in temp_path] 
        highlights = [r for r in result_csv_files if f"{patient}/" in r]
        temp_patient_label =  [l for l in patient_label_files if f"{patient}/" in l] 

        #print(f"Length of temp_path : {len(temp_path)}, ") #debug
        #print(f"Length of temp_mask : {len(temp_mask)}, ") #debug
        if len(temp_patient_label) != 0:
            with open(temp_patient_label[0], "r") as f:
                temp_patient_label = f.readline().strip()
        
        if len(temp_path) != 0: 
            print("Updating {} image series for patient {}".format(len(temp_path), patient))
            # check if masks
            if len(temp_mask) != 0: 
                print("Mask found for image series")
            else: 
                temp_mask = "null"
                print("No mask for image series")
            # check if feature vectors
            if len(temp_vector) != 0: 
                print("Feature vectors found for image series")
            else: 
                temp_vector = "null"
                print("No feature vectors for image series")
            #check if labels
            if len(temp_patient_label) !=0: 
                print("Classification labels found for image series")
            else: 
                temp_patient_label = "null"
                print("No classification labels found for image series")
                
            if len(highlights) != 0: 
                print("Labels per slice found for image series")
            else: 
                highlights = "null"
                print("No labels per slice for image series")
                
            #update tables
            # image_concept_id = procedure_concept_id
            procedure_occurrence_id = update_PROCEDURE_OCCURRENCE_TABLE(person_id,procedure_concept_id, visit_id) # add procedure occurrence 
            for ind, el in enumerate(temp_path): 
                print("Adding image tensor: {}".format(el))
                update_DIAGNOSTIC_IMAGES_TABLE(procedure_occurrence_id,temp_path[ind], temp_mask[ind], temp_vector[ind], temp_patient_label,highlights, note) # classification task,  
        else: 
            print("No images for patient {}".format(patient))

def update_genomic_tables(protocol):
    print("Updating genomic data")

    vcf2matrix_version = add_vcf2matrix_version(protocol['genomic data']['VCF2Matrix_version'],
                             protocol['genomic data']['VCF2matrix_config'],
                             protocol['genomic data']['VCF2matrix_bed'])
    

    patient_ids = [p for p in os.listdir(genomic_base_dir) if p.startswith("patient")]
    for patient in patient_ids:
        mapping = [["person_source_value", patient],["gender_concept_id", 0],["gender_source_value","null"], ["year_of_birth",  0]]  # Default values 
        person_id = update_ONE_PERSON(mapping)
        # For each patient, iterate over all genomic sample subfolders in genomic_base_dir/{patient}
        patient_folder = os.path.join(genomic_base_dir, patient)
        if not os.path.isdir(patient_folder):
            continue

        for genomic_sample_name in os.listdir(patient_folder):
            sample_path = os.path.join(patient_folder, genomic_sample_name)
            # Consider only directories as genomic samples
            if not os.path.isdir(sample_path):
                continue

            # e.g. patient_gen_sample_name = "patient001/sampleA"
            patient_gen_sample_name = f"{patient}/{genomic_sample_name}"


            if gen_sample_name_exists(patient_gen_sample_name):
                print(f"Genomic sample {genomic_sample_name} already exists for patient {patient}. Skipping update.")
                continue

            # Create a visit for this genomic sample (raccolta dati genomici)
            visit_concept_id = 1  # to define genomic data collection
            visit = [person_id, visit_concept_id]
            visit_id = update_ONE_VISIT(visit)

            # Path of the file whose name starts with 'genomic_data' inside this genomic sample folder
            genomic_files = glob.glob(os.path.join(sample_path, "genomic_data*"))
            if not genomic_files:
                print(f"No genomic_data file found in sample {genomic_sample_name} for patient {patient}. Skipping sample.")
                continue
            vcf2matrix_path = genomic_files[0]

            # Path of the file whose name starts with 'experiment_metadata'
            exp_meta_files = glob.glob(os.path.join(sample_path, "experiment_metadata*"))
            if not exp_meta_files:
                print(f"No experiment_metadata file found in sample {genomic_sample_name} for patient {patient}. Skipping sample.")
                continue
            experiment_metadata_path = exp_meta_files[0]

            # Path of the file whose name starts with 'mapping_'
            mapping_files = glob.glob(os.path.join(sample_path, "mapping_*"))
            if not mapping_files:
                print(f"No mapping_ file found in sample {genomic_sample_name} for patient {patient}. Skipping sample.")
                continue
            mapping_sample_path = mapping_files[0]

            # Path of the file whose name starts with 'patient_label'
            patient_label_files = glob.glob(os.path.join(sample_path, "patient_label*"))
            if not patient_label_files:
                print(f"No patient_label file found in sample {genomic_sample_name} for patient {patient}. Skipping sample.")
                continue
            patient_label_path = patient_label_files[0]

            # Read patient-level label from text file (first line)
            with open(patient_label_path, "r") as f:
                label = f.readline().strip()

            # Read experiment metadata JSON
            with open(experiment_metadata_path, "r") as f:
                experiment_metadata = json.load(f)

            # Read mapping CSV as a single-row dict: column_name -> value
            with open(mapping_sample_path, newline="") as f:
                reader = csv.DictReader(f)
                mapping_row = next(reader, None)
            if mapping_row is None:
                print(f"Mapping file {mapping_sample_path} is empty for sample {genomic_sample_name}, patient {patient}. Skipping sample.")
                continue
            mapping_dict = dict(mapping_row)

            procedure_concept_id = 102 # Define a procedure_concept for genomic data collection (omop concept id)

            procedure_occurrence_id = update_PROCEDURE_OCCURRENCE_TABLE(person_id,procedure_concept_id, visit_id) # add procedure occurrence 
            update_GENOMIC_RESULT_TABLE(procedure_occurrence_id,
                            mapping_dict,
                            experiment_metadata,
                            vcf2matrix_path,
                            vcf2matrix_version)

            # Store patient label as condition occurrence
            add_condition_occurrence(person_id, label)






            # Further processing of this genomic sample will go here,
            # using patient_gen_sample_name and sample_path as needed.
    

    
if __name__ == "__main__":
    
    protocol_path = os.path.join("/home/protocol", "protocol.json")
    if not os.path.exists(protocol_path):
        raise FileNotFoundError(f"File not found: {protocol_path}")
    try:
        with open(protocol_path, 'r') as f:
            protocol = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON PARSING ERROR: {e}")

    # Clinical data table: check and update
    if protocol.get('clinical data'):  
        print("Updating clinical data")
        print()
        update_clinical_tables(protocol['clinical data'])

    # Image data: check and update 
    series_list = [v for v in protocol.keys() if v.startswith("series")]
    print(len(series_list), " image series to upload ")
    print()
    for series in series_list: 
        if protocol.get(series):  
            print("Updating image series:", series)
            print()
            update_images(protocol, series) #(series_fields, series_name)         
            
    # Genomic data
    if protocol.get('genomic data'):  
        print("Updating genomic data")
        print()
        update_genomic_tables(protocol)
























