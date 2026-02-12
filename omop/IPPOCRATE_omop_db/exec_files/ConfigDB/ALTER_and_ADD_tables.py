import os
import psycopg2

################################################
# Configure database connection
config = dict(dbname=os.getenv('POSTGRES_DB'), 
            user=os.getenv('POSTGRES_USER'),
            password=os.getenv('POSTGRES_PASSWORD'),
            host=os.getenv('POSTGRES_HOST'),
            # port=os.getenv('POSTGRES_PORT'))
            port=5432)
#################################################
# Due to the similarity between lipidomic and proteomic tables, we define the following data types to create the tables with a for loop
analysis_type = {"g": "genomic", "r": ["image_occurrence"]}
omic_types = ['lipidomic', 'proteomic']
#################################################

# Customize existing tables
def alter_clinical_tables_omopv6(cursor):            
    """
    """
    # DROP NOT NULL constraints (mainly introduced not officially in v6)
    #cursor.execute(remove_constrain("observation", "observation_source_concept_id")) # drop not null
    #cursor.execute(remove_constrain("observation", "obs_event_field_concept_id")) # drop not null
    cursor.execute("""
    
    --1.  Set visit_occurrence_id column to SERIAL
    ALTER TABLE visit_occurrence DROP COLUMN visit_occurrence_id;
    ALTER TABLE visit_occurrence ADD visit_occurrence_id SERIAL NOT NULL PRIMARY KEY;
    
    -- DROP NOT NULL constraint from some columns of 'visit_occurrence' table
    ALTER TABLE visit_occurrence ALTER COLUMN visit_concept_id DROP NOT NULL;
    ALTER TABLE visit_occurrence ALTER COLUMN visit_type_concept_id DROP NOT NULL;
    ALTER TABLE visit_occurrence ALTER COLUMN visit_source_concept_id DROP NOT NULL;
    ALTER TABLE visit_occurrence ALTER COLUMN admitted_from_concept_id DROP NOT NULL;
    ALTER TABLE visit_occurrence ALTER COLUMN discharge_to_concept_id DROP NOT NULL;
    
    --2.  Set procedure_occurrence_id column to SERIAL
    ALTER TABLE procedure_occurrence DROP COLUMN procedure_occurrence_id;
    ALTER TABLE procedure_occurrence ADD procedure_occurrence_id SERIAL NOT NULL PRIMARY KEY;
    
    -- DROP NOT NULL constraint from some columns of 'procedure_occurrence' table
    ALTER TABLE procedure_occurrence ALTER COLUMN procedure_datetime DROP NOT NULL;
    ALTER TABLE procedure_occurrence ALTER COLUMN procedure_type_concept_id DROP NOT NULL;
    ALTER TABLE procedure_occurrence ALTER COLUMN modifier_concept_id DROP NOT NULL;
    ALTER TABLE procedure_occurrence ALTER COLUMN procedure_source_concept_id DROP NOT NULL;
    
    --3.  DROP NOT NULL constraint from some columns of 'concept' table
    ALTER TABLE concept ALTER COLUMN domain_id DROP NOT NULL;
    ALTER TABLE concept ALTER COLUMN vocabulary_id DROP NOT NULL;
    ALTER TABLE concept ALTER COLUMN concept_class_id DROP NOT NULL;
    ALTER TABLE concept ALTER COLUMN concept_code DROP NOT NULL;
    ALTER TABLE concept ALTER COLUMN valid_start_date DROP NOT NULL;
    ALTER TABLE concept ALTER COLUMN valid_end_date DROP NOT NULL;
    
    --4.  DROP NOT NULL constraints from some columns of "person"
    ALTER TABLE person ALTER COLUMN race_concept_id DROP NOT NULL;
    ALTER TABLE person ALTER COLUMN ethnicity_concept_id DROP NOT NULL;
    ALTER TABLE person ALTER COLUMN gender_source_concept_id DROP NOT NULL;
    ALTER TABLE person ALTER COLUMN race_source_concept_id DROP NOT NULL;
    ALTER TABLE person ALTER COLUMN ethnicity_source_concept_id DROP NOT NULL;
    
    --5.  Set measurement_id  and observation_id column to SERIAL
    ALTER TABLE measurement DROP COLUMN measurement_id;
    ALTER TABLE measurement ADD measurement_id SERIAL NOT NULL PRIMARY KEY;
    
    ALTER TABLE observation DROP COLUMN observation_id;
    ALTER TABLE observation ADD observation_id SERIAL NOT NULL PRIMARY KEY;
    
    --6 Alter measurement and observation tables 
    ALTER TABLE measurement ALTER COLUMN measurement_source_concept_id DROP NOT NULL; 
    ALTER TABLE observation ALTER COLUMN observation_source_concept_id DROP NOT NULL; 
    ALTER TABLE observation ALTER COLUMN obs_event_field_concept_id DROP NOT NULL; 

    """)
        
def load_new_tables(cursor):
    # SQL statements to create proteomic and lipidomic tables
        for omic in omic_types:
            # RESULTS table
            cursor.execute("""CREATE TABLE {omic_type}_result(
                {omic_type}_result_id BIGSERIAL NOT NULL PRIMARY KEY,
                procedure_occurrence_id BIGINT NOT NULL,
                sample_date DATE NOT NULL, -- Consider adding it to the 'specimen' table
                {omic_type}_exp_id INTEGER NOT NULL,
                {omic_type}_batch_id INTEGER NOT NULL,
                {omic_type}_file VARCHAR(150) NOT NULL, -- path to the file
                {omic_type}_list VARCHAR(30) ARRAY -- list of lipids revealed in the analysis
            );""".format(omic_type=omic))
    
            # BATCH table
            cursor.execute("""CREATE TABLE {omic_type}_batch (
                {omic_type}_batch_id BIGSERIAL NOT NULL PRIMARY KEY,
                replicates SMALLINT,
                batch_effect_flag BIT NOT NULL
            ); """.format(omic_type=omic))
    
            # EXPERIMENT table
            cursor.execute("""CREATE TABLE {omic_type}_exp (
                {omic_type}_exp_id BIGSERIAL NOT NULL PRIMARY KEY,
                {omic_type}_batch_list INTEGER ARRAY,
                operator VARCHAR(50) NOT NULL,
                organization VARCHAR(50) NOT NULL,
                machine_name VARCHAR(50) NOT NULL,
                note TEXT
            );""".format(omic_type=omic))
    
            # ALTER tables
            cursor.execute("""
                ALTER TABLE {omic_type}_result ADD FOREIGN KEY (procedure_occurrence_id) REFERENCES procedure_occurrence(procedure_occurrence_id);
                ALTER TABLE {omic_type}_result ADD FOREIGN KEY ({omic_type}_exp_id) REFERENCES {omic_type}_exp({omic_type}_exp_id);
                ALTER TABLE {omic_type}_result ADD FOREIGN KEY ({omic_type}_batch_id) REFERENCES {omic_type}_batch({omic_type}_batch_id);
                """.format(omic_type=omic))

        # Create genomic_result table
        cursor.execute("""CREATE TABLE {g}_result(
                        {g}_result_id BIGSERIAL NOT NULL PRIMARY KEY,
                        procedure_occurrence_id BIGINT NOT NULL,
                        sample_date DATE, -- Consider storing this in the 'specimen' table
                        {g}_exp_id INTEGER,
                        {g}_batch_id INTEGER,
                        VCF_raw VARCHAR(150),  -- path to the raw VCF
                        VCF_annotate VARCHAR(150),  -- path to the annotated VCF
                        pipeline_version VARCHAR(50),
                        VCF2matrix VARCHAR(150) NOT NULL,  -- path to the VCF2matrix file
                        VCF2matrix_version VARCHAR(50) NOT NULL,
                        VCF_old  VARCHAR(150) ARRAY -- ARRAY of pairs (file_path, pipeline_version) to keep track of old annotate VCF 
                        );""".format(g=analysis_type["g"]))              #[][]

        # Create genomic_batch table
        cursor.execute("""CREATE TABLE {g}_batch (
                    {g}_batch_id BIGSERIAL NOT NULL PRIMARY KEY,
                    batch_effect_flag BIT
                    );""".format(g=analysis_type["g"]))

        # Create genomic_exp table
        cursor.execute("""CREATE TABLE {g}_exp (
                        {g}_exp_id BIGSERIAL NOT NULL PRIMARY KEY,
                        {g}_batch_list INTEGER ARRAY,
                        operator VARCHAR(50),
                        organization VARCHAR(50),
                        machine_name VARCHAR(50),
                        note TEXT
                        );""".format(g=analysis_type["g"]))

        # Create VCF2matrix_versions table
        cursor.execute("""CREATE TABLE VCF2matrix_versions (
                        VCF2matrix_vers_id BIGSERIAL NOT NULL PRIMARY KEY,
                        VCF2matrix_version VARCHAR(50) NOT NULL UNIQUE, -- version of the VCF2matrix tool
                        VCF2matrix_config VARCHAR(200),
                        VCF2matrix_bed VARCHAR(200)
                        );""")

        # ALTER tables
        cursor.execute("""
                        ALTER TABLE {g}_result ADD FOREIGN KEY (procedure_occurrence_id) REFERENCES procedure_occurrence(procedure_occurrence_id);
                        ALTER TABLE {g}_result ADD FOREIGN KEY ({g}_exp_id) REFERENCES {g}_exp({g}_exp_id);
                        ALTER TABLE {g}_result ADD FOREIGN KEY ({g}_batch_id) REFERENCES {g}_batch({g}_batch_id);
                        ALTER TABLE {g}_result ADD FOREIGN KEY (VCF2matrix_version) REFERENCES VCF2matrix_versions(VCF2matrix_version);
                        """.format(g=analysis_type["g"]))
            
        # Create Database Tables for Images
        for ri in analysis_type["r"]:
            # Create Database Tables for Images
            cursor.execute("""CREATE TABLE {r} (
                        {r}_id BIGSERIAL NOT NULL PRIMARY KEY,
                        procedure_occurrence_id BIGINT NOT NULL,
                        filename TEXT NOT NULL,
                        seg_mask TEXT,
                        feature_vector TEXT,
                        patient_label VARCHAR(50),
                        highlights TEXT, 
                        note TEXT);""".format(r=ri)
                        )
                        
            # Alter Images Tables
            cursor.execute("""ALTER TABLE {r} 
                        ADD FOREIGN KEY (procedure_occurrence_id) 
                        REFERENCES procedure_occurrence(procedure_occurrence_id)""".format(r=ri)
                        )               

def alter_concept_id_int(cursor):
    cursor.execute("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema='public' AND table_type='BASE TABLE';
    """)
    tables = [row[0] for row in cursor.fetchall()]

    for table in tables:
        cursor.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = %s AND column_name LIKE '%%_concept_id';
        """, (table,))
        columns_raw = cursor.fetchall()
        if not columns_raw:
            continue  # nessuna colonna da modificare
        columns = [row[0] for row in columns_raw]

        for col in columns:
            print(f"Modifico {table}.{col} in BIGINT...")
            cursor.execute(f"""
                ALTER TABLE {table}
                ALTER COLUMN {col} TYPE BIGINT;
            """)
            
            
if __name__ == "__main__":
    cursor = None
    connection = None
    try:
        # Connect to the PostgreSQL database (replace the parameters with your database credentials)
        connection = psycopg2.connect(**config)
        # Create a cursor to interact with the database
        cursor = connection.cursor()
        
        print("Connected to the database successfully.")
        
        alter_clinical_tables_omopv6(cursor)
        
        print("Altered existing clinical tables successfully.")
        print("Loading new tables...")
        
        load_new_tables(cursor)
        
        #alter_concept_id_int(cursor)
        
        connection.commit()
        
    except psycopg2.Error as e:
        print("Error connecting to the database or creating tables:", e)
        
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()