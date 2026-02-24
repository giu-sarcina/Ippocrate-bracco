# Database For Federated Learning

IPPOCRATE promotes the adoption of the OMOP Common Data Model (CDM) for the standardization of various medical data. OMOP (Observational Medical Outcomes Partnership) CDM is an open, community-driven data model that standardizes observational health data, enabling consistent representation and reproducible analyses across institutions (https://ohdsi.github.io/CommonDataModel/cdm60.html). 

This page describes how to create and populate the database with all the required information for federated learning experiments. Here, we assume that you have already preprocessed your data through the Data Harmonization pipeline [].

# Database description
The OMOP tables used in this project are:

- **person TABLE**: personal information about the patients
    - *person_id*: PRIMARY KEY;
    - *gender_concept_id*: FOREIGN KEY;
       
- **visit_occurrence TABLE**: visit information
    - *visit_occurrence_id*: PRIMARY KEY;
    - *person_id*: FOREIGN KEY;
    - *visit_concept_id*: FOREIGN KEY;
    - *visit_start_datetime*
    - *visit_end_datetime*
    - *visit_type_concept_id*: FOREIGN KEY; 
    
- **procedure_occurrence TABLE**: activities and analysis on the patient with a diagnostic purpose.
    - *procedure_occurrence_id*: PRIMARY KEY;
    - *person_id*: FOREIGN KEY;
    - *procedure_concept_id*: FOREIGN KEY;
    - *visit_occurrence_id*: FOREIGN KEY;
      
- **measurement TABLE**: records of structured values (numerical or categorical) obtained through systematic and standardized examination or testing of a person.
    - *measurement_id*: PRIMARY KEY;
    - *person_id*: FOREIGN KEY; 
    - *measurement_concept_id*: FOREIGN KEY;
    - *measurement_type_concept_id*: FOREIGN KEY;
    - *unit_concept_id*: FOREIGN KEY;
    - *visit_occurrence_id*: FOREIGN KEY;
    - *measurement_source_value*
    - *value_as_concept_id*
    - *value_source_value*
    - *unit_source_value*

- **observation TABLE**:clinical facts about a person obtained in the context of examination, questioning or a procedure.
    - *observation_id*: PRIMARY KEY;
    - *person_id*: FOREIGN KEY; 
    - *observation_concept_id*: FOREIGN KEY;
    - *observation_type_concept_id*: FOREIGN KEY;
    - *unit_concept_id*: FOREIGN KEY;
    - *visit_occurrence_id*: FOREIGN KEY;
    - *observation_source_value*
    - *unit_source_value*
    - *value_as_concept_id*
    - *value_as_string*
    
### Additional Tables
Historically, the presence of imaging examinations has been identified in the OMOP common data model only as imaging procedure codes in the procedure_occurrence table. To support imaging experiments and link generated imaging measurements into the OMOP data model, we added a new imaging table:

- **image_occurrence TABLE**: records of diagnostic images of a person.
    - *image_occurrence_id*: PRIMARY KEY;
    - *procedure_occurence_id*: FOREIGN KEY; 
    - *filename*: local path to the image object file stored in .npy Numpy format, NOT NULL;
    - *seg_mask*: local path to the segmentation object file stored in .npy Numpy format, OPTIONAL;
    - *feature_vector*: local path to the features vector object file stored in .npy Numpy format, OPTIONAL;
    - *patient_label*: txt file containing diagnosis label for the patient,OPTIONAL;
    - *highlights*: CSV file containing slice labels for 3D images,OPTIONAL;
    - *note*: OPTIONAL;
      
The fields of the tables follow the OMOP CDM conventions.The concept_id is the code for the vocabulary feature being measured. The type_concept_id describes the provenance of the source that feature came from. The source_value describes the original name of the corresponding object. 

We populate the database tables automatically by parsing a protocol that manages the variables needed for each experiment.

# Standalone container installation 
The project provides a [Docker](https://docs.docker.com/)  installation procedure that sets up an image for creating database storage on the host system. The image is built on postgres:14-alpine, an official version of PostgreSQL 14 built on top of the Alpine Linux distribution.

Before starting, ensure that you have all dependencies and Docker installed (https://docs.docker.com/get-started/get-docker). 

By default, you need sudo permissions to run Docker commands on Linux. However, you can configure your system to allow non-root users to run Docker commands and access to the host filesystem by adding your user to the docker group. See manual for reference (https://docs.docker.com/engine/install/linux-postinstall).

## Download the project repository

Download the project repository by typing:
```shell
git clone https://github.com/giu-sarcina/IPPOCRATE_db.git
```
and then enter into the main directory:
```shell
cd ./IPPOCRATE_db
```

## Build the Docker image
In the project directory, you can build the docker image in your terminal by typing:  
```shell
docker build -t ippocratedb .
```
## Run the container 
Edit and customize the *.env* file in **exec_files/ConfigDB** directory with your environment variables:

```
#1. Database name
POSTGRES_DB=omop4ippocrate
#2. Enter the Database user
POSTGRES_USER=user
#3. Enter your Database password
POSTGRES_PASSWORD=password
#4. Host machine
POSTGRES_HOST=localhost
#5. Host port
POSTGRES_PORT=5432

#DEFINE THE DIRECTORIES TO YOUR DATA after the data harmonization pipeline 

CLINICAL_DATA_DIR=/local/clinical_data_path
IMAGES_DIR=/local/images_path
PROTOCOL_DIR=/local/protocol_path
OUTPUT_DIR=/local/output_path
```
Then run and enter the container by running 
```shell
./run_container.sh 
```
## Execute the container
You are now into *ippocratedb* in interactive mode. 
You can exit the container by typing:
```shell
exit
```
To enter again in interactive mode, type:
```shell
docker exec -it ippocratedb bash
```
Inside the Docker container you can configure the PostgreSQL database. 

## Initialize the database 
As *$POSTGRES_USER* you can enter the *$POSTGRES_DB* already created and explore it by using psql\SQL commands.
The database is now empty,create the OMOP schema by typing:
```shell
psql -U $POSTGRES_USER -d $POSTGRES_DB -c "\c omop4ippocrate" -c "CREATE SCHEMA omopcdm;"
```
In order to have the shell provided by OMOP v6.0, run the command:
```shell
psql -U $POSTGRES_USER -d $POSTGRES_DB -f "omopv6ddl/OHDSI_PostgreSQL_DDLs/OMOP_CDM_postgresql_ddl.txt"
```
Database is created with empty elements and with OMOP standards. Add the new tables and alter existing ones with: 
```shell
python3 ./exec_files/ConfigDB/ALTER_and_ADD_tables.py
```
Now let's fill the body of the database.
## Populate the database
Type the following command and upload images and tabular data to OMOP: 
```shell
python3 ./exec_files/PopulateDB/ETL.py
```
## Quick commands
```shell
psql -U $POSTGRES_USER -d $POSTGRES_DB -p $POSTGRES_PORT -c "\c omop4ippocrate" -c "CREATE SCHEMA omopcdm;"
psql -U $POSTGRES_USER -d $POSTGRES_DB -P $POSTGRES_PORT -f "/home/omopv6ddl/OHDSI_PostgreSQL_DDLs/OMOP_CDM_postgresql_ddl.txt"
python3 /home/exec_files/ConfigDB/ALTER_and_ADD_tables.py
python3 /home/exec_files/PopulateDB/ETL.py

```


# Docker Compose Installation 

To stick with real-world scenarios, in our federated models each client loads data from a local and containerized OMOP database. Thus, we connect the OMOP container with the NVFlare container via Docker Compose. 

First, make sure that you have built the image of `NVIDIA PyTorch 24.05`, as described in the *SETUP* section of `~/nvflare/README.md`.

Next, customize the `docker-compose.yml` with the correct folder paths to the data. Customize the container names according to the client (e.g., `ippocratedb_1/2` and `nvflare-client_1/2`).
<!--
TODO: consider not hardcoding the paths but getting them from .env

-->
Next, build the compose on each client via:
```
docker-compose up
```
Access the OMOP container in interactive mode:

```
docker exec -it ippocratedb_1/2 /bin/bash
```
and populate the database with the commands reported in the *Quick Commands* section.

After connecting the client container to the server (see `nvflare/README.md`), you can run the jobs.




