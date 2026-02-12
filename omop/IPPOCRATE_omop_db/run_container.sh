#!/bin/bash
set -a && source ./exec_files/ConfigDB/.env && set +a  # import all variables from .env file

# Variable definition (host)
export BASE_DIR
export CLINICAL_DATA_DIR
export IMAGES_DIR
export PROTOCOL_DIR
export OUTPUT_DIR
export POSTGRES_PORT
export POSTGRES_USER

for var in CLINICAL_DATA_DIR IMAGES_DIR PROTOCOL_DIR OUTPUT_DIR; do
  dir_val="${!var}" 
  if [ -z "$dir_val" ]; then
    default_path="$BASE_DIR/${var,,}" 
    mkdir -p "$default_path"
    export $var="$default_path"
  fi
done

# DA LOCALE A CONTAINER 
echo "Creating database container"

home_dir="/home/$POSTGRES_USER"
exec="$home_dir/exec_files"
build="$home_dir/build_files"
ddl="$home_dir/omopv6ddl"
vocabulary_build="$home_dir/data/vocabulary"

# run the container in detached mode
docker run -d --name ippocratedb \
  --hostname omop6postgres14 \
  --workdir $home_dir \
  --env-file ./exec_files/ConfigDB/.env \
  -v $CLINICAL_DATA_DIR:$home_dir/clinical_data \
  -v $IMAGES_DIR:$home_dir/images \
  -v $PROTOCOL_DIR:$home_dir/protocol \
  -v $OUTPUT_DIR:$home_dir/output \
  -v $(realpath ./vocabulary):$vocabulary_build \
  -v $(realpath ./exec_files):$exec \
  -v $(realpath ./build_files):$build \
  -v $(realpath ./OMOPv6-PSQL-Scripts):$ddl \
  -e IMAGES_DIR \
  -e POSTGRES_USER \
  -e POSTGRES_PASSWORD \
  -e POSTGRES_DB \
  -p $POSTGRES_PORT:5432 \
  ippocratedb

# enter the container
docker exec -it ippocratedb bash
