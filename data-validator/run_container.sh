#!/bin/bash
set -a && source ./.env && set +a # import all variables from .env file

export HOST_ROOT_DIR
export HOST_BASE_DATA_DIR

# Subfolders to mount
SCRIPT_DIR="$(dirname "$PWD")/script_files" 

docker run --rm -it --name data-validator \
   --env-file ./.env \
   -v "$HOST_BASE_DATA_DIR":/home/input_data \
   -v "$SCRIPT_DIR":/home/script_files \
   -v "$HOST_ROOT_DIR":/home/experiment \
   -e HOST_ROOT_DIR \
   -e HOST_BASE_DATA_DIR \
   data-validator 
   




