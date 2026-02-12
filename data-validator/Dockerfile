# Base image
FROM python:3.11

ARG UID=1234
ARG GID=1234
ARG USERNAME=customuser
 
# Crea gruppo e utente con gli stessi UID/GID dell'host
RUN groupadd -g $GID $USERNAME && \
    useradd -m -u $UID -g $GID -s /bin/bash $USERNAME 

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
 && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV ROOT_NAME=/home/experiment \
    PROTOCOL_DIR=/home/experiment/PROTOCOL \
    CONFIG_DIR=/home/experiment/LOCALCONFIG \
    MAPPINGFILE_DIR=/home/experiment/MAPPINGFILE \
    TAGLIST_DIR=/home/experiment/TAGLIST \
    INPUT_DIR=/home/input_data

# Copy files
COPY ./requirements.txt /home/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r /home/requirements.txt

# Set the working directory 
WORKDIR /home/experiment

# RUN chmod -R 777 /home/experiment

# USER $USERNAME	



# CMD ["python", "-c", "import time; time.sleep(5 * 60)"]

CMD ["python", "/home/script_files/main.py"]



