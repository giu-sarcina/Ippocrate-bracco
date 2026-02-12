FROM postgres:14-alpine
# Base image for the IPPOCRATE database container

#import static files 
COPY ./build_files/requirements.txt ./build_files/requirements.txt

#Update and install base python packages
RUN apk update && apk upgrade && \
    apk add --no-cache vim zip python3 py3-pip py3-pandas && \
    apk add --no-cache dos2unix && \
    python3 -m pip install --break-system-packages --upgrade pip setuptools wheel && \
    python3 -m pip install --break-system-packages python-dotenv && \
    python3 -m pip install --break-system-packages -r ./build_files/requirements.txt && \
    python3 -c 'from PIL import Image' && \
    python3 -m pip list