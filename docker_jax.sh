#!/bin/bash

# Define the Docker image name and tag
DOCKER_IMAGE=""

# Ensure NVIDIA runtime and GPU access is correctly set
docker run --gpus all --runtime=nvidia \
    --memory=46g --memory-swap=46g \
    -it $DOCKER_IMAGE /bin/bash
