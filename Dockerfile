FROM nvcr.io/nvidia/jax:23.10-py3

# Upgrade pip, setuptools, and wheel as root
RUN pip install --upgrade pip setuptools wheel

# Install necessary packages as root
RUN apt-get update && \
    apt-get install -y tmux \
    build-essential \
    python3-dev \
    libsdl2-dev \
    libsdl2-image-dev \
    libsdl2-mixer-dev \
    libsdl2-ttf-dev \
    libsmpeg-dev \
    libportmidi-dev \
    libswscale-dev \
    libavformat-dev \
    libavcodec-dev \
    libfreetype6-dev \
    pkg-config

# Default to the workspace directory
WORKDIR /workspace

# Copy source code including the requirements directory
COPY . .

# Install dependencies from requirements file
RUN pip install -r requirements/requirements.txt

# Add the workspace directory to PYTHONPATH so that jaxmarl is discoverable
ENV PYTHONPATH=/workspace:$PYTHONPATH

# Disabling preallocation and setting environment variables
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false
ENV XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
ENV TF_FORCE_GPU_ALLOW_GROWTH=true

# Set git safe directory for researcher
RUN git config --global --add safe.directory /workspace

# For secrets and debug
ENV WANDB_API_KEY=""
ENV WANDB_ENTITY=""

# Set default command
CMD ["/bin/bash"]
