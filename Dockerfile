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

# Create the researcher user with specific uid, gid, and groups as root
RUN getent group users || groupadd -g 100 users && \
    groupadd -g 652 flair-users && \
    groupadd -g 999 docker && \
    useradd -d /home/researcher -u 3556 -g 100 -G flair-users,docker --create-home researcher

# Add researcher to the video group for GPU access
RUN usermod -aG video researcher

# Create the workspace directory as root and set correct ownership
RUN mkdir -p /workspace && chown -R researcher:users /workspace

# Default to the workspace directory
WORKDIR /workspace

# Install jaxmarl from source and any other dependencies as root
COPY . .
RUN pip install -e .

# Switch to researcher for non-root operations
USER researcher

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
