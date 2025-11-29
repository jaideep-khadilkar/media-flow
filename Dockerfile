# Use the official Ubuntu 22.04 image as our base
FROM ubuntu:22.04

# Set the working directory for subsequent commands
WORKDIR /app

# 1. Install System Dependencies (Minimal Python 3, pip, curl, FFmpeg, Git, and SSH)
# Note: This installs the default Python 3 (3.10 on Ubuntu 22.04)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    curl \
    ffmpeg \
    git \
    # *** FIX: Add openssh-client to enable git push (ssh protocol) ***
    openssh-client \
    # Clean up apt lists to reduce image size
    && rm -rf /var/lib/apt/lists/*

# 2. Install Pixi
# Downloads and installs the Pixi executable
RUN curl -fsSL https://pixi.sh/install.sh | bash

# Add Pixi's installation directory to the PATH for the current shell and all subsequent layers.
ENV PATH="/root/.pixi/bin:$PATH"

# 3. Copy project files
# Only copy pixi.toml needed for environment build
COPY pixi.toml ./

# 4. Install Python dependencies using Pixi (This step forces 3.12)
RUN /root/.pixi/bin/pixi install

# 5. Copy the rest of the application code (main.py and any other files)
COPY . /app

# 6. Set the default command when the container starts
CMD ["pixi", "run", "start"]