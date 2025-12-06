# Use the official Ubuntu 22.04 image as our base
FROM ubuntu:22.04

# Set the working directory for subsequent commands
WORKDIR /app

# Install System Dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    curl \
    ffmpeg \
    git \
    # Add openssh-client to enable git push (ssh protocol)
    openssh-client \
    # Clean up apt lists to reduce image size
    && rm -rf /var/lib/apt/lists/*

# Download and install the Pixi executable
RUN curl -fsSL https://pixi.sh/install.sh | bash

# Add Pixi's installation directory to the PATH for the current shell and all subsequent layers.
ENV PATH="/root/.pixi/bin:$PATH"

# Only copy pixi.toml needed for environment build
COPY pixi.toml ./

# Install Python dependencies using Pixi
RUN /root/.pixi/bin/pixi install
