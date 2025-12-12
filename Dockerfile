FROM nvidia/cuda:12.9.0-devel-ubuntu22.04

WORKDIR /app

# Install System Dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    openssh-client \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install pixi directly into /usr/local/bin (already on PATH)
RUN curl -fsSL https://pixi.sh/install.sh | PIXI_BIN_DIR=/usr/local/bin bash

# # Copy env spec
# COPY pixi.toml ./

# # Install dependencies using pixi
# RUN CONDA_OVERRIDE_CUDA=12.9 pixi install
