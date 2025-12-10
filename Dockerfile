FROM ubuntu:22.04

WORKDIR /app

# Install System Dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    openssh-client \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install pixi directly into /usr/local/bin (already on PATH)
RUN curl -fsSL https://pixi.sh/install.sh | PIXI_BIN_DIR=/usr/local/bin PIXI_NO_PATH_UPDATE=1 bash

# Copy env spec
COPY pixi.toml ./

# Install dependencies using pixi
RUN pixi install
