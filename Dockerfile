# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Multi-stage build using openenv-base
# This Dockerfile is flexible and works for both:
# - In-repo environments (with local OpenEnv sources)
# - Standalone environments (with openenv from PyPI/Git)
# The build script (openenv build) handles context detection and sets appropriate build args.

ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest

FROM ${BASE_IMAGE} AS kwok-builder

# Install dependencies needed for downloading
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Fetch latest versions and download binaries sequentially for stability
RUN set -x && \
    ARCH=$(uname -m) && \
    case "${ARCH}" in \
    x86_64) ARCH=amd64 ;; \
    aarch64) ARCH=arm64 ;; \
    esac && \
    # Get latest version tags
    K8S_VERSION=$(curl -sfL https://dl.k8s.io/release/stable.txt) && \
    KWOK_REPO=kubernetes-sigs/kwok && \
    KWOK_LATEST_RELEASE=$(curl -sf "https://api.github.com/repos/${KWOK_REPO}/releases/latest" | grep '"tag_name":' | sed -E 's/.*"([^"]+)".*/\1/') && \
    # Download tools sequentially using the -f (fail) flag for safety
    curl -sfL -o /usr/local/bin/kubectl "https://dl.k8s.io/release/${K8S_VERSION}/bin/linux/${ARCH}/kubectl" && \
    curl -sfL -o /usr/local/bin/kwok "https://github.com/${KWOK_REPO}/releases/download/${KWOK_LATEST_RELEASE}/kwok-linux-${ARCH}" && \
    curl -sfL -o /usr/local/bin/kwokctl "https://github.com/${KWOK_REPO}/releases/download/${KWOK_LATEST_RELEASE}/kwokctl-linux-${ARCH}" && \
    chmod +x /usr/local/bin/kubectl /usr/local/bin/kwok /usr/local/bin/kwokctl

# Pre-download Kubernetes components into kwokctl cache using a cache mount for speed
RUN --mount=type=cache,target=/root/.kwok/cache \
    set -x && \
    ARCH=$(uname -m) && \
    case "${ARCH}" in x86_64) ARCH=amd64 ;; aarch64) ARCH=arm64 ;; esac && \
    # Force download of required components
    KWOK_KUBE_VERSION=v1.33.0 kwokctl create cluster --name=pre-download-cache --runtime=binary || true && \
    # "Bake" the cache into a persistent directory so it's saved in the image layer
    mkdir -p /root/.kwok_baked && \
    cp -rp /root/.kwok/* /root/.kwok_baked/ && \
    # Verify we found the binaries
    find /root/.kwok_baked -name kube-apiserver

FROM ${BASE_IMAGE} AS builder

WORKDIR /app

# Ensure git is available (required for installing dependencies from VCS)
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# Build argument to control whether we're building standalone or in-repo
ARG BUILD_MODE=in-repo
ARG ENV_NAME=ko2cube_env

# Copy environment code (always at root of build context)
COPY . /app/env

# For in-repo builds, openenv is already vendored in the build context
# For standalone builds, openenv will be installed via pyproject.toml
WORKDIR /app/env

# Ensure uv is available (for local builds where base image lacks it)
RUN if ! command -v uv >/dev/null 2>&1; then \
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/uv && \
    mv /root/.local/bin/uvx /usr/local/bin/uvx; \
    fi

# Install dependencies using uv sync
# If uv.lock exists, use it; otherwise resolve on the fly
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
    uv sync --frozen --no-install-project --no-editable; \
    else \
    uv sync --no-install-project --no-editable; \
    fi

RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
    uv sync --frozen --no-editable; \
    else \
    uv sync --no-editable; \
    fi

# Final runtime stage
FROM ${BASE_IMAGE}

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Copy Kubernetes tools and baked artifacts from kwok-builder
COPY --from=kwok-builder /usr/local/bin/kubectl /usr/local/bin/kubectl
COPY --from=kwok-builder /usr/local/bin/kwok /usr/local/bin/kwok
COPY --from=kwok-builder /usr/local/bin/kwokctl /usr/local/bin/kwokctl
COPY --from=kwok-builder /root/.kwok_baked /root/.kwok

# Copy the virtual environment from builder
COPY --from=builder /app/env/.venv /app/.venv

# Copy the environment code
COPY --from=builder /app/env /app/env

# Set PATH to use the virtual environment
ENV PATH="/app/.venv/bin:$PATH"

# Set PYTHONPATH so 'import ko2cube' works correctly from the parent directory
ENV PYTHONPATH="/app:/app/env:$PYTHONPATH"

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the FastAPI server
# The module path is constructed to work with the /app/env structure
CMD ["sh", "-c", "cd /app/env && uvicorn server.app:app --host 0.0.0.0 --port 8000 --log-level debug"]
