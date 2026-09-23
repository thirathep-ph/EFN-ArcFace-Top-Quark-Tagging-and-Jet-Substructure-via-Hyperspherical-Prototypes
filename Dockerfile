FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libhdf5-dev \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Editable install of core library
COPY setup.py README.md ./
COPY arcefn/ ./arcefn/
RUN pip install --no-cache-dir -e .

# Scripts for training and analysis
COPY scripts/ ./scripts/

# Default: show help
CMD ["python", "scripts/train_arcface.py", "--help"]
