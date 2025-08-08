FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

WORKDIR /workdir

# copy all files
COPY . .

# install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    curl \
    ca-certificates

# install conda
ENV PATH=/opt/conda/bin:$PATH
RUN apt-get update && apt-get install -y wget bzip2 ca-certificates \
    && wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && /bin/bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh \
    && conda init \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# install python packages
RUN conda create -n mla python=3.10 \
    && conda run -n mla pip install -r requirements.txt \
    && conda run -n mla pip install --no-deps -e .

RUN git clone https://github.com/deepseek-ai/FlashMLA.git \
    && cd FlashMLA \
    && git checkout 41b611f \
    && conda run -n mla pip install .