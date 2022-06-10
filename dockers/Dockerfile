# supported are cuda and cpu
ARG HARDWARE=cuda

FROM nvcr.io/nvidia/pytorch:21.12-py3 as base_cuda
ARG PYPI_OPTIONS=""

ARG PYPI_OPTIONS=""

FROM continuumio/miniconda3:4.10.3p1 as base_cpu
# look for cpu torch releases
ARG PYPI_OPTIONS="-f https://download.pytorch.org/whl/cpu/torch_stable.html"

FROM base_${HARDWARE}

# Make Directories for config, caches and workdir
# and set permissions to also use them as non-root
RUN mkdir /workdir && \
    chmod -R 777 /workdir && \
    mkdir /.cache && \
    chmod -R 777 /.cache && \
    mkdir /.config && \
    chmod -R 777 /.config

# necessary to resolve dependencies and compile simpleitk
RUN apt update && \
    DEBIAN_FRONTEND="noninteractive" apt install -y bash build-essential cmake
RUN pip install --upgrade pip scikit-build

COPY . /workdir/medical-shape

RUN pip install /workdir/medical-shape ${PYPI_OPTIONS}

WORKDIR /workdir
