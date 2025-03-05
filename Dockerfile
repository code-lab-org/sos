FROM ubuntu:22.04

WORKDIR /opt
COPY . /opt

USER root
ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.10

RUN apt-get -o Acquire::Check-Valid-Until=false -o Acquire::Check-Date=false update -y

# Add necessary dependencies
RUN apt-get update && apt-get install -y \
    wget \
    unzip \
    software-properties-common \
    libgdbm-dev \
    libc6-dev \
    liblzma-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-dev \
    curl \
    gnupg2 \
    libnetcdf-dev \
    libhdf4-dev \
    libhdf5-dev \
    build-essential \
    zlib1g-dev \
    libcurl4-gnutls-dev \
    libssl-dev

# # Add deadsnakes PPA and install Python 3.10
# RUN add-apt-repository ppa:deadsnakes/ppa && apt-get update && apt-get install -y python3.10 python3.10-venv python3.10-dev python3-pip

# # Install GDAL
# RUN add-apt-repository ppa:ubuntugis/ppa && apt-get update
# RUN apt-get install -y gdal-bin libgdal-dev proj-bin proj-data

# # Set environment variables for GDAL
# ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
# ENV C_INCLUDE_PATH=/usr/include/gdal

# RUN python3 -m pip install --upgrade pip
# RUN python3 -m pip install -r /opt/requirements.txt
# RUN wget https://github.com/emmanuelgonz/nost-tools/archive/refs/heads/main.zip \
#     && unzip main.zip \
#     && cd nost-tools-main \
#     && python3 -m pip install -e .

#install AWS command line interface (CLI)
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
    && unzip awscliv2.zip \
    && ./aws/install -i /usr/local/aws-cli -b /usr/local/bin \
    && rm awscliv2.zip
# ENTRYPOINT ["/usr/local/bin/aws"]


# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /opt/miniconda.sh \
    && bash /opt/miniconda.sh -b -p /opt/miniconda \
    && rm /opt/miniconda.sh

# Add Miniconda to PATH
ENV PATH=/opt/miniconda/bin:$PATH

# Create and activate conda environment
RUN conda create --name sos python=${PYTHON_VERSION} -y \
    && conda clean -a -y

# Activate the conda environment and install dependencies
RUN /bin/bash -c "source activate sos \
    && pip install -r /opt/requirements.txt \
    && conda install xarray netcdf4 \
    && conda install conda-forge::rioxarray \
    # && conda install conda-forge::gdal \
    && wget https://github.com/emmanuelgonz/nost-tools/archive/refs/heads/main.zip \
    && unzip main.zip \
    && cd nost-tools-main \
    && pip install -e ."