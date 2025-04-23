FROM ubuntu:24.04

# Set working directory
WORKDIR /opt
COPY . /opt

# Set environment variables
USER root
ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.9.21

# Update package list
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
    libssl-dev \
    libffi-dev

# Download and extract Python sources
RUN cd /opt \
    && wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz \                                              
    && tar xzf Python-${PYTHON_VERSION}.tgz

# Build Python and remove left-over sources
RUN cd /opt/Python-${PYTHON_VERSION} \ 
    && ./configure --enable-optimizations --with-ensurepip=install \
    && make install \
    && rm /opt/Python-${PYTHON_VERSION}.tgz /opt/Python-${PYTHON_VERSION} -rf

# Install Python packages
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install -r /opt/requirements.txt \
    && rm /opt/requirements.txt  

# Install AWS command line interface (CLI)
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
    && unzip awscliv2.zip \
    && ./aws/install -i /usr/local/aws-cli -b /usr/local/bin \
    && rm awscliv2.zip \
    && rm aws -rf