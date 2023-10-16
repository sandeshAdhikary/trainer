FROM nvcr.io/nvidia/cuda:11.8.0-devel-ubuntu18.04
RUN 'debconf debconf/frontend select Noninteractive' | debconf-set-selections
# Install system dependencies
RUN apt-get update \
    && apt-get install -y \
    vim \
    # General dependencies
    wget \
    git \
    tmux \
    rsync \
    cron

# Set up working directory
RUN mkdir -p $HOME/project 
WORKDIR $HOME/project
COPY trainer_conda_env.yaml .

# Install Miniconda
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O Miniconda.sh && \
        /bin/bash Miniconda.sh -b -p /opt/conda && \
        rm Miniconda.sh
ENV PATH /opt/conda/bin:$PATH

# Create a conda environment
RUN conda env create -f trainer_conda_env.yaml -v

# Install as package
COPY setup.py .
RUN pip install -e .

# Activate the conda env
RUN echo "source activate trainer" > ~/.bashrc