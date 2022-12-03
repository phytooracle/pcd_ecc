FROM ubuntu:22.04

WORKDIR /opt
COPY . /opt

USER root
ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.10.6
RUN apt-get -o Acquire::Check-Valid-Until=false -o Acquire::Check-Date=false update -y

RUN apt-get update 
RUN apt-get install -y wget \
                       gdal-bin \
                       libgdal-dev \
                       libspatialindex-dev \
                       build-essential \
                       software-properties-common \
                       apt-utils \
                       libgl1-mesa-glx \
                       ffmpeg \
                       libsm6 \
                       libxext6 \
                       libffi-dev \
                       libbz2-dev \
                       zlib1g-dev \
                       #libreadline-gplv2-dev \
                       libncursesw5-dev \
                       libssl-dev \
                       libsqlite3-dev \
                       tk-dev \
                       libgdbm-dev \
                       libc6-dev \
                       liblzma-dev \
                       libsm6 \
                       libxext6 \
                       libxrender-dev \
                       libgl1-mesa-dev \
                       libprotobuf-dev \
                       protobuf-compiler \
                       cmake

# Download and extract Python sources
RUN cd /opt \
    && wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz \                                              
    && tar xzf Python-${PYTHON_VERSION}.tgz

# Build Python and remove left-over sources
RUN cd /opt/Python-${PYTHON_VERSION} \ 
    && ./configure --with-ensurepip=install \
    && make install \
    && rm /opt/Python-${PYTHON_VERSION}.tgz /opt/Python-${PYTHON_VERSION} -rf

# Install packages
RUN pip3 install --upgrade pip
RUN pip3 install "pybind11[global]"
RUN pip3 install -r /opt/requirements.txt
RUN apt-get install -y locales && locale-gen en_US.UTF-8
ENV LANG='en_US.UTF-8' LANGUAGE='en_US:en' LC_ALL='en_US.UTF-8'

# Install demeter
# RUN cd /opt \
#     wget https://data.cyverse.org/dav-anon/iplant/projects/phytooracle/software_installation/demeter.tar \
#     tar -xvf demeter.tar \
#     cd demeter \
#     /usr/local/bin/python3.10 -m pip install -e .

ENTRYPOINT [ "/usr/local/bin/python3.10", "/opt/extract_ecc.py" ]