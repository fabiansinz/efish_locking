FROM datajoint/datajoint

MAINTAINER Fabian Sinz <sinz@bcm.edu>

# install tools to compile
RUN \
  apt-get update && \
  apt-get install -y -q \
    build-essential && \
  apt-get update && \
  apt-get install  --fix-missing -y -q \
    autoconf \
    automake \
    libtool


# install HDF5 reader and rabbit-mq client lib
RUN pip install h5py

WORKDIR /efish


RUN \
  pip install git+https://github.com/circstat/pycircstat.git && \
  pip install git+https://github.com/fabiansinz/pyrelacs.git


# Install pipeline
COPY . /efish

RUN \
  rm -rf figures/* __pycache__ scripts/__pycache__

RUN \
  pip install -e .


