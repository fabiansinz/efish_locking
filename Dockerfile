FROM datajoint/datajoint

MAINTAINER Fabian Sinz <sinz@bcm.edu>

# install tools to compile
RUN \
  apt-get update && \
  apt-get install -y -q \
    build-essential && \
  apt-get update && \
  apt-get install -y git


# install HDF5 reader and rabbit-mq client lib
RUN pip install h5py

WORKDIR /efish

# downgrade to previous version
RUN pip install datajoint==0.6.1

RUN \
  pip install git+https://github.com/circstat/pycircstat.git && \
  pip install git+https://github.com/fabiansinz/pyrelacs.git && \
  pip install matplotlib_venn


# Install pipeline
COPY . /efish

RUN \
  rm -rf figures/* __pycache__ scripts/__pycache__

RUN \
  pip install -e .


