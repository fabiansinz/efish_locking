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


# Build HDF5
RUN cd ; wget https://www.hdfgroup.org/ftp/HDF5/current/src/hdf5-1.8.17.tar.gz \
    && tar zxf hdf5-1.8.17.tar.gz \
    && mv hdf5-1.8.17 hdf5-setup \
    &&  cd hdf5-setup \
    && ./configure --prefix=/usr/local/ \
    &&  make -j 12 && make install \
    && cd  \
    && rm -rf hdf5-setup \
    && apt-get -yq autoremove \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*


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


