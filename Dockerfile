FROM datajoint/datajoint

MAINTAINER Fabian Sinz <sinz@bcm.edu>

WORKDIR /efish


RUN \
  pip install git+https://github.com/circstat/pycircstat.git && \
  pip install git+https://github.com/fabiansinz/pyrelacs.git


# Install pipeline
COPY . /efish

RUN \
  rm -rf figures/*

RUN \
  pip install -e .


