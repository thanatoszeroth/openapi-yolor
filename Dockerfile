##############################
FROM nvcr.io/nvidia/pytorch:20.11-py3

### ENV
ENV DEBIAN_FRONTEND=noninteractive

### Localtime
ENV LC_ALL C.UTF-8
ENV TimeZone=Asia/Taipei
RUN ln -snf /usr/share/zoneinfo/$TimeZone /etc/localtime && echo $TimeZone > /etc/timezone
RUN apt-get update -y --fix-missing && \
    apt-get install -y --no-install-recommends \
    tzdata && \
    dpkg-reconfigure --frontend noninteractive tzdata

### Install Python Package
COPY requirements.txt /tmp/requirements.txt
RUN apt-get -y update && \
    pip3 --no-cache-dir install -r /tmp/requirements.txt && \
    apt-get clean autoclean && \
    apt-get autoremove -y && \
    rm -rf /var/lib/{apt,dpkg,cache,log}/

COPY ./app /app
WORKDIR "/app"

COPY docker-entrypoint.sh /usr/local/bin/ 
RUN chmod 777 /usr/local/bin/docker-entrypoint.sh

##############################
LABEL Master=ThanatosHsiao Version=2022.09
##############################




