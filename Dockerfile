FROM ubuntu:18.04
  
ENV DEBIAN_FRONTEND=noninteractive
ENV DJANGO_SETTINGS_MODULE=website.settings
ENV C_FORCE_ROOT=true

WORKDIR /app

RUN apt-get update \
    && apt-get install -y git python3.6 python3-pip python3-tk \
       mysql-client libmysqlclient-dev python-mysqldb \
       openjdk-8-jdk vim openssh-client wget unzip 

WORKDIR /app/ottertune/server/website

ADD server /app/ottertune/server/

# FOR test
RUN mkdir /app/ottertune/script
COPY script /app/ottertune/script
RUN mkdir /app/ottertune/client
COPY client /app/ottertune/client

RUN pip3 install -r requirements.txt 

COPY ./docker/credentials.txt ./website/settings/credentials.py
COPY ./docker/start.sh /app/ottertune/server/website
COPY ./docker/createadmin.txt /app/ottertune/server/website/createadmin.py
COPY ./docker/wait-for-it.sh /app/ottertune/server/website
COPY ./docker/test.sh /app/ottertune/server/website

RUN chmod +x /app/ottertune/server/website/start.sh
RUN chmod +x /app/ottertune/server/website/test.sh

RUN sed s/'@localhost'/'@rabbitmq'/g ./website/settings/common.py > tmp \
    && mv tmp ./website/settings/common.py

VOLUME ["/app/ottertune/server/website/website/migrations"]
VOLUME ["/app/ottertune/server/website/media"]
ENTRYPOINT ["./start.sh"]

