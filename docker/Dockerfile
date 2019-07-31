FROM ubuntu:18.04
  
ENV DEBIAN_FRONTEND=noninteractive
ENV DJANGO_SETTINGS_MODULE=website.settings
ENV C_FORCE_ROOT=true

WORKDIR /app

RUN apt-get update \
    && apt-get install -y git python3.6 python3-pip python3-tk \
       mysql-client libmysqlclient-dev python-mysqldb \
    && python3 --version \
    && pip3 --version

COPY ./server /app

WORKDIR /app/website

RUN pip3 install -r requirements.txt 

COPY ./docker/credentials.py ./website/settings
COPY ./docker/start.sh .
COPY ./docker/createadmin.py .
COPY ./docker/wait-for-it.sh .

RUN chmod +x /app/website/*.sh

RUN sed s/'@localhost'/'@rabbitmq'/g ./website/settings/common.py > tmp \
    && mv tmp ./website/settings/common.py

ENTRYPOINT ["./start.sh"]

