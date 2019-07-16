FROM ubuntu:18.04
  
ENV DEBIAN_FRONTEND=noninteractive
ENV DJANGO_SETTINGS_MODULE=website.settings
ENV C_FORCE_ROOT=true

EXPOSE 8000

WORKDIR /app

RUN apt-get update \
    && apt-get install -y git python3.6 python3-pip python3-tk \
       mysql-client libmysqlclient-dev python-mysqldb \
    && git clone https://github.com/cmu-db/ottertune.git

WORKDIR /app/ottertune/server/website

RUN pip3 install -r requirements.txt 

COPY ./credentials.txt ./website/settings/credentials.py
COPY ./start.sh /app/ottertune/server/website
COPY ./createadmin.txt /app/ottertune/server/website/createadmin.py
COPY ./wait-for-it.sh /app/ottertune/server/website

RUN chmod +x /app/ottertune/server/website/start.sh

RUN sed s/'@localhost'/'@rabbitmq'/g ./website/settings/common.py > tmp \
    && mv tmp ./website/settings/common.py

VOLUME ["/app/ottertune/server/website/website/migrations"]
ENTRYPOINT ["./start.sh"]

