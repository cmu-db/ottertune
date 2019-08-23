FROM ottertune-base
  
ENV DJANGO_SETTINGS_MODULE=website.settings
ENV C_FORCE_ROOT=true

RUN mkdir -p /app

COPY . /app

WORKDIR /app/server/website

COPY ./docker/credentials.py ./website/settings
COPY ./docker/start.sh .
COPY ./docker/createadmin.py .
COPY ./docker/wait-for-it.sh .

RUN chmod +x ./*.sh

RUN sed s/'@localhost'/'@rabbitmq'/g ./website/settings/common.py > tmp \
    && mv tmp ./website/settings/common.py

ENTRYPOINT ["./start.sh"]

