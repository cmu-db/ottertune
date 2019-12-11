FROM ottertune-base

COPY ./server /app

WORKDIR /app/website

COPY ./docker/credentials.py ./website/settings
COPY ./docker/start.sh ./docker/start-test.sh ./docker/wait-for-it.sh ./

RUN /install.sh web \
    && chmod +x ./*.sh

ENV DJANGO_SETTINGS_MODULE=website.settings
ENV C_FORCE_ROOT=true

ENTRYPOINT ["./start.sh"]

