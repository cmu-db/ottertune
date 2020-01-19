FROM ubuntu:18.04

ARG DEBUG=true
ARG GIT_COMMIT=

ENV DEBIAN_FRONTEND=noninteractive

COPY ./docker/install.sh ./server/website/requirements.txt /
WORKDIR /

RUN mkdir -p /app \
    && ([ -n "$GIT_COMMIT" ] && echo "base=$GIT_COMMIT" > /app/.git_commit || :) \
    && chmod +x install.sh \
    && ./install.sh base

ENV DEBUG=$DEBUG
