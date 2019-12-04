FROM ottertune-base

ARG GRADLE_VERSION=gradle-5.5.1

ENV GRADLE_HOME=/opt/${GRADLE_VERSION}
ENV PATH=${GRADLE_HOME}/bin:${PATH}

RUN /install.sh driver

COPY ./client /app

WORKDIR /app/driver

