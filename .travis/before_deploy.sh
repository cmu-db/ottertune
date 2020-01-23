#!/bin/bash

set -ex

for tag in base web driver; do
    docker tag "ottertune-${tag}" "${DOCKER_REPO}:${tag}"
done

echo "$DOCKER_PASSWD" | docker login -u "$DOCKER_USER" --password-stdin

