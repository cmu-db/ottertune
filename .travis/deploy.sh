#!/bin/bash

set -ex

for tag in base web driver; do
    docker push "${DOCKER_REPO}:${tag}"
done
