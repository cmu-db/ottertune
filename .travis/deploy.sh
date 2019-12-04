#!/bin/bash

images="base web driver driver-internal"

for img in $images
do
    docker push "${DOCKER_REPO}:${img}"
done

