#!/bin/bash

cd $ROOT/docker
mkdir tmp
cd tmp
git clone "https://${GIT_TOKEN}@github.com/${GIT_ORG}/${GIT_REPO}"
cd $GIT_REPO
cp $ROOT/docker/install.sh $WEB/requirements.txt .
git_commit=`git log --format="%H" -n 1`
docker-compose -f docker-compose.build.yml build --build-arg GIT_COMMIT="$git_commit"

docker tag ottertune-base "${DOCKER_REPO}:base"
docker tag ottertune-web "${DOCKER_REPO}:web"
docker tag ottertune-driver "${DOCKER_REPO}:driver"
docker tag ottertune-driver-internal "${DOCKER_REPO}:driver-internal"
echo "$DOCKER_PASSWD" | docker login -u "$DOCKER_USER" --password-stdin

