#! /bin/bash

rm -rf ./tmp/mobilenet

# using version as the subfolder (version 5 here)
mkdir -p ./tmp/mobilenet/5/
curl -sSL 'https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/classification/5?tf-hub-format=compressed' -o ./tmp/mobilenet.tar.gz
tar zxvf ./tmp/mobilenet.tar.gz -C ./tmp/mobilenet/5/

# echo "listing files"
ls ./tmp/mobilenet/*

docker build . -t tfserving-mobilenet

DOCKER_TAG=$(date +%Y%m%d%H%M%S)
DOCKER_IMAGE=ghcr.io/nimamahmoudi/tfserving-mobilenet:$DOCKER_TAG

docker tag tfserving-mobilenet $DOCKER_IMAGE
docker push $DOCKER_IMAGE

# save for future usage
echo -n $DOCKER_IMAGE > ./.dockerimage

# print some info
echo "Docker Image: $DOCKER_IMAGE"
echo "command to test it out:"
echo "docker run --rm -p 5000:5000 $DOCKER_IMAGE"
