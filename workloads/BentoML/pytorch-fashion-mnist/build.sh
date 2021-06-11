#! /bin/bash

# stop on error
set -e

# load the model into bentoml
pip install -q --requirement requirements.in

# create docker container
CLASSIFIER_NAME=PyTorchFashionClassifier
DOCKER_TAG=$(date +%Y%m%d%H%M%S)
DOCKER_IMAGE=ghcr.io/nimamahmoudi/bentoml-pytorch-fashion-mnist:$DOCKER_TAG

# package in a docker container
bentoml containerize $CLASSIFIER_NAME:latest -t $DOCKER_IMAGE
docker push $DOCKER_IMAGE

# save for future usage
echo -n $DOCKER_IMAGE > ./.dockerimage

# print some info
echo "Docker Image: $DOCKER_IMAGE"
echo "command to test it out:"
echo "docker run --rm -p 5000:5000 $DOCKER_IMAGE --workers=2"

