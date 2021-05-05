#! /bin/bash


# create docker container
CLASSIFIER_NAME=OnnxResnet50
DOCKER_TAG=$(date +%Y%m%d%H%M%S)
DOCKER_IMAGE=ghcr.io/nimamahmoudi/bentoml-onnx-resnet50:$DOCKER_TAG

# package in a docker container
bentoml containerize $CLASSIFIER_NAME:latest -t $DOCKER_IMAGE

# print some info
echo "Docker Image: $DOCKER_IMAGE"
echo "command to test it out:"
echo "docker run --rm -p 5000:5000 $DOCKER_IMAGE --workers=2"

# test it out:
docker run --rm -p 5000:5000 $DOCKER_IMAGE --workers=2
