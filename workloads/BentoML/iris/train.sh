#!/bin/bash

# source: https://knative.dev/community/samples/serving/machinelearning-python-bentoml/
# source2: https://www.kubeflow.org/docs/external-add-ons/serving/bentoml/

CLASSIFIER_NAME=IrisClassifier
DOCKER_TAG=$(date +%Y%m%d%H%M%S)
DOCKER_IMAGE=ghcr.io/nimamahmoudi/bentoml-iris-classifier:$DOCKER_TAG

# train
python train.py
# package in a docker container
bentoml containerize $CLASSIFIER_NAME:latest -t $DOCKER_IMAGE
docker push $DOCKER_IMAGE

# save for future usage
echo -n $DOCKER_IMAGE > ./.dockerimage

# print some info
echo "Docker Image: $DOCKER_IMAGE"
echo "command to test it out:"
echo "docker run -p 5000:5000 $DOCKER_IMAGE --workers=2"
