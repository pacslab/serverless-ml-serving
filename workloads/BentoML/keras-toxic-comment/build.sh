#! /bin/bash

# you need to do this only once when you download the dataset:
# (cd tmp/data/ && unzip jigsaw-multilingual-toxic-comment-classification.zip)

# you need to have activated conda env before running train script:
# conda activate keras-toxic-comment
python --version
which python

CLASSIFIER_NAME=ToxicCommentClassification
DOCKER_TAG=$(date +%Y%m%d%H%M%S)
DOCKER_IMAGE=ghcr.io/nimamahmoudi/bentoml-keras-toxic-comment-classification:$DOCKER_TAG

python train.py

# to serve:
# bentoml serve ToxicCommentClassification:latest

# package in a docker container
bentoml containerize $CLASSIFIER_NAME:latest -t $DOCKER_IMAGE
docker push $DOCKER_IMAGE

# save for future usage
echo -n $DOCKER_IMAGE > ./.dockerimage

# print some info
echo "Docker Image: $DOCKER_IMAGE"
echo "command to test it out:"
echo "docker run --rm -p 5000:5000 $DOCKER_IMAGE --workers=2"
