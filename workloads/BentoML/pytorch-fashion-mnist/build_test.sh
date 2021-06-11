#! /bin/bash

# create docker container
CLASSIFIER_NAME=PyTorchFashionClassifier
DOCKER_TAG=$(date +%Y%m%d%H%M%S)
DOCKER_IMAGE=ghcr.io/nimamahmoudi/bentoml-pytorch-fashion-mnist:$DOCKER_TAG

python build.py

curl https://raw.githubusercontent.com/bentoml/gallery/master/pytorch/fashion-mnist/sample_image.png -o ./tmp/sample-image.png


# bentoml serve $CLASSIFIER_NAME:latest

# package in a docker container
bentoml containerize $CLASSIFIER_NAME:latest -t $DOCKER_IMAGE

# print some info
echo "Docker Image: $DOCKER_IMAGE"
echo "command to test it out:"
echo "docker run --rm -p 5000:5000 $DOCKER_IMAGE --workers=2"

# test it out:
TEST_CONTAINER_NAME=bentoml-pytorch-fashion-mnist-test
docker run --rm -d -p 5000:5000 --name $TEST_CONTAINER_NAME $DOCKER_IMAGE --workers=2

python test.py

docker stop $TEST_CONTAINER_NAME
