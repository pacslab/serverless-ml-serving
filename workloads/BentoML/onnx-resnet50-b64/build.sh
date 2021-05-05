#! /bin/bash

onnx_model_url=https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet50v2/resnet50v2.tar.gz
imagenet_labels_url=https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json

# download files and create folders
TMP_FOLDER=./tmp
mkdir -p $TMP_FOLDER

curl $onnx_model_url -o $TMP_FOLDER/resnet50v2.tar.gz
curl $imagenet_labels_url -o $TMP_FOLDER/imagenet-simple-labels.json
curl https://raw.githubusercontent.com/onnx/onnx-docker/master/onnx-ecosystem/inference_demos/images/dog.jpg \
  -o $TMP_FOLDER/dog.jpg

tar xvzf $TMP_FOLDER/resnet50v2.tar.gz --directory $TMP_FOLDER


# load the model into bentoml
pip install -q --requirement requirements.in

# create docker container
CLASSIFIER_NAME=OnnxResnet50B64
DOCKER_TAG=$(date +%Y%m%d%H%M%S)
DOCKER_IMAGE=ghcr.io/nimamahmoudi/bentoml-onnx-resnet50-b64:$DOCKER_TAG

# package in a docker container
bentoml containerize $CLASSIFIER_NAME:latest -t $DOCKER_IMAGE
docker push $DOCKER_IMAGE

# save for future usage
echo -n $DOCKER_IMAGE > ./.dockerimage

# print some info
echo "Docker Image: $DOCKER_IMAGE"
echo "command to test it out:"
echo "docker run --rm -p 5000:5000 $DOCKER_IMAGE --workers=2"

