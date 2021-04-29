#! /bin/bash

rm -rf ./tmp/resnet

mkdir -p ./tmp/resnet
curl -s http://download.tensorflow.org/models/official/20181001_resnet/savedmodels/resnet_v2_fp32_savedmodel_NHWC_jpg.tar.gz | \
tar --strip-components=2 -C ./tmp/resnet -xvz

echo "listing files"
ls ./tmp/resnet/*

docker build . -t tfserving-resnet

DOCKER_TAG=$(date +%Y%m%d%H%M%S)
DOCKER_IMAGE=ghcr.io/nimamahmoudi/tfserving-resnet:$DOCKER_TAG

docker tag tfserving-resnet $DOCKER_IMAGE
docker push $DOCKER_IMAGE

# save for future usage
echo -n $DOCKER_IMAGE > ./.dockerimage

# print some info
echo "Docker Image: $DOCKER_IMAGE"
echo "command to test it out:"
echo "docker run --rm -p 5000:5000 $DOCKER_IMAGE"
