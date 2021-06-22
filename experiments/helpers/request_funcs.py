import requests
import base64
import json
import os
import random
import traceback
import io
import time

from tqdm.auto import tqdm
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from tensorflow.keras.applications import imagenet_utils

IMAGE_SAMPLE_SIZE = 100

WORKLOAD_BENTOML_IRIS_NAME = 'bentoml-iris'
WORKLOAD_BENTOML_ONNX_RESNET50 = 'bentoml-onnx-resnet50'
WORKLOAD_TFSERVING_RESNETV2 = 'tfserving-resnetv2'
WORKLOAD_TFSERVING_MOBILENETV1 = 'tfserving-mobilenetv1'
WORKLOAD_BENTOML_PYTORCH_FASHION_MNIST = 'bentoml-pytorch-fashion-mnist'
WORKLOAD_BENTOML_KERAS_TOXIC_COMMENTS = 'bentoml-keras-toxic-comments'

default_server_urls = {
    WORKLOAD_BENTOML_IRIS_NAME: 'http://bentoml-iris.default.kn.nima-dev.com/predict',
    WORKLOAD_BENTOML_ONNX_RESNET50: 'http://bentoml-onnx-resnet50.default.kn.nima-dev.com/predict',
    WORKLOAD_TFSERVING_RESNETV2: 'http://tfserving-resnetv2.default.kn.nima-dev.com/v1/models/resnet:predict',
    WORKLOAD_TFSERVING_MOBILENETV1: 'http://tfserving-mobilenetv1.default.kn.nima-dev.com/v1/models/mobilenet:predict',
    WORKLOAD_BENTOML_PYTORCH_FASHION_MNIST: 'http://bentoml-pytorch-fashion-mnist.default.kn.nima-dev.com/predict',
    WORKLOAD_BENTOML_KERAS_TOXIC_COMMENTS: 'http://bentoml-keras-toxic-comments.default.kn.nima-dev.com/predict',
}

ds_iris = tfds.load('iris', split='train', shuffle_files=False)
iris_featurs = [list(d['features'].numpy().tolist()) for d in ds_iris]
ds_toxic_comments_file = os.path.join(os.path.dirname(__file__), 'jigsaw_comments_test.csv')
ds_toxic_comments = pd.read_csv(ds_toxic_comments_file)['comment_text'].values


def keras_toxic_prediction_parsing(p):
    if p['toxic'] > 0.5:
        return 'toxic'
    else:
        return 'normal'

# Keras Toxic Comment Classification
def request_keras_toxic_comments(batch_size=1, url=None):
    if url is None:
        url = default_server_urls[WORKLOAD_BENTOML_KERAS_TOXIC_COMMENTS]

    predict_comments = random.choices(ds_toxic_comments, k=batch_size)
    predict_request = [
        {
            "comment_text": c,
        }
        for c in predict_comments 
    ]
    response = requests.post(url, json=predict_request)
    response.raise_for_status()
    return {
        'prediction': [keras_toxic_prediction_parsing(p) for p in response.json()],
        'response_time_ms': response.elapsed.total_seconds()*1000,
        'headers': {k: response.headers[k] for k in response.headers if k.startswith('X-')},
    }

# Iris on BentoML
def request_bentoml_iris(batch_size=1, url=None):
    if url is None:
        url = default_server_urls[WORKLOAD_BENTOML_IRIS_NAME]

    predict_request = random.choices(iris_featurs, k=batch_size)
    response = requests.post(url, json=predict_request)
    response.raise_for_status()
    return {
        'prediction': response.json(),
        'response_time_ms': response.elapsed.total_seconds()*1000,
        'headers': {k: response.headers[k] for k in response.headers if k.startswith('X-')},
    }


# parse image resizes the image to the size we need for our workload
def resize_image(image, target_size=(224, 224)):
    image = tf.cast(image, tf.float32)
    image = image / 255.0
    image = tf.image.resize(image, target_size)
    image = image * 255.0
    image = tf.cast(image, tf.uint8)
    return image


def convert_image_to_bytes(image_np):
    image = tf.keras.preprocessing.image.array_to_img(image_np)
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='jpeg')
    img_byte_arr = img_byte_arr.getvalue()
    return img_byte_arr


def convert_image_to_b64(image_np):
    img_byte_arr = convert_image_to_bytes(image_np)
    jpeg_bytes = base64.b64encode(img_byte_arr).decode('utf-8')
    return jpeg_bytes


def preprocess_imagenet_224(d):
    image_np = d['image']
    image_np = resize_image(image_np)
    return image_np


def preprocess_mobilenet(image_np):
    image_input = image_np.numpy()
    image_parsed = tf.keras.applications.mobilenet.preprocess_input(image_input)
    image_parsed = image_parsed.tolist()
    # instances will be added later on the request function
    # predict_request = {"instances" : [{"inputs": image_parsed}]}
    predict_request = {"inputs": image_parsed}
    return predict_request


print('fetching imagenet v2')
ds_imagenet = tfds.load('imagenet_v2', split='test', shuffle_files=False)
print('resizing images')
ds_imagenet_images = [preprocess_imagenet_224(d) for d in tqdm(ds_imagenet.take(IMAGE_SAMPLE_SIZE))]
print('converting to bentoml files')

# ds_imagenet_images_bentoml_files = [{
#     'image': ('test.jpg', convert_image_to_bytes(image_np))
# } for image_np in tqdm(ds_imagenet_images)]

# base64 version of file submission
ds_imagenet_images_bentoml_files = [{
    'b64': convert_image_to_b64(image_np)
} for image_np in tqdm(ds_imagenet_images)]


# bentoml onnx resnet50 workload
# def request_bentoml_onnx_resnet50(batch_size=1):
#     # this currently has issues with batch!
#     if batch_size > 1:
#         raise NotImplementedError('cannot handle batch_size > 1 for image input')

#     files = random.choices(ds_imagenet_images_bentoml_files, k=batch_size)
#     prep_files = {}
#     for i, file in enumerate(files):
#         prep_files[f"image{i}"] = file['image']

#     response = requests.post(default_server_urls[WORKLOAD_BENTOML_ONNX_RESNET50], files=prep_files)
#     response.raise_for_status()
#     return {
#         'prediction': response.json(),
#         'response_time_ms': response.elapsed.total_seconds()*1000,
#     }

# bentoml onnx resnet50 workload with base64 decoding
def request_bentoml_onnx_resnet50(batch_size=1, url=None):
    if url is None:
        url = default_server_urls[WORKLOAD_BENTOML_ONNX_RESNET50]

    predict_request = random.choices(ds_imagenet_images_bentoml_files, k=batch_size)
    response = requests.post(url, json=predict_request)
    response.raise_for_status()
    return {
        'prediction': response.json(),
        'response_time_ms': response.elapsed.total_seconds()*1000,
        'headers': {k: response.headers[k] for k in response.headers if k.startswith('X-')},
    }


print('extracting base64 files')
# ds_imagenet_images_tfserving_b64 = [
#     '{"instances" : [{"b64": "%s"}]}' % convert_image_to_b64(image_np)
#     for image_np in tqdm(ds_imagenet_images)
# ]
ds_imagenet_images_tfserving_b64 = [
    { "b64" : convert_image_to_b64(image_np) }
    for image_np in tqdm(ds_imagenet_images)
]


# tfserving resnet v2 workload
def request_tfserving_resnetv2(batch_size=1, url=None):
    if url is None:
        url = default_server_urls[WORKLOAD_TFSERVING_RESNETV2]

    files = random.choices(ds_imagenet_images_tfserving_b64, k=batch_size)
    predict_request = {
        "instances": files,
    }

    response = requests.post(url, json=predict_request)
    response.raise_for_status()

    resp_predictions = []
    for prediction in response.json()['predictions']:
        result = prediction['probabilities']
        p = np.array(result[1:])
        p = p.reshape((1,1000))
        p = imagenet_utils.decode_predictions(p)
        resp_predictions.append([c[1] for c in p[0]])

    return {
        'prediction': resp_predictions,
        'response_time_ms': response.elapsed.total_seconds()*1000,
        'headers': {k: response.headers[k] for k in response.headers if k.startswith('X-')},
    }


print('preprocessing for mobilenet')
ds_imagenet_images_tfserving_mobilenet = [preprocess_mobilenet(image_np) for image_np in tqdm(ds_imagenet_images)]

def request_tfserving_mobilenetv1(batch_size=1, url=None):
    if url is None:
        url = default_server_urls[WORKLOAD_TFSERVING_MOBILENETV1]

    predict_files = random.choices(ds_imagenet_images_tfserving_mobilenet, k=batch_size)
    predict_request = {"instances" : predict_files}
    response = requests.post(url, json=predict_request)
    response.raise_for_status()

    resp_predictions = []
    for prediction in response.json()['predictions']:
        p = np.array(prediction[1:])
        # apply softmax: https://github.com/bentoml/gallery/blob/master/onnx/resnet50/resnet50.ipynb
        p = np.exp(p)/sum(np.exp(p))
        p = p.reshape((1,1000))
        p = imagenet_utils.decode_predictions(p)
        resp_predictions.append([c[1] for c in p[0]])

    return {
        'prediction': resp_predictions,
        'response_time_ms': response.elapsed.total_seconds()*1000,
        'headers': {k: response.headers[k] for k in response.headers if k.startswith('X-')},
    }



def request_bentoml_pytorch_fashion_mnist(batch_size=1, url=None):
    if url is None:
        url = default_server_urls[WORKLOAD_BENTOML_PYTORCH_FASHION_MNIST]

    predict_request = random.choices(ds_imagenet_images_bentoml_files, k=batch_size)
    response = requests.post(url, json=predict_request)
    response.raise_for_status()
    return {
        'prediction': response.json(),
        'response_time_ms': response.elapsed.total_seconds()*1000,
        'headers': {k: response.headers[k] for k in response.headers if k.startswith('X-')},
    }


workload_funcs = {
    WORKLOAD_BENTOML_IRIS_NAME: request_bentoml_iris,
    WORKLOAD_BENTOML_ONNX_RESNET50: request_bentoml_onnx_resnet50,
    WORKLOAD_TFSERVING_RESNETV2: request_tfserving_resnetv2,
    WORKLOAD_TFSERVING_MOBILENETV1: request_tfserving_mobilenetv1,
    WORKLOAD_BENTOML_PYTORCH_FASHION_MNIST: request_bentoml_pytorch_fashion_mnist,
    WORKLOAD_BENTOML_KERAS_TOXIC_COMMENTS: request_keras_toxic_comments
}


if __name__ == "__main__":
    print('Starting testing...')

    for k in workload_funcs:
        for batch_size in [1,2,5]:
            print(f'running {k} workload function, batch_size: {batch_size}')
            try:
                for count in range(10):
                    result = workload_funcs[k](batch_size=batch_size)
                    print(f'[{count+1}]: result: {result}')
            except Exception:
                print('exception occured:')
                traceback.print_exc()

        print('waiting for resources to release')
        time.sleep(30)
