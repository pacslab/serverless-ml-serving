import requests
import base64
import json
import io


import tensorflow as tf
import numpy as np
import pandas as pd

from PIL import Image

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



image = Image.open('./tmp/dog.jpg')
image_np = np.array(image)
image_np = resize_image(image_np)
image_b64 = convert_image_to_b64(image_np)


predict_request = [{'b64': image_b64}] * 5
response = requests.post('http://localhost:5000/predict', json=predict_request)
response.raise_for_status()

print(response.json())

print('='*50)
print((' '*5) + 'test was successful!')
print('='*50)
