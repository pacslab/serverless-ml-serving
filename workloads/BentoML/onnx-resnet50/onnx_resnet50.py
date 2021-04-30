from typing import List

import numpy as np

import bentoml
from bentoml.frameworks.onnx import OnnxModelArtifact
from bentoml.service.artifacts.common import PickleArtifact
from bentoml.adapters import ImageInput


@bentoml.env(infer_pip_packages=True)
@bentoml.artifacts([OnnxModelArtifact('model'), PickleArtifact('labels')])
class OnnxResnet50(bentoml.BentoService):
    def preprocess(self, input_data):
        # convert the input data into the float32 input
        img_data = np.stack(input_data).transpose(0, 3, 1, 2)

        # normalize
        mean_vec = np.array([0.485, 0.456, 0.406])
        stddev_vec = np.array([0.229, 0.224, 0.225])

        norm_img_data = np.zeros(img_data.shape).astype('float32')

        for i in range(img_data.shape[0]):
            for j in range(img_data.shape[1]):
                norm_img_data[i, j, :, :] = (
                    img_data[i, j, :, :]/255 - mean_vec[j]) / stddev_vec[j]

        # add batch channel
        norm_img_data = norm_img_data.reshape(-1,
                                              3, 224, 224).astype('float32')
        return norm_img_data

    def softmax(self, x):
        x = x.reshape(-1)
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def post_process(self, raw_result):
        return self.softmax(np.array(raw_result)).tolist()

    @bentoml.api(input=ImageInput(), batch=True)
    def predict(self, image_ndarrays: List[np.ndarray]) -> List[str]:
        input_datas = self.preprocess(image_ndarrays)
        input_name = self.artifacts.model.get_inputs()[0].name

        outputs = []
        for i in range(input_datas.shape[0]):
            raw_result = self.artifacts.model.run(
                [], {input_name: input_datas[i:i+1]})
            result = self.post_process(raw_result)
            idx = np.argmax(result)
            sort_idx = np.flip(np.squeeze(np.argsort(result)))

            # return top 5 labels
            outputs.append(self.artifacts.labels[sort_idx[:5]])
        return outputs
