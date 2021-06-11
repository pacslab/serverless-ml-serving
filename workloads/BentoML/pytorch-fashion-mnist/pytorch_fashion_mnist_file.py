from typing import BinaryIO, List

import bentoml
from PIL import Image
import torch
from torchvision import transforms

from bentoml.frameworks.pytorch import PytorchModelArtifact
from bentoml.adapters import FileInput, JsonOutput


FASHION_MNIST_CLASSES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                         'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


@bentoml.env(pip_packages=['torch', 'numpy', 'torchvision', 'scikit-learn'])
@bentoml.artifacts([PytorchModelArtifact('classifier')])
class PyTorchFashionClassifier(bentoml.BentoService):
    
    @bentoml.utils.cached_property  # reuse transformer
    def transform(self):
        return transforms.Compose([transforms.CenterCrop((29, 29)), transforms.ToTensor()])

    @bentoml.api(input=FileInput(), output=JsonOutput(), batch=True)
    def predict(self, file_streams: List[BinaryIO]) -> List[str]:
        img_tensors = []
        for fs in file_streams:
            img = Image.open(fs).convert(mode="L").resize((28, 28))
            img_tensors.append(self.transform(img))
        outputs = self.artifacts.classifier(torch.stack(img_tensors))
        _, output_classes = outputs.max(dim=1)
        
        return [FASHION_MNIST_CLASSES[output_class] for output_class in output_classes]