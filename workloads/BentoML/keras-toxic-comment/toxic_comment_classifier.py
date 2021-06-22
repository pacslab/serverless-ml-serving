from typing import List

from bentoml import api, artifacts, env, BentoService
from bentoml.frameworks.keras import KerasModelArtifact
from bentoml.service.artifacts.common import PickleArtifact
from bentoml.adapters import DataframeInput, JsonOutput

from tensorflow.keras.preprocessing import text, sequence
import numpy as np
import pandas as pd

list_of_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
max_text_length = 400

@env(pip_packages=['tensorflow==2.4.1', 'scikit-learn==0.24.2', 'pandas', 'numpy'])
@artifacts([PickleArtifact('x_tokenizer'), KerasModelArtifact('model')])
class ToxicCommentClassification(BentoService):
    
    def tokenize_df(self, df):
        comments = df['comment_text'].values
        tokenized = self.artifacts.x_tokenizer.texts_to_sequences(comments)        
        input_data = sequence.pad_sequences(tokenized, maxlen=max_text_length)
        return input_data
    
    @api(input=DataframeInput(), output=JsonOutput(), batch=True)
    def predict(self, df: pd.DataFrame) -> List[str]:
        input_data = self.tokenize_df(df)
        prediction = self.artifacts.model.predict(input_data)
        result = []
        for i in prediction:
            result.append({ list_of_classes[n]: i[n] for n in range(len(list_of_classes)) })
        return result
