# %%
# source: https://colab.research.google.com/github/bentoml/gallery/blob/master/legacy-keras/toxic-comment-classification/keras-toxic-comment-classification.ipynb

import bentoml
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import text, sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D
from sklearn.model_selection import train_test_split


list_of_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
max_features = 20000
max_text_length = 400
embedding_dims = 50
filters = 250
kernel_size = 3
hidden_dims = 250
batch_size = 32
epochs = 2


# prepare dataset
train_df = pd.read_csv('./tmp/data/jigsaw-toxic-comment-train.csv')

print(train_df.head())

# %%
x = train_df['comment_text'].values
print(x)

# %%
y = train_df[list_of_classes].values
print(y)

# %%
x_tokenizer = text.Tokenizer(num_words=max_features)
print(x_tokenizer)
x_tokenizer.fit_on_texts(list(x))
print(x_tokenizer)
x_tokenized = x_tokenizer.texts_to_sequences(x) #list of lists(containing numbers), so basically a list of sequences, not a numpy array
#pad_sequences:transform a list of num_samples sequences (lists of scalars) into a 2D Numpy array of shape 
x_train_val = sequence.pad_sequences(x_tokenized, maxlen=max_text_length)


# %%
x_train, x_val, y_train, y_val = train_test_split(x_train_val, y, test_size=0.1, random_state=1)

# %%
print('Build model...')
model = Sequential()

# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
model.add(Embedding(max_features,
                    embedding_dims,
                    input_length=max_text_length))
model.add(Dropout(0.2))

# we add a Convolution1D, which will learn filters
# word group filters of size filter_length:
model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))
# we use max pooling:
model.add(GlobalMaxPooling1D())

# We add a vanilla hidden layer:
model.add(Dense(hidden_dims))
model.add(Dropout(0.2))
model.add(Activation('relu'))

# We project onto 6 output layers, and squash it with a sigmoid:
model.add(Dense(6))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

# %%
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
validation_data=(x_val, y_val))

# %%
test_df = pd.read_csv('./tmp/data/test.csv')
x_test = test_df['content'].values
x_test_tokenized = x_tokenizer.texts_to_sequences(x_test)
x_testing = sequence.pad_sequences(x_test_tokenized, maxlen=max_text_length)

# %%
y_testing = model.predict(x_testing, verbose=1)

# %%
sample_submission = pd.read_csv("./tmp/data/sample_submission.csv")
sample_submission[list_of_classes] = y_testing
sample_submission.to_csv("./tmp/data/toxic_comment_classification.csv", index=False)

# %%
# 1) import the custom BentoService defined above
from toxic_comment_classifier import ToxicCommentClassification

# 2) `pack` it with required artifacts
svc = ToxicCommentClassification()
svc.pack('x_tokenizer', x_tokenizer)
svc.pack('model', model)

# 3) save your BentoSerivce
saved_path = svc.save()