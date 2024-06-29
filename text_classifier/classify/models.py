from django.db import models

# Create your models here.

# text_classification_model.py

import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Sample data
data = {
    'text': ["Free entry in 2 a weekly competition", "Hey, how are you?", "Win cash now!!!", "Hello, wanna grab lunch?"],
    'label': ["spam", "ham", "spam", "ham"]
}

df = pd.DataFrame(data)

# Prepare the data
max_words = 5000
max_len = 100
tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
tokenizer.fit_on_texts(df['text'].values)
word_index = tokenizer.word_index
vocab_size = len(word_index) + 1

X = tokenizer.texts_to_sequences(df['text'].values)
X = pad_sequences(X, maxlen=max_len)

# Encode labels
label_encoder = LabelEncoder()
Y = label_encoder.fit_transform(df['label'].values)
Y = np.reshape(Y, (-1, 1))

# Train test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Define the model
model = Sequential()
model.add(Embedding(vocab_size, 128, input_length=max_len))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, Y_train, epochs=5, batch_size=64, validation_data=(X_test, Y_test))

# Save the model
model.save("spam_classifier_model.h5")

# Save the tokenizer
with open('tokenizer.pkl', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Save the label encoder
with open('label_encoder.pkl', 'wb') as handle:
    pickle.dump(label_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)

