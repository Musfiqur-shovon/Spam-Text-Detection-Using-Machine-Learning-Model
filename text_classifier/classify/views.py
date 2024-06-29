# classify/views.py
# classify/views.py

import numpy as np
import pickle
from django.shortcuts import render
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import os

# Print the directory to verify the path
print("Current directory:", os.path.dirname(__file__))

# Load the tokenizer and label encoder
tokenizer_path = os.path.join(os.path.dirname(__file__), 'tokenizer.pkl')
label_encoder_path = os.path.join(os.path.dirname(__file__), 'label_encoder.pkl')

print("Tokenizer path:", tokenizer_path)
print("Label encoder path:", label_encoder_path)

with open(tokenizer_path, 'rb') as handle:
    tokenizer = pickle.load(handle)

with open(label_encoder_path, 'rb') as handle:
    label_encoder = pickle.load(handle)

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), 'spam_classifier_model.h5')
model = load_model(model_path)

max_len = 100

def classify_text(request):
    prediction = None
    if request.method == 'POST':
        text = request.POST['text']
        sequences = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequences, maxlen=max_len)
        pred = model.predict(padded)
        prediction = label_encoder.inverse_transform([int(pred[0] > 0.5)])[0]
        
    return render(request, 'form.html', {'prediction': prediction})


