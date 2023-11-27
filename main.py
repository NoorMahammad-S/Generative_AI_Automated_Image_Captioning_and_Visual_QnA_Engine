import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense

# Load InceptionV3 pre-trained on ImageNet data
base_model = InceptionV3(weights='imagenet')
image_model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)

# Load the tokenized captions and image features
# This part assumes you have a preprocessed dataset with captions and corresponding image features
captions = [...]  # List of tokenized captions
image_features = [...]  # Extracted image features using InceptionV3

# Preprocess captions
tokenizer = ...  # Use your favorite tokenizer (e.g., from keras.preprocessing.text)
vocab_size = ...  # Define your vocabulary size

# Pad sequences to a fixed length
max_sequence_length = max(len(seq) for seq in captions)
padded_captions = pad_sequences(captions, maxlen=max_sequence_length, padding='post')

# One-hot encode captions
one_hot_captions = to_categorical(padded_captions, num_classes=vocab_size)

# Define the image captioning model
embedding_size = 256
inputs1 = Input(shape=(2048,))
fe1 = Dense(embedding_size, activation='relu')(inputs1)
inputs2 = Input(shape=(max_sequence_length,))
se1 = Embedding(vocab_size, embedding_size, mask_zero=True)(inputs2)
se2 = LSTM(256)(se1)
decoder1 = add([fe1, se2])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)
model = Model(inputs=[inputs1, inputs2], outputs=outputs)

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Train the model
model.fit([image_features, padded_captions], one_hot_captions, epochs=10, verbose=2)

# For generating captions for a new image
def generate_caption(photo):
    # Convert image to features
    photo = image.load_img(photo, target_size=(299, 299))
    photo = image.img_to_array(photo)
    photo = np.expand_dims(photo, axis=0)
    photo = preprocess_input(photo)
    # Get the image features
    in_text = 'startseq'
    for i in range(max_sequence_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_sequence_length)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = word_for_id(yhat, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text

# Helper function to map an integer to a word
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None
