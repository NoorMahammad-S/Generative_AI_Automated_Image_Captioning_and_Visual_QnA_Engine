import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, add
from tensorflow.keras.preprocessing.text import Tokenizer

# Function to load and preprocess an image
def load_and_preprocess_image(photo_path):
    photo = image.load_img(photo_path, target_size=(299, 299))
    photo = image.img_to_array(photo)
    photo = np.expand_dims(photo, axis=0)
    return preprocess_input(photo)

# Function to generate captions for a new image
def generate_caption(model, tokenizer, photo):
    in_text = 'startseq'
    max_sequence_length = model.input[1].shape[1]  # Get the max sequence length from the model architecture
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

# Directory paths
data_dir = "path/to/your/data/"
model_dir = "path/to/your/models/"

# Load InceptionV3 pre-trained on ImageNet data
base_model = InceptionV3(weights='imagenet')
image_model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)

# Load the tokenized captions and image features
captions = [['startseq', 'cat', 'on', 'the', 'mat', 'endseq'], ['startseq', 'dog', 'playing', 'in', 'the', 'yard', 'endseq'], ...]
image_features = np.array([...])  # Replace [...] with your actual image features

# Preprocess captions
tokenizer = Tokenizer()
tokenizer.fit_on_texts(captions)
vocab_size = len(tokenizer.word_index) + 1  # Add 1 for the padding token

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

# Save the trained model
model.save(os.path.join(model_dir, "image_captioning_model.h5"))

# Example of generating a caption for a new image
new_photo_path = os.path.join(data_dir, "path/to/your/new/image.jpg")
new_photo = load_and_preprocess_image(new_photo_path)
generated_caption = generate_caption(model, tokenizer, new_photo)
print("Generated Caption:", generated_caption)
