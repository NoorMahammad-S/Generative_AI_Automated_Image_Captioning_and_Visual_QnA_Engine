# Generative AI Automated Image Captioning and Visual QnA Engine

![Generated Image Caption](generated_image.jpg)

## Overview

This repository contains the implementation of a Generative AI system for Automated Image Captioning and Visual Question-Answering (Visual QnA) using TensorFlow and Keras libraries. The model is built upon the InceptionV3 architecture for image feature extraction and employs an LSTM-based language model for generating captions.

## Features

- **Automated Image Captioning:** The model generates descriptive captions for input images.
- **Visual Question-Answering (Visual QnA):** It answers questions related to input images.

## Requirements

Ensure you have the required Python packages installed. You can install them using the following command:

```bash
pip install -r requirements.txt
```

Refer to `requirements.txt` for the list of required packages.

## Usage

1. **Data Preparation:**
   - Replace `[...]` in the script with your actual data, including tokenized captions and image features.

2. **Model Training:**
   - Train the model using the provided script. Adjust hyperparameters as needed.

3. **Save Trained Model:**
   - The trained model is saved to the `models/` directory.

4. **Generate Captions:**
   - Use the `generate_caption` function to generate captions for new images.

5. **Adjust and Customize:**
   - Fine-tune the model architecture, hyperparameters, and data preprocessing based on your specific requirements.

## Example

```python
# Example of generating a caption for a new image
new_photo_path = "path/to/your/new/image.jpg"
new_photo = load_and_preprocess_image(new_photo_path)
generated_caption = generate_caption(model, tokenizer, new_photo)
print("Generated Caption:", generated_caption)
```

