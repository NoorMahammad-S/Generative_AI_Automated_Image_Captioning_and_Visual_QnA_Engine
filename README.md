# ```             Generative AI Project             ```

#  Automated Image Captioning and Visual QnA Engine

![Generated Image Caption](generated_image.jpg)

## Overview

Welcome to the **Automated Image Captioning and Visual QnA Engine**! This repository implements a **Generative AI system** for creating **automated image captions** and answering **visual questions** using **TensorFlow** and **Keras**. The model leverages the **InceptionV3** architecture for image feature extraction and utilizes an **LSTM-based language model** to generate descriptive captions. It's a powerful tool for anyone working with computer vision and natural language processing.

---

### Key Features

* **Automated Image Captioning**: Automatically generates human-readable descriptions for images.
* **Visual Question-Answering (Visual QnA)**: Provides accurate answers to questions related to the contents of an image.

---

### Requirements

To get started, you'll need the following Python packages:

```bash
pip install -r requirements.txt
```

Check the `requirements.txt` file for the full list of dependencies.

---

### Installation & Setup

1. **Clone the Repository**:
   First, clone the repository to your local machine:

   ```bash
   git clone https://github.com/NoorMahammad-S/Generative_AI_Automated_Image_Captioning_and_Visual_QnA_Engine.git
   cd Generative_AI_Automated_Image_Captioning_and_Visual_QnA_Engine
   ```

2. **Install Dependencies**:
   Install the necessary Python packages:

   ```bash
   pip install -r requirements.txt
   ```

3. **Download Pre-trained Models** *(Optional)*:
   If you don’t want to train the model from scratch, you can download pre-trained weights from [here](link-to-pretrained-model).

---

### Usage

#### 1. **Data Preparation**

Before training the model, ensure your dataset includes tokenized captions and image features. Replace `[...]` in the provided script with your actual data paths.

#### 2. **Model Training**

Use the following script to train the model. You can adjust hyperparameters like learning rate, batch size, etc., to suit your data:

```bash
python main.py
```

#### 3. **Saving the Trained Model**

Once the training is complete, the model is saved in the `models/` directory. You can use it for future predictions or further fine-tuning.

#### 4. **Generate Image Captions**

Generate captions for new images with the following code:

```python
# Example: Generate a caption for a new image
new_photo_path = "path/to/your/new/image.jpg"
new_photo = load_and_preprocess_image(new_photo_path)
generated_caption = generate_caption(model, tokenizer, new_photo)
print("Generated Caption:", generated_caption)
```

#### 5. **Visual Question-Answering**

You can also use the model for answering questions related to images. Here’s an example:

```python
# Example: Answer a question about an image
question = "What is the color of the car?"
answer = answer_visual_question(model, tokenizer, new_photo, question)
print("Answer:", answer)
```

#### 6. **Customization**

Feel free to fine-tune the model architecture, adjust the hyperparameters, or modify the data preprocessing pipeline to meet your specific needs.

---

### Example Outputs

* **Caption for an Image**: "A dog is playing with a ball in the park."
* **Visual QnA Answer**: "The dog is brown with white spots."

---

### Contributing

We welcome contributions! If you’d like to improve the project, please fork the repository, make changes, and submit a pull request. Feel free to check out our [Contributing Guide](CONTRIBUTING.md) for more information.

---

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

### Topics and Tags

* **Image Captioning**
* **Visual Question-Answering (Visual QnA)**
* **Deep Learning**
* **Computer Vision**
* **Natural Language Processing (NLP)**
* **TensorFlow**
* **Keras**
* **LSTM**
* **InceptionV3**

---

### Additional Resources

* [TensorFlow Documentation](https://www.tensorflow.org)
* [Keras Documentation](https://keras.io)
* [InceptionV3 Model Paper](https://arxiv.org/abs/1512.00567)

---
