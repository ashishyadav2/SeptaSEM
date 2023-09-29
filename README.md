# Image Caption Generator using VGG16 CNN and LSTM

![GitHub](https://img.shields.io/github/license/ashishyadav2/SeptaSEM)
![GitHub stars](https://img.shields.io/github/stars/ashishyadav2/SeptaSEM?style=social)
![GitHub forks](https://img.shields.io/github/forks/yashishyadav2/SeptaSEM?style=social)
![GitHub issues](https://img.shields.io/github/issues/ashishyadav2/SeptaSEM)

This repository contains the code and resources for building an Image Caption Generator using the VGG16 Convolutional Neural Network (CNN) and Long Short-Term Memory (LSTM) architecture. It uses the Flickr8k dataset to train the model.

## Overview

The Image Caption Generator is a deep learning model that generates textual descriptions for images. It combines the power of a pre-trained VGG16 CNN to extract image features and an LSTM network to generate captions. This project showcases how to preprocess images, build and train the model, and generate captions for new images.

## Features

- Utilizes the VGG16 model pre-trained on ImageNet for image feature extraction.
- Uses LSTM (Long Short-Term Memory) for sequence generation.
- Trains on the Flickr8k dataset, a widely used dataset for image captioning tasks.
- Provides a user-friendly interface for generating captions for custom images.
- Easily customizable for different datasets and model architectures.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.x
- TensorFlow 2.x
- Keras
- NumPy
- Matplotlib
- [Flickr8k dataset](https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip)
- [Flickr8k text descriptions](https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip)

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/ashishyadav2/SeptaSEM.git
   ```

2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Download and unzip the Flickr8k dataset and place it in the `data` directory.

## Usage

1. Train the model using the following command:

   ```bash
   python train.py
   ```

2. Once the model is trained, you can generate captions for your own images using:

   ```bash
   python generate_caption.py --image <path_to_image>
   ```

## Model Architecture

![Model Architecture](images/model_architecture.png)

## Results

Here are some sample results from the model:

![Sample Result 1](images/sample_result_1.png)
![Sample Result 2](images/sample_result_2.png)

## Contributing

Contributions are welcome! If you have any ideas, enhancements, or bug fixes, please open an issue or create a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This project is inspired by the [Show and Tell: A Neural Image Caption Generator](https://arxiv.org/abs/1411.4555) paper by Oriol Vinyals, et al.
- The Flickr8k dataset was originally compiled by Samy Bengio, Pierre-Emmanuel Leni, and others.

---

