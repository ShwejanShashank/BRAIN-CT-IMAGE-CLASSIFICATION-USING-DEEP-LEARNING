# Brain CT Image Classification Using Deep Learning

## Project Overview

This project aims to develop a deep learning-based model to classify brain CT images, specifically Non-Contrast Head CT scans, as either normal or exhibiting hemorrhage. The model utilizes Convolutional Neural Networks (CNNs) to detect hemorrhage regions, providing an automated tool to assist radiologists and medical professionals in diagnosing critical neurological conditions.

## Key Features

- **Deep Learning Model**: Uses Convolutional Neural Networks (CNNs) for image classification.
- **Preprocessing**: Data augmentation and normalization for enhanced model generalization.
- **High Accuracy**: Achieves significant accuracy in distinguishing normal and hemorrhage images.

## Data Collection

The dataset for this project was sourced from the Kaggle community and consists of brain CT images, with labels indicating whether hemorrhage is present or not.

## Model Architecture

### Simple Convolutional Neural Network:
- **Convolutional Layers**: 32 filters, 3x3 kernel, ReLU activation.
- **Max-Pooling**: Applied after each convolutional layer.
- **Global Average Pooling**: Reduces spatial dimensions to 1x1.
- **Dropout Layers**: To reduce overfitting, with a dropout rate of 0.4.
- **Dense Layers**: One dense layer with 32 units (ReLU), and a final output layer with 1 unit (sigmoid).

### Larger Convolutional Neural Network:
- **Convolutional Layers**: 32 filters, 3x3 kernel, ReLU activation.
- **Max-Pooling**: Applied after each convolutional layer.
- **Global Average Pooling**: Reduces spatial dimensions to 1x1.
- **Dropout Layers**: With a dropout rate of 0.4.
- **Dense Layers**: First dense layer with 64 units (ReLU) for increased capacity to capture complex patterns.

## Preprocessing

- **ImageDataGenerator**: Applied for real-time data augmentation including transformations like rotation, flipping, and normalization.
- **Batching**: Efficient memory usage during training.

## Results

- **Simple CNN**: Achieved a training accuracy of 78.8% and validation accuracy of 76.8%.
- **Larger CNN**: Achieved a training accuracy of 89.8% and validation accuracy of 82.3%.

## Installation

To get started, clone the repository and install the required dependencies:

```bash
git clone https://github.com/your-repository/brain-ct-image-classification.git
cd brain-ct-image-classification
pip install -r requirements.txt
```

### Requirements

- Python 3.x
- TensorFlow / Keras
- Numpy
- Matplotlib
- Scikit-learn
- OpenCV
- PIL

## Usage

1. **Prepare Dataset**: Place your CT scan images in the `data/` directory.
2. **Training**: Run the following script to train the model:

```bash
python train.py
```

3. **Evaluation**: After training, evaluate the model's performance using the testing dataset:

```bash
python evaluate.py
```

4. **Prediction**: To predict on new CT images:

```bash
python predict.py --image_path path_to_image
```

## Conclusion

This project demonstrates the potential of deep learning to assist in medical diagnostics, specifically for detecting brain hemorrhages in CT scans. By leveraging deep learning techniques, this model can serve as a powerful tool for radiologists, improving diagnostic accuracy and speed.

