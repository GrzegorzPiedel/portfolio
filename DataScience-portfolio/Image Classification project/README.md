# Fashion MNIST Image Classification with CNN and Hyperparameter Tuning

This project demonstrates building and tuning Convolutional Neural Networks (CNNs) for multi-class image classification using the Fashion MNIST dataset

## Project Overview

The workflow includes:

1. **Data Preprocessing**
   - Loading Fashion MNIST dataset (70,000 grayscale images, 28x28 pixels)
   - Normalizing pixel values to the range [0, 1]
   - Splitting data into training, validation, and test sets
   - Reshaping images to include a channel dimension (required for CNNs)

2. **Model Building**
   - Designing a CNN with:
     - `Conv2D` layers for feature extraction
     - `MaxPooling2D` layers to reduce spatial dimensions
     - `Flatten` layers to prepare for fully connected layers
     - `Dense` layers for classification
     - `Dropout` layers to prevent overfitting
   - Using Leaky ReLU activation in hidden layers
   - Output layer uses Softmax for multi-class probabilities

3. **Hyperparameter Tuning**
   - Using `keras_tuner` to search for optimal:
     - Number of filters in convolutional layers
     - Kernel sizes
     - Number of units in dense layer
     - Dropout rate
     - Learning rate

4. **Evaluation**
   - Calculating test accuracy
   - Generating predictions on test data
   - Visualizing performance with a confusion matrix

## Dependencies

- Python 3.10+
- TensorFlow / Keras
- scikit-learn
- Matplotlib
- Seaborn
- KerasTuner
