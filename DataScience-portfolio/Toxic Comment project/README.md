# Toxic Comment Classification with BERT

This project demonstrates building and training a multi-label text classification model using BERT to detect toxic comments

## Project Overview

The workflow includes:

1. **Data Preprocessing**
   - Loading `toxic_subset.csv` containing comments and six binary labels
   - Splitting data into training, validation, and test sets
   - Tokenizing text with a pre-trained BERT tokenizer
   - Converting text into input IDs and attention masks for BERT

2. **Model Building**
   - Using a pre-trained `TFBertModel` as the feature extractor
   - Adding custom classification head with:
     - Dense layers for learning task-specific features
     - Dropout layers for regularization
   - Using Leaky ReLU activation in hidden layers
   - Output layer uses Sigmoid activation for multi-label probabilities

3. **Training**
   - Early stopping based on validation accuracy to prevent overfitting
   - Batch training with defined epochs
   - Reproducible results using fixed seeds for all random operations

4. **Threshold Tuning**
   - Optimizing thresholds for each class to maximize F1 score
   - Converting predicted probabilities into binary predictions using best thresholds

5. **Evaluation**
   - Calculating accuracy on the test set
   - Generating multi-label confusion matrices for each class
   - Visualizing training and validation accuracy per epoch

## Dependencies

- Python 3.10+
- TensorFlow / Keras
- Transformers (HuggingFace)
- scikit-learn
- Pandas
- Matplotlib
- NumPy
