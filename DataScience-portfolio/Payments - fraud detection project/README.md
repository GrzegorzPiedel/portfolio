# Credit Card Fraud Detection with Autoencoders

This project demonstrates how autoencoders can be used for anomaly detection in financial data.  
We use the well-known Credit Card Fraud Detection dataset to train an autoencoder only on normal transactions (Class = 0).  
Fraudulent transactions are then identified based on their high reconstruction error.

## Steps:
1. Preprocessing and normalization of the data  
2. Building the autoencoder with Batch Normalization and Dropout  
3. Training on normal transactions only  
4. Evaluating reconstruction error and detecting anomalies  
5. Measuring performance using ROC AUC and classification metrics  

## Technologies used:
- Python, TensorFlow/Keras
- Scikit-learn
- Matplotlib, Seaborn

## Results:
The trained model is able to identify fraudulent transactions with a high ROC AUC score, demonstrating the effectiveness of autoencoders in unsupervised anomaly detection.
