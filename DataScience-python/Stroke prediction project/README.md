# Stroke Prediction Project

This project focuses on predicting the risk of stroke based on patient health data.  
It was developed as part of my postgraduate studies in Data Science.  

## Dataset
The dataset contains medical and demographic information such as:
- Gender
- Age
- Hypertension
- Heart disease
- Glucose level
- BMI
- Smoking status  
and others.  
The target variable is `stroke` (1 = stroke, 0 = no stroke).

## Methodology
1. **Data preprocessing**  
   - Handling missing values  
   - Encoding categorical variables  
   - Feature scaling  

2. **Modeling**  
   - Building and training neural network models using TensorFlow/Keras  
   - Experimenting with different architectures (Dense layers, Dropout, etc.)  
   - Hyperparameter tuning  

3. **Evaluation**  
   - Accuracy, precision, recall, F1-score  
   - Confusion matrix  
   - Visualization of training history  

## Technologies used
- Python
- Pandas, NumPy
- Scikit-learn
- TensorFlow / Keras
- Matplotlib, Seaborn

## Results
The trained model achieves good performance in predicting stroke risk, showing that deep learning models can be effectively applied to medical data classification tasks.

## File
- `Stroke_prediction.ipynb` – Jupyter Notebook containing the full code, explanations, and results.
