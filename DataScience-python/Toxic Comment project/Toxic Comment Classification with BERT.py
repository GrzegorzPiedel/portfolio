import os
import random
import numpy as np
import tensorflow as tf
import keras

# 1. Set seed for reproducibility
seed = 42

# 2. Apply seed to all libraries
os.environ["PYTHONHASHseed"] = str(seed)
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
keras.utils.set_random_seed(seed)

# 3. Enforce deterministic operations in TensorFlow
os.environ["TF_DETERMINISTIC_OPS"] = "1"
tf.config.experimental.enable_op_determinism()

bert_mini_path = "./bert-tiny"
bert_mini_path2 = "prajjwal1/bert-tiny"

import pandas as pd
from transformers import BertTokenizer, TFBertModel
df = pd.read_csv('toxic_subset.csv')

texts = df['comment_text'].tolist()
labels = df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult',
       'identity_hate']].values

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained(bert_mini_path2)

from sklearn.model_selection import train_test_split

# Split data into train, validation, and test sets
texts_train_val, texts_test, y_train_val, y_test = train_test_split(texts, labels, test_size=0.1, random_state=seed)
texts_train, texts_val, y_train, y_val = train_test_split(texts_train_val, y_train_val, test_size=0.2, random_state=seed)

max_len = 128

# Tokenize input texts
X_train = tokenizer(
    texts_train,
    max_length = max_len,
    truncation = True,
    padding = "max_length",
    return_tensors = "tf"
)

X_val = tokenizer(
    texts_val,
    max_length = max_len,
    truncation = True,
    padding = "max_length",
    return_tensors = "tf"
)

X_test = tokenizer(
    texts_test,
    max_length = max_len,
    truncation = True,
    padding = "max_length",
    return_tensors = "tf"
)

def build_model():
    # Load pre-trained BERT model
    bert = TFBertModel.from_pretrained(bert_mini_path2, from_pt = True)

    # Define input layers for BERT
    input_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_ids")
    attention_mask = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="attention_mask")

    # Pass inputs through BERT and get pooled output
    bert_output = bert(input_ids, attention_mask=attention_mask)[1]

    # Dropout for regularization
    x = tf.keras.layers.Dropout(0.3, seed=seed)(bert_output)

    # Dense layer with leaky_relu activation
    x = tf.keras.layers.Dense(
        64,
        activation = "leaky_relu",
        kernel_initializer = tf.keras.initializers.GlorotUniform(seed = seed)
    )(x)

    # Output layer with sigmoid for multi-label classification
    x = tf.keras.layers.Dense(
        6,
        activation = "sigmoid",
        kernel_initializer = tf.keras.initializers.GlorotUniform(seed = seed)
    )(x)

    # Build Keras model
    model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=x)

    # Compile model with Adam optimizer and binary cross-entropy
    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate = 2e-5),
        loss = "binary_crossentropy",
        metrics = ["accuracy"]
    )

    return model

model = build_model()

# Early stopping callback
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor = "val_accuracy",
    patience = 2,
    restore_best_weights = True
)

# Train the model
history = model.fit(
    x = {
        "input_ids": X_train["input_ids"],
        "attention_mask": X_train["attention_mask"]
    },
    y = y_train,
    validation_data = (
        {"input_ids": X_val["input_ids"],
         "attention_mask": X_val["attention_mask"]},
        y_val),
    epochs = 5,
    batch_size = 32,
    callbacks = [early_stop],
    shuffle = False
)

model.save("toxic_model.h5", include_optimizer = False)

# Evaluate on test set
loss, acc = model.evaluate(
    {"input_ids": X_test["input_ids"],
     "attention_mask": X_test["attention_mask"]},
    y_test
)

print(f"Test accuracy: {acc:.4f}")

import matplotlib.pyplot as plt

# Plot training and validation accuracy
train_accuracy = history.history["accuracy"]
val_accuracy = history.history["val_accuracy"]
epochs = range(1, len(train_accuracy)+1)

plt.figure(figsize = (8, 6))
plt.plot(epochs, train_accuracy, label='Train accuracy', color='blue', marker="o")
plt.plot(epochs, val_accuracy, label='Val accuracy', color='red',  marker="s")
plt.title('Train and validation accuracy per epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.xticks(epochs)
plt.legend()
plt.show()

# Predict probabilities on test set
y_pred_probs = model.predict(
    {
        "input_ids": X_test["input_ids"],
        "attention_mask": X_test["attention_mask"]
    }
)

# Convert probabilities to binary predictions
y_pred = (y_pred_probs > 0.5).astype(int)

from sklearn.metrics import multilabel_confusion_matrix, f1_score

# Compute confusion matrices for all labels
cm = multilabel_confusion_matrix(y_test, y_pred)
for i, matrix in enumerate(cm):
    print(f"Class {i}:\n{matrix}\n")

# Predict probabilities on validation set
y_val_probs = model.predict(
    {
        "input_ids": X_val["input_ids"],
        "attention_mask": X_val["attention_mask"]
    }
)

best_thresholds = []

# Tune threshold per class using F1 score
for i in range(6):
    best_f1 = 0
    best_thres = 0.5
    for thresh in np.arange(0.1, 0.9, 0.01):
        preds = (y_val_probs[:, i] > thresh).astype(int)
        f1 = f1_score(y_val[:, i], preds)
        if f1 > best_f1:
            best_f1 = f1
            best_thres = thresh
    best_thresholds.append(best_thres)

# Apply best thresholds on test set
y_preds_probs =  model.predict(
    {
        "input_ids": X_test["input_ids"],
        "attention_mask": X_test["attention_mask"]
    }
)

y_test_pred = np.zeros_like(y_preds_probs, dtype = np.int8)

for i in range(6):
    y_test_pred[:, i] = (y_preds_probs[:, i] > best_thresholds[i]).astype(np.int8)

# Confusion matrices for each class
cm = multilabel_confusion_matrix(y_test, y_test_pred)
for i, matrix in enumerate(cm):
    print(f"Class {i}:\n{matrix}")
