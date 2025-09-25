import pandas as pd

df = pd.read_csv("heart.csv")

categorical_cols = ["Sex", "ST_Slope", "ExerciseAngina", "RestingECG", "ChestPainType"]

df = pd.get_dummies(df, columns=categorical_cols)

X = df.drop(columns =["Heart disease project"])
y = df["Heart disease project"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

import keras
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(6, activation="relu", input_shape = (X_train.shape[1],)),
    tf.keras.layers.Dense(6, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])

model.compile(optimizer = "adam",
              loss = "binary_crossentropy")

model.fit(
    X_train,
    y_train,
    epochs = 100,
    batch_size = 10,
    validation_split = 0.33
)