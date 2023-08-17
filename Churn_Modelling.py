
import pandas as pd 

dataset = pd.read_csv("Churn_Modelling.csv")
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

# DATA PREPROCESSING

# Encoding Categorical Data

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])

ct = ColumnTransformer(transformers=[("encoder", OneHotEncoder(), [1])], remainder="passthrough")
X = ct.fit_transform(X)

# Splitting Train, Test

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0, test_size= 0.2)

# Feature Scalling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# ARTIFICIAL NEURAL NETWORK 

import tensorflow as tf 

ann = tf.keras.models.Sequential()

# input layer and first hidden layer
ann.add(tf.keras.layers.Dense(units = 6, activation = "relu"))

# second hidden layer
ann.add(tf.keras.layers.Dense(units = 6, activation = "relu"))

# output layer
ann.add(tf.keras.layers.Dense(units = 1, activation = "sigmoid"))

# compiling ANN
ann.compile(optimizer= "adam", loss = "binary_crossentropy", metrics= ["accuracy"])
ann.fit(X_train, y_train, batch_size = 32, epochs=30)

# predict a result with single observation
single_obervation = ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 5000]]))
print(single_obervation)

# ACCURACY
from sklearn.metrics import accuracy_score

y_pred = ann.predict(X_test)
y_pred = y_pred > 0.5

acu = accuracy_score(y_pred, y_test)
print(acu)