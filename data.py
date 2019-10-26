import pickle
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

testing_image_path = "data/test.p"
training_image_path = "data/train.p"
validation_image_path = "data/valid.p"

with open(training_image_path, mode='rb') as file:
    train = pickle.load(file)
with open(testing_image_path, mode='rb') as file:
    test = pickle.load(file)
with open(validation_image_path, mode='rb') as file:
    valid = pickle.load(file)

X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']
X_valid, y_valid = valid['features'], valid['labels']

print(X_train.shape)
print(y_train.shape)
print("Trainingsdaten:", len(X_train))
print("Testdaten:", len(y_test))
print("Validierungsdaten:", len(X_valid))
print("Bilddimensionen:", np.shape(X_train[1]))
print("Anzahl der Klassen:", len(np.unique(y_train)))