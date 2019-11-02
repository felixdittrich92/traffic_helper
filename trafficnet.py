import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import to_categorical
import numpy as np
import cv2
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # dont show warnings from Tensorflow

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

print("Trainingsdaten:", len(X_train))
print("Testdaten:", len(y_test))
print("Validierungsdaten:", len(X_valid))
print("Bilddimensionen:", np.shape(X_train[1]))
print("Anzahl der Klassen:", len(np.unique(y_train)))

n_classes = 43

model = Sequential()

model.add(Dense(64, activation='relu', input_shape=(32, 32, 3,)))
model.add(BatchNormalization())

model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())

model.add(Dense(n_classes, activation='softmax'))

model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

X_train_baseline = X_train.reshape(len(X_train), 32, 32, 3).astype('float32')
X_valid_baseline = X_valid.reshape(len(X_valid), 32, 32, 3).astype('float32')
y_train_baseline = tf.keras.utils.to_categorical(y_train, n_classes)
y_valid_baseline = tf.keras.utils.to_categorical(y_valid, n_classes)

model.fit(X_train_baseline, y_train_baseline, batch_size=128, epochs=10, verbose=1, validation_data=(X_valid_baseline, y_valid_baseline))

X_test_baseline = X_test.reshape(len(X_test), 32, 32, 3).astype('float32')
y_test_baseline = tf.keras.utils.to_categorical(y_test, n_classes)

model.evaluate(X_test_baseline, y_test_baseline, verbose=0)

model.save('traffic_signs_100epochs.h5', save_format='h5')
