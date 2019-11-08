import cv2

import pickle

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from skimage import transform

training_image_path = "data/train.p"
with open(training_image_path, mode='rb') as file:
    train = pickle.load(file)
print(train['labels'])

model = load_model('./data/traffic_signs_10epochs.h5')

image = cv2.imread('4.jpg')
image = transform.resize(image,(32,32))
print(image.shape)

image = image.astype("float32") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)
pred = model.predict_classes(image)[0]


print(pred)

