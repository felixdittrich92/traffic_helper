import cv2

import pickle

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from skimage import transform


model = load_model('traffic_signs_10_epochs.h5')

image = cv2.imread('3.jpg')
image = transform.resize(image,(30,30))
print(image.shape)

image = image.astype("float32") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)
pred = model.predict_classes(image)[0]


print(pred)

https://www.kaggle.com/rkuo2000/gtsrb-cnn