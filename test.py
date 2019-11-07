import cv2
import numpy as np
from tensorflow.keras.models import load_model
from skimage import transform

model = load_model('./data/traffic_signs_10epochs.h5')

image = cv2.imread('4.jpg')
image = transform.resize(image,(32,32))

image = image.astype("float32") / 255.0
image = np.expand_dims(image, axis=0)
preds = model.predict(image)
j = preds.argmax(axis=1)[0]

print(j)

