import cv2
import numpy as np
from tensorflow.keras.models import load_model
from skimage import transform

model = load_model('./data/traffic_signs_50_epochs.h5')

def load_own_image(filepath):
    image = cv2.imread(filepath)
    image = transform.resize(image, (30, 30))
    image = np.expand_dims(image, axis=0)
    return image

image = load_own_image('./test_bilder/60_sign.jpg')
predictions = model.predict(image)[0]
pred_class = np.argmax(predictions)
print(f"Predicted Class: {pred_class} Accuracy: {predictions[pred_class]}")

image = load_own_image('./test_bilder/stop_sign.jpg')
predictions = model.predict(image)[0]
pred_class = np.argmax(predictions)
print(f"Predicted Class: {pred_class} Accuracy: {predictions[pred_class]}")

image = load_own_image('./test_bilder/vorfahrt.jpg')
predictions = model.predict(image)[0]
pred_class = np.argmax(predictions)
print(f"Predicted Class: {pred_class} Accuracy: {predictions[pred_class]}")

#ToDo: weitere Tests inkl. Video

