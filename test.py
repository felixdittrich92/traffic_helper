import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from skimage import transform
from os import listdir
from pathlib import Path
from PIL import Image

classes = ['Tempo: 20', 'Tempo: 30', 'Tempo: 50', 'Tempo: 60', 'Tempo: 70',
            'Tempo: 80', 'Auflösung 80', 'Tempo: 100', 'Tempo: 120', 'Überholverbot',
            'LKW Überholverbot', 'Vorfahrt nächste Kreuzung', 'Vorfahrtsstraße',
            'Vorfahrt gewehren', 'Stop', 'Einfahrt verboten', 'LKW Einfahrt verboten',
            'Einfahrt verboten Einbahnstraße', 'Achtung', 'Achtung Kurve Links', 'Achtung Kurve Rechts',
            'Achtung Kurvenkombination', 'Achtung unebene Fahrbahn', 'Achtung Schleudergefahr',
            'einseitig verengte Fahrbahn', 'Achtung Baustelle', 'Achtung Ampel', 'Achtung Fußgänger',
            'Achtung Kinder', 'Achtung Fahrrad', 'Achtung Glätte', 'Achtung Wildwechsel', 'Auflösung',
            'Zwangspfeil Rechts', 'Zwangspfeil Links', 'Zwangspfeil Geradeaus', 'Zwangspfeil Geradeaus und Rechts',
            'Zwangspfeil Geradeaus und Links', 'Vorbeifahrt Rechts', 'Vorbeifahrt Links', 'Kreisverkehr', 'Auflösung Überholverbot',
            'Auflösung LKW Überholverbot']

model = load_model('./data/traffic_signs_50_epochs.h5')

"""
def load_own_image(filepath):
    image = cv2.imread(filepath)
    image = transform.resize(image, (30, 30))
    image = np.expand_dims(image, axis=0)
    return image

folder = './test_bilder/'
for file in listdir(folder):
    path = Path(folder, file)
    path = "./" + str(path)
    image = load_own_image(path)
    print(path)
    predictions = model.predict(image)[0]
    pred_class = np.argmax(predictions)
    class_name = classes[pred_class]
    if predictions[pred_class] < 0.5:
        print(f"Prediction Fail: {path}")
    else:
        print(f"Class Name: {class_name} Predicted Class: {pred_class} Accuracy: {predictions[pred_class]}")

#ToDo: weitere Tests inkl. Video
"""

def load_frame(frame):
    frame = transform.resize(frame, (30, 30))
    frame = np.expand_dims(frame, axis=0)
    return frame

cap = cv2.VideoCapture(0)
index = 0

while(True):
    if index % 25 == 0:
        print(index)
        ret, frame = cap.read()

        #copy = load_frame(frame)
        image = Image.fromarray(frame, 'RGB')
        image = image.resize((30,30))
        img_array = np.array(image)
        img_array = np.expand_dims(img_array, axis=0).astype('float16')
        predictions = model.predict(img_array)[0]
        pred_class = np.argmax(predictions)
        class_name = classes[pred_class]
        index += 1
    else:
        print(':D')
    
    if predictions[pred_class] < 0.5:
        pass
    else:
        print(f"Class Name: {class_name} Predicted Class: {pred_class} Accuracy: {predictions[pred_class]}")
    
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
