import cv2
import numpy as np
from tensorflow.keras.models import load_model
from skimage import transform

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

def load_own_image(filepath):
    image = cv2.imread(filepath)
    image = transform.resize(image, (30, 30))
    image = np.expand_dims(image, axis=0)
    return image

image = load_own_image('./test_bilder/60_sign.jpg')
predictions = model.predict(image)[0]
pred_class = np.argmax(predictions)
print(classes[pred_class])
print(f"Predicted Class: {pred_class} Accuracy: {predictions[pred_class]}")

image = load_own_image('./test_bilder/stop_sign.jpg')
predictions = model.predict(image)[0]
pred_class = np.argmax(predictions)
print(classes[pred_class])
print(f"Predicted Class: {pred_class} Accuracy: {predictions[pred_class]}")

image = load_own_image('./test_bilder/vorfahrt.jpg')
predictions = model.predict(image)[0]
pred_class = np.argmax(predictions)
print(classes[pred_class])
print(f"Predicted Class: {pred_class} Accuracy: {predictions[pred_class]}")

#ToDo: weitere Tests inkl. Video
