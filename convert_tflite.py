import tensorflow as tf
from tensorflow.keras.models import load_model

# .h5 to .tflite
new_model= tf.keras.models.load_model(filepath="./data/traffic_signs_20_epochs.h5")
tflite_converter = tf.lite.TFLiteConverter.from_keras_model(new_model)
tflite_model = tflite_converter.convert()
open("./data/tf_lite_model.tflite", "wb").write(tflite_model)

""" .pb to .tflite
model.save(filepath="save_model/")
converter = tf.lite.TFLiteConverter.from_saved_model('save_model')
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)
"""
