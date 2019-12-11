# Doesn´t work with TF2.0 :( !!
import tensorflow as tf

graph_def_file = "traffic_graph.pb"
input_arrays = ["image_tensor"] 
output_arrays = ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes']

converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
        graph_def_file, input_arrays, output_arrays, input_shapes={"image_tensor":[1,800,600,3]})
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)