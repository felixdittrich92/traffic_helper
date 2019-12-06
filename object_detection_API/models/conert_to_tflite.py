import tensorflow as tf
gf = tf.compat.v1.GraphDef()
m_file = open('traffic_graph.pb','rb')
for n in gf.node:
    print( n.name )


graph_def_file = "traffic_graph.pb"
input_arrays = ["image_tensor"] #ToDo
output_arrays = ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes']

converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
        graph_def_file, input_arrays, output_arrays)
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)