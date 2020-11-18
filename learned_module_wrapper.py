import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf


def load_graph(frozen_graph_filename):
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="prefix")
    return graph


# function wrapper for learned model of forward, identity and readout layers
class gol_fun:

    def __init__(self, filename="recognition_CNN/networks/gray_deep-gpu_net.pb",
                 input_tensor_name="prefix/inputs:0", output_tensor_names=['prefix/predictions/Sigmoid:0']):
        self.graph = load_graph(filename)

        self.inputs = self.graph.get_tensor_by_name('prefix/inputs:0')
        self.outputs = [self.graph.get_tensor_by_name(name) for name in output_tensor_names]

        # We launch a Session
        config = tf.ConfigProto(
            device_count={'GPU': 0}
        )
        self.sess = tf.Session(graph=self.graph, config=config)

    def predict(self, input_batch):
        prediction_probs = self.sess.run(self.outputs, feed_dict={
            self.inputs: input_batch,
        })
        predictions = (np.array(prediction_probs) > .5).astype(np.int8)
        return predictions, prediction_probs


# function wrapper for learned model of the counter and control network
class gol_fun_counter:

    def __init__(self, filename="recognition_CNN/networks/gray_deep-gpu_net.pb",
                 input_tensor_name="prefix/inputs:0", output_tensor_names=['prefix/predictions/Sigmoid:0']):
        self.graph = load_graph(filename)

        self.inputs = self.graph.get_tensor_by_name('prefix/inputs:0')
        self.outputs = [self.graph.get_tensor_by_name(name) for name in output_tensor_names]

        # We launch a Session
        self.sess = tf.Session(graph=self.graph)

    def predict(self, input_batch):
        counts, decisions = self.sess.run(self.outputs, feed_dict={
            self.inputs: input_batch,
        })
        return counts, decisions
