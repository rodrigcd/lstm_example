import tensorflow as tf
import numpy as np


class FullyConnectedLayer(object):

    def __init__(self, input_tensor, weights_shape, layer_name, activation="relu", dataset_type="regression"):

        self.input_tensor = input_tensor
        self.layer_name = layer_name
        self.weights_shape = weights_shape
        self.activation_function = activation

        with tf.variable_scope(layer_name):
            self.weights = tf.Variable(tf.truncated_normal(self.weights_shape, stddev=0.1),
                                       name="weights")
            self.biases = tf.Variable(tf.constant(0.1, shape=[self.weights_shape[1]]),
                                      name="biases")
            print(input_tensor.get_shape())
            if dataset_type == "regression":
                print("regression fully connected layer")
                print(self.weights.get_shape())
                linear_out = tf.tensordot(self.input_tensor, self.weights,
                                          axes=[[1], [0]])
                self.linear_out = tf.reshape(linear_out,
                                             shape=linear_out.get_shape()[:2])
            else:
                print("classification fully connected layer")
                self.linear_out = tf.matmul(self.input_tensor, self.weights) + self.biases

            if activation == "relu":
                self.output_tensor = tf.nn.relu(self.linear_out, name="output")
            elif activation == "tanh":
                self.output_tensor = tf.nn.tanh(self.linear_out, name="output")
            elif activation == "linear":
                self.output_tensor = tf.identity(self.linear_out, name="output")
            else:  # sigmoid
                self.output_tensor = tf.sigmoid(self.linear_out, name="output")
            tf.summary.histogram("weights", self.weights)
            tf.summary.histogram("biases", self.biases)
            tf.summary.histogram("activation", self.output_tensor)


class LSTMLayer(object):

    def __init__(self, input_tensor, input_dim, n_hidden, seq_length, layer_name, dataset_type="regression"):

        self.input_dim = input_dim
        self.input_tensor = input_tensor
        self.layer_name = layer_name
        self.n_hidden = n_hidden
        self.batch_length = tf.shape(input_tensor)[0]
        sequence_length_array = tf.ones(shape=[self.batch_length])*seq_length

        with tf.variable_scope(layer_name):
            x = tf.unstack(input_tensor, seq_length, input_dim, name="unstack")
            self.lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)
            outputs, self.states = tf.contrib.rnn.static_rnn(self.lstm_cell, x,
                                                             dtype=tf.float32,
                                                             sequence_length=sequence_length_array)

            outputs = tf.stack(outputs, name="re_stack")
            print("lstm: ", outputs.get_shape())
            if dataset_type == "classification":
                print("classification lstm layer")
                outputs = tf.transpose(outputs, [1, 0, 2])
                # Hack to build the indexing and retrieve the right output.
                batch_size = tf.shape(outputs)[0]
                # Start indices for each sample
                index = tf.range(0, batch_size) * seq_length + (seq_length - 1)
                # Indexing
                self.output_tensor = tf.gather(tf.reshape(outputs, [-1, n_hidden]), index, name="gather")
            else:
                print("regression lstm layer")
                outputs = tf.transpose(outputs, [1, 2, 0])
                self.output_tensor = outputs


class BatchNormalizationLayer(object):

    def __init__(self, input_tensor, output_channels, phase_train, layer_name):

        with tf.variable_scope(layer_name):
            self.beta = tf.Variable(tf.constant(0.0, shape=output_channels),
                                    name='beta', trainable=True)
            self.gamma = tf.Variable(tf.constant(1.0, shape=output_channels),
                                     name='gamma', trainable=True)
            batch_mean, batch_var = tf.nn.moments(input_tensor, [0], name='moments')
            ema = tf.train.ExponentialMovingAverage(decay=0.5)

            def mean_var_with_update():
                ema_apply_op = ema.apply([batch_mean, batch_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)

            mean, var = tf.cond(phase_train,
                                mean_var_with_update,
                                lambda: (ema.average(batch_mean), ema.average(batch_var)))
            self.output_tensor = tf.nn.batch_normalization(input_tensor, mean, var, self.beta, self.gamma, 1e-3)