import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from layers import *
from input_sequences import SequenceDatabase
import os

class RecurrentNetwork(object):

    def __init__(self, **kwargs):
        self.dataset_type = kwargs["dataset_type"]
        self.sequence_length = kwargs["sequence_length"]
        self.logdir = kwargs["logdir"]
        self.batch_size = kwargs["batch_size"]
        self.dataset = SequenceDatabase(**kwargs)

        self.sess = tf.Session()

        if self.dataset_type == "regression":
            self.build_regression_model()
        elif self.dataset_type == "classification":
            self.build_classification_model()

        self.summ = tf.summary.merge_all()
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())
        self.writer1 = tf.summary.FileWriter(self.logdir + "/train")
        self.writer2 = tf.summary.FileWriter(self.logdir + "/test")
        self.writer1.add_graph(self.sess.graph)

    def build_regression_model(self):
        # tf Graph input
        self.x = tf.placeholder("float", [self.batch_size, self.sequence_length, 1],
                                name="input_sequence")
        self.y = tf.placeholder("float", [self.batch_size, self.sequence_length],
                                name="target_value")
        self.train_phase = tf.placeholder("bool", name="train_phase")
        self.batch_norm_layer = BatchNormalizationLayer(input_tensor=self.x,
                                                        output_channels=[self.sequence_length, 1],
                                                        phase_train=self.train_phase,
                                                        layer_name="batch_norm_layer")

        self.lstm_layer = LSTMLayer(input_tensor=self.x,
                                    input_dim=1,
                                    n_hidden=20,
                                    seq_length=self.sequence_length,
                                    layer_name="lstm_layer")

        self.output_layer = FullyConnectedLayer(input_tensor=self.lstm_layer.output_tensor,
                                                weights_shape=[20, 1],
                                                layer_name="output_layer",
                                                activation="linear",
                                                dataset_type=self.dataset_type)

        # Define loss and optimizer
        self.cost = tf.reduce_mean(tf.squared_difference(x=self.y,
                                                         y=self.output_layer.output_tensor))
        tf.summary.scalar("mse", self.cost)
        self.optimizer = tf.train.AdamOptimizer()
        self.train_all_params = self.optimizer.minimize(self.cost)

    def build_classification_model(self):
        # tf Graph input
        self.x = tf.placeholder("float", [self.batch_size, self.sequence_length, 1],
                                name="input_sequence")
        self.y = tf.placeholder("float", [self.batch_size, 4],
                                name="target_value")
        self.train_phase = tf.placeholder("bool", name="train_phase")
        self.batch_norm_layer = BatchNormalizationLayer(input_tensor=self.x,
                                                        output_channels=[self.sequence_length, 1],
                                                        phase_train=self.train_phase,
                                                        layer_name="batch_norm_layer")

        self.lstm_layer = LSTMLayer(input_tensor=self.x,
                                    input_dim=1,
                                    n_hidden=20,
                                    seq_length=self.sequence_length,
                                    layer_name="lstm_layer",
                                    dataset_type=self.dataset_type)

        self.output_layer = FullyConnectedLayer(input_tensor=self.lstm_layer.output_tensor,
                                                weights_shape=[20, 4],
                                                layer_name="output_layer",
                                                activation="linear",
                                                dataset_type=self.dataset_type)

        # Define loss and optimizer
        with tf.name_scope("loss_function"):
            self.model_output = self.output_layer.output_tensor

            self.softmax_output = tf.nn.softmax(self.model_output, name="model_prob")

            self.cost = tf.reduce_mean(
                        tf.nn.softmax_cross_entropy_with_logits(
                            logits=self.model_output,
                            labels=self.y,
                            name='loss'))

            tf.summary.scalar("cross_entropy", self.cost)
            self.optimizer = tf.train.AdamOptimizer()
            self.train_all_params = self.optimizer.minimize(self.cost)

        with tf.name_scope("accuracy"):
            self.correct_predictions = tf.equal(tf.argmax(self.model_output, 1),
                                                tf.argmax(self.y, 1), name="correct_pred")
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, tf.float32), name='accuracy')
            self.acc_sum = tf.summary.scalar("accuracy", self.accuracy)

    def train_iterations(self, n_iterations):

        for iteration in range(n_iterations):
            batch_x, batch_y = self.dataset.next_batch()
            #print(batch_y.shape)
            #print(batch_x.shape)
            if len(batch_x.shape) == 1:
                continue
            _ = self.sess.run(self.train_all_params, feed_dict={
                self.x: batch_x,
                self.y: batch_y,
                self.train_phase: True
            })
            if iteration % 10 == 0:
                train_metric, s = self.sess.run([self.cost, self.summ], feed_dict={
                    self.x: batch_x,
                    self.y: batch_y,
                    self.train_phase: False
                })
                self.writer1.add_summary(s, iteration)
            if iteration % 100 == 0:
                if self.dataset_type == "regression":
                    print("train mse = " + str(train_metric))
                else:
                    print("train cross_entropy = "+str(train_metric))
                self.saver.save(self.sess, os.path.join(self.logdir, "model.ckpt"), iteration)
            if iteration % 10 == 0:
            #if False:
                val_batch_x, val_batch_y = self.dataset.next_batch(sub_set="test")
                if len(val_batch_x.shape) == 1:
                    continue
                test_metric, test_acc_sum = self.sess.run([self.cost, self.summ], feed_dict={
                    self.x: val_batch_x,
                    self.y: val_batch_y,
                    self.train_phase: False
                })
                self.writer2.add_summary(test_acc_sum, iteration)
                # print("test mse = " + str(test_metric))

        self.saver.save(self.sess, os.path.join(self.logdir, "final_model"))
        self.saver.export_meta_graph(os.path.join(self.logdir, "final_model.meta"))


if __name__ == "__main__":
    set_prop = [0.8, 0.2]
    logdir = "/home/rodrigo/lstm_test/results/"
    experiment_name = "classification_15"
    model = RecurrentNetwork(batch_size=128,
                             dataset_type="classification",
                             set_prop=set_prop,
                             sequence_length=15,
                             n_points=100,
                             logdir=logdir+experiment_name,
                             n_examples=1000000)
    model.train_iterations(n_iterations=100000)