import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

class Policy(tf.keras.Model):
    
    def __init__(self, input_dim, action_dim, activation_func = tf.nn.relu, h_neurons=256, regularizer=tf.keras.regularizers.l2(0.0001)):
        super(Policy, self).__init__()

        self.action_dim = action_dim

        #vector input for policy
        if(len(input_dim) == 1):
            self.l1 = tf.keras.layers.Dense(units=h_neurons, activation=activation_func, kernel_regularizer=regularizer)
            self.l2 = tf.keras.layers.Dense(units=h_neurons, activation=activation_func, kernel_regularizer=regularizer)
            self.l3 = tf.keras.layers.Dense(units=action_dim[0], activation=tf.keras.activations.softmax, kernel_regularizer=regularizer)
        #image input
        else:
            raise NotImplementedError

    def regularization_loss(self):
        return tf.math.reduce_sum(self.l1.losses) + tf.math.reduce_sum(self.l2.losses) + tf.math.reduce_sum(self.l3.losses)

    def call(self, input, training=False):
        #Feed Policy
        x = self.l1(input)
        x = self.l2(x)
        x = self.l3(x)

        return x

class CnnPolicy(tf.keras.Model):

    def __init__(self, action_dim, activation_func = tf.nn.relu, regularizer=tf.keras.regularizers.l2(0.0001)):
        super(CnnPolicy, self).__init__()

        self.action_dim = action_dim

        self.c1 = tf.keras.layers.Conv2D(32, 3, activation=activation_func, kernel_regularizer=regularizer)
        self.c2 = tf.keras.layers.Conv2D(64, 3, activation=activation_func, kernel_regularizer=regularizer)
        self.f3 = tf.keras.layers.Dense(units = action_dim, activation=tf.keras.activations.softmax, kernel_regularizer=regularizer)

    #TODO Regularization Loss?

    def call(self, input, training=False):

        x = self.c1(input)
        x = tf.keras.layers.MaxPool2D()(x)
        x = self.c2(x)
        x = tf.keras.layers.MaxPool2D()(x)
        x = tf.keras.layers.Flatten()(x)
        return self.f3(x)