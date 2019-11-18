import tensorflow as tf 
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
import numpy as np 

class DenseNetwork(Model):
    def __init__(self, hidden_size, act_f='relu'):
        super(DenseNetwork, self).__init__()
        self.ly = []
        for hs in hidden_size:
            self.ly.append(Dense(hs, activation=act_f))
        self.ly.append(Dense(128, activation='softmax'))

    def call(self, x):
        for f in self.ly:
            x = f(x)
        return x

class ConvolutionNetwork(Model):
    def __init__(self, act_f='relu'):
        super(ConvolutionNetwork, self).__init__()
        ly = []
        # ly.append(Conv2D(filters=, kernel_size=, strides=1, padding='same'))
        # ly.append(Conv2D)


    