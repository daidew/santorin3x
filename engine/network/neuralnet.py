import tensorflow as tf 
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
import numpy as np 

class DenseNetwork(Models, act_f='relu'):
    def __init__(self, hidden_size):
        super(DenseNetwork, self).__init__()
        self.ly = []
        for hs in hidden_size:
            self.ly.append(Dense(hs, activation=act_f))
        self.ly.append(Dense(1, activation='linear'))

    def call(x):
        for f in self.ly:
            x = f(x)
        return x

class ConvolutionNetwork(Models, act_f='relu'):
    def __init__(self):
        super(ConvolutionNetwork, self).__init__()
        ly = []
        ly.append(Conv2D(filters=, kernel_size=, strides=1, padding='same'))
        ly.append(Conv2D)


    