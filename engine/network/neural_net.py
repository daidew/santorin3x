import tensorflow as tf 
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.applications import MobileNetV2
import numpy as np 

class DenseNetwork(Model):
    def __init__(self, hidden_size, act_f='relu'):
        super().__init__()
        self.ly = []
        for hs in hidden_size:
            self.ly.append(Dense(hs, activation=act_f))
        self.ly.append(Dense(128, activation='softmax'))

    def call(self, x):
        for f in self.ly:
            x = f(x)
        return x

class ToyConvNet(Model):
    def __init__(self):
        super().__init__()
        filters = [16, 32, 64, 64, 128]
        kernels = [5,  5,  3,  3,  1]
        strides = [1,  1,  1,  1,  1]
        padding = ['same', 'same', 'same', 'same', 'same']
        self.ly = []
        for f, k, s, p in zip(filters, kernels, strides, padding):
            self.ly.append(Conv2D(filters=f, kernel_size=k, strides=s, padding=p))
            self.ly.append(BatchNormalization())
            self.ly.append(Activation('relu'))
        self.ly.append(Flatten())
        self.ly.append(Dense(256, activation='relu'))
        self.ly.append(BatchNormalization())
        self.ly.append(Dense(128, activation='softmax'))

    def call(self, x):
        for f in self.ly:
            x = f(x)
        return x
    
    def get_model(self):
        x = Input(shape=(5,5,3))
        return Model(inputs=[x], outputs=self.call(x), name='ToyConvNet')

class MobileNetConv(Model):
    def __init__(self):
        super().__init__()
        self.mobile_net = MobileNetV2(include_top=False, weights='imagenet')
        self.ly = []
        self.ly.append(Flatten())
        self.ly.append(Dense(128, activation='softmax'))

    def call(self, x):
        x = self.mobile_net(x)
        for f in ly:
            x = f(x)
        return x

class ResNet19(Model):
    def __init__(self):
        super().__init__()
        self.p_head = []
        self.v_head = []
        self.ly = []

        self.ly += self._gen_conv_b()
        for i in range(19):
            if i == 15:
                self.ly.append(MaxPooling2D(pool_size=2))
            self.ly += self._gen_res_b()

        self.p_head.append(Conv2D(filters=2, kernel_size=(1,1), strides=1, padding='same'))
        self.p_head.append(Flatten())
        self.p_head.append(BatchNormalization())
        self.p_head.append(Activation('relu'))
        self.p_head.append(Dense(128, activation='softmax'))

        self.v_head.append(Conv2D(filters=2, kernel_size=(1,1), strides=1, padding='same'))
        self.v_head.append(Flatten())
        self.v_head.append(BatchNormalization())
        self.v_head.append(Activation('relu'))
        self.v_head.append(Dense(256))
        self.v_head.append(Activation('relu'))
        self.v_head.append(Dense(1, activation='tanh'))

    def _gen_conv_b(self):
        conv_b = []
        conv_b.append(Conv2D(filters=256, kernel_size=(3,3), strides=1, padding='same'))
        conv_b.append(BatchNormalization())
        conv_b.append(Activation('relu'))
        return conv_b

    def _gen_res_b(self):
        res_b = []
        res_b += self._gen_conv_b()
        res_b.append(BatchNormalization())
        res_b.append(Activation('relu'))
        return res_b
    
    def call(self, x):
        for f in self.ly:
            x = f(x)
        x1 = x
        x2 = x

        for f in self.p_head:
            x1 = f(x1)
    
        for f in self.v_head:
            x2 = f(x2)

        return [x1, x2]

    def get_model(self):
        x = Input(shape=(5,5,3))
        return Model(inputs=[x], outputs=self.call(x), name='ResNet19')
