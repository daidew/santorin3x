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

def get_resnet_k(k = 19):
    '''
    Return model with K ResNet Blocks
    '''

    inp = Input(shape=(5,5,3))
    
    #first conv block
    x = Conv2D(filters=256, kernel_size=(3,3), strides=1, padding='same')(inp)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    #resnet blocks (19?)
    for i in range(k//2):
        x = Conv2D(filters=256, kernel_size=(3,3), strides=1, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x2 = Conv2D(filters=256, kernel_size=(3,3), strides=1, padding='same')(x)
        x = x2 + x #residual
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        if i == 4:
            x = MaxPooling2D(pool_size=2)(x)

    p = Conv2D(filters=2, kernel_size=(1,1), strides=1, padding='same')(x)
    p = Flatten()(p)
    p = BatchNormalization()(p)
    p = Activation('relu')(p)
    p = Dense(128, activation='softmax')(p)

    v = Conv2D(filters=2, kernel_size=(1,1), strides=1, padding='same')(x)
    v = Flatten()(v)
    v = BatchNormalization()(v)
    v = Activation('relu')(v)
    v = Dense(256, activation='relu')(v)
    v = Dense(1, activation='tanh')(v)

    return Model(inputs=[inp], outputs=[p, v], name='ResNet{}'.format(k))

def get_VGG_k(k = 10):
    raise NotImplementedError
