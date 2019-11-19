import numpy as np
from engine.network.neural_net import *
from collections import deque
from tensorflow.keras.optimizers import *
import datetime

class MCTS():
    def __init__(self):
        self.Q = {}
        self.P = {}
        self.N = {}
        self.W = {}

    def expand(self):
        pass

class AlphaSantorini():
    def __init__(self):
        self.mcts = MCTS()
        self.q_network = DenseNetwork()

    def step(self, env):
        pass

