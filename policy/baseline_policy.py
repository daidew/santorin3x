from .policy import *
import numpy as np

class RandomPolicy(Policy):
    '''
    Policy for sanity check.
    '''
    def __init__(self, seed=42):
        super(self, Policy).__init__()
        self.seed = seed

    def step(self, state):
        np.random.randint(10)

    def optimize_policy(self, data):
        pass


