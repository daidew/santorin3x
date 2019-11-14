import numpy as np
from engine.network.neuralnet import *

class OneStepAgent():
    
    def __init__(self, policy, player_id):
        self.policy = policy
        self.player_id = player_id
        
    def step(self, env):
        #get what action to make
        a = self.policy.step(env)

        #step on env
        s,r,done,current_player = env.step(a, switch_player=True)
        
        return s,a,r,done,current_player
    
class NStepAgent():

    def __init__():
        raise NotImplementedError()
        
    def step(self, env):
        raise NotImplementedError()


class DQNAgent():
    
    def __init__(self, lr=1e-4, tau=1e-3):
        self.local_network = DenseNetwork([256])
        self.target_network = DenseNetwork([256])
        self.target_network.set_weights(self.local_network.get_weights())
        self.lr = lr
        self.tau = tau
        
    def step(self, env):
        pass