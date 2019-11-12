from .policy import *
from environment.environment import *
import numpy as np

class RandomPolicy():
    '''
    This policy choose action randomly (uniform distribution)
    '''
    def __init__(self, seed=42):
        self.seed = seed

    def step(self, env):
        '''
        This method return the action index selected randomly from currently available move given current environment
        (which contain current state)
        '''
        #get legal moves for current player
        all_legal_moves = env.legal_moves()
        
        #convert all legal actions to action index
        all_legal_actions = []
        for legal_move in all_legal_moves:
            all_legal_actions.append(legal_move)
            
        #sample random action index uniformly from all legal actions
        return all_legal_actions[np.random.randint(len(all_legal_actions))]

    def optimize_policy(self, data):
        pass