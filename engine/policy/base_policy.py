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
        if len(all_legal_moves) == 0: #no legal moves available
            return -1

        #convert all legal actions to action index
        all_legal_actions = []
        for legal_move in all_legal_moves:
            all_legal_actions.append(legal_move)
            
        #sample random action index uniformly from all legal actions
        return all_legal_actions[np.random.randint(len(all_legal_actions))]

    def optimize_policy(self, data):
        pass

class HumanPlayer():
    '''
    Interactive human player which able receive input and act accordingly.
    '''
    def __init__(self):
        pass

    def step(self, env):
        all_legal_moves = env.legal_moves()
        if len(all_legal_moves) == 0: #no legal moves available
            return -1
        
        #convert all legal actions to action index
        all_legal_actions = []
        for legal_move in all_legal_moves:
            all_legal_actions.append(legal_move)

        while True:
            #(ex. -1 a q)
            w, m, b = input('Please enter next move: ')
            action_idx = env.atoi[(int(w), m, b)]
            if action_idx not in all_legal_moves:
                print('You entered illegal moves. Please try again.')
            else:
                return action_idx

