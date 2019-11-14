import numpy as np


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


