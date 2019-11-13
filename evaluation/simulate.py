import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append("..") # Add s higher directory to python modules path.
from environment.environment import *
from engine.agent.base_agent import *
from engine.policy.base_policy import *

def simulate():
    #initiate env
    env = Santorini()
    #initiate policy
    policy = RandomPolicy()
    #initiate players
    player1 = OneStepAgent(policy, -1)
    player2 = OneStepAgent(policy,  1)

    #automatically switches player in env.step()
    env = Santorini()
    n_eps = 1
    current_player=-1
    #set current_player to be -1
    env.current_player=current_player
    while True:
        #visualize board
        env.print_board(mode=1)
        if current_player == -1:
            s, a, r, done, next_player = player1.step(env)
        else:
            s, a, r, done, next_player = player2.step(env)
        print('Turn:{}, Score={}, Done={}'.format(env.turns,env.score(),done))

        if a == -1:
            print('Draw (no legal move available)')
            break
        #check if board is done
        if done:
            #visualize board
            env.print_board(mode=1)
            if r == 1:
                if current_player == -1:
                    print('Player 1 Won')
                else:
                    print('Player 2 Won')
            else:
                print('Draw')
            break
        #switch player
        current_player = next_player

simulate()