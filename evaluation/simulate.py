import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from progressbar import progressbar
import sys
sys.path.append("..") # Add s higher directory to python modules path.
from environment.environment import *
from engine.agent.base_agent import *
from engine.policy.base_policy import *

def simulate(n_eps, verbose=False):
    '''
    This method simulates the game by count of n_eps episodes, and analyze the outcome/turns.
    Param:
        n_eps: int;  the number of simulations to run
        verbose: boolean default True; whether to enable output logging and board visualization or not
    Returns:
        turns: np.ndarray; the count of total turns in each episode
        winner: the winner of that game; 1 = Player 1, 0 = Draw, 2 = Player 2
    '''
    turns = []
    winner = []
    for i in progressbar(range(n_eps)):
        #initiate env
        env = Santorini()

        #initiate policy
        policy = RandomPolicy()

        #initiate players
        player1 = OneStepAgent(policy, -1)
        player2 = OneStepAgent(policy,  1)

        #first player is player1
        current_player=-1
        env.current_player=current_player
        done = False
        
        while True:
            if verbose:
                print('Turn:{}, Score={}, Done={}, cur_player={}'.format(env.turns,env.score(),done,current_player))
                #visualize board
                env.print_board(mode=1)
 
            if env.turns > 500: #must be bug
                print('\nTurn:{}, Score={}, Done={}, cur_player={} latest_action={}'.format(env.turns,env.score(),done,current_player,a))
                env.print_board(mode=1)
                raise ValueError('Exceed maximum turn')

            if current_player == -1:
                s, a, r, done, next_player = player1.step(env)
            else:
                s, a, r, done, next_player = player2.step(env)
            
            
            #check if board is done
            if done:
                #append number of turns
                turns.append(env.turns)
                if verbose:
                    print('====== Done ======')
                    print('*Turn:{}, Score={}, Done={}, cur_player={}'.format(env.turns,env.score(),done,current_player))
                    #visualize board
                    env.print_board(mode=1)
                
                if r == 1:
                    if current_player == -1:
                        if verbose:
                            print('Player 1 Won')
                        winner.append(1)
                    else:
                        if verbose:
                            print('Player 2 Won')
                        winner.append(2)
                elif r == -1:
                    if current_player == -1:
                        if verbose:
                            print('Player 2 Won (Player 1 could not make a move)')
                        winner.append(2)
                    else:
                        if verbose:
                            print('Player 1 Won (Player 2 could not make a move)')
                        winner.append(1)
                else:
                    if verbose:
                        print('Draw')
                    winner.append(0)
                break
                
            #switch player
            current_player *= -1

    return np.array(turns), np.array(winner)

if __name__ == '__main__':
    print('enter number of episode to simulate')
    n_eps = int(input())
    print('simulating...')
    turns, winner = simulate(n_eps, verbose=False)
    print('Average number of turns: ', np.mean(turns))
    print('Minimum number of turns: ', np.min(turns))
    print('Maximum number of turns: ', np.max(turns))
    print('Standard deviation of turns: ',np.std(turns))
    print('Player1 won {} games'.format(np.sum(winner[winner == 1])))
    print('Player2 won {} games'.format(np.sum(winner[winner == 2])))
    print('Draw {} games'.format(np.sum(winner[winner == 0])))
    plt.title('total turns plot')
    plt.plot(np.arange(n_eps), turns )
    plt.xlabel('eps')
    plt.ylabel('turns')
    plt.show()
    plt.title('winner plot')
    plt.xlabel('eps')
    plt.ylabel('winner')
    plt.plot(np.arange(n_eps), winner)
    plt.show()