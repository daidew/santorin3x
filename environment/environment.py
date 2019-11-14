import numpy as np
from collections import deque
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
'''
========================================================================================
                                        READ ME

Credit: This Santorini environment was created by cstorm125
Reference: https://github.com/cstorm125/santorini/blob/master/santorinigo/environment.py

Santorini is the environment for the agent.
========================================================================================
'''

class Santorini:
    def __init__(self, board_dim = (5,5), starting_parts = np.array([0,22,18,14,18]),
                 history_len = 3, winning_floor=3):
        #optional rules for curriculum learning
        self.winning_floor = winning_floor
        
        #history recorder
        self.history_len = history_len
        self.buildings_layers = deque(maxlen=history_len)
        self.minus_worker1_layers = deque(maxlen=history_len)
        self.minus_worker2_layers = deque(maxlen=history_len)
        self.plus_worker1_layers = deque(maxlen=history_len)
        self.plus_worker2_layers = deque(maxlen=history_len)
        
        #action_space: 2 workers * 8 moves * 8 builds = 128 options
        #moves/builds: q,w,e,a,d,z,x,c
        self.board_dim = board_dim
        self.workers = [-1,-2]
        self.moves = ['q','w','e','a','d','z','x','c']
        self.builds = ['q','w','e','a','d','z','x','c']
        #board[buildings/workers, vertical, horizontal]
        #key to coordinates
        self.ktoc = {'q':(-1,-1),
                     'w':(-1,0),
                     'e':(-1,1),
                     'a':(0,-1),
                     'd':(0,1),
                     'z':(1,-1),
                     'x':(1,0),
                     'c':(1,1)}
        #index to action
        self.itoa = [(w,m,b) for w in self.workers for m in self.moves for b in self.builds]
        #action to index
        self.atoi = {action:index for index,action in enumerate(self.itoa)}
        self.action_dim = len(self.itoa)
        
        self.reset(board_dim, starting_parts)
        self.state_dim = self.get_board().shape

    def reset(self, board_dim = (5,5), starting_parts=np.array([0,22,18,14,18])):        
        #turn counter
        self.turns = 0
        
        #building pieces
        #floor, base, mid, top, dome
        self.starting_parts = starting_parts
        
        #keep track of players
        #-1, 0 , 1 for player 1, blank, player 2
        self.current_player = -1
        
        #three layers: buildings, workers, building parts
        self.board_dim = board_dim
        self.board = np.zeros((3, board_dim[0], board_dim[1]), dtype=np.int64)
        np.fill_diagonal(self.board[2,:,:],np.array(self.starting_parts))
        
        self.board[1,0,2], self.board[1,4,2] = -1, -2 #negative workers for player 1
        self.board[1,2,0], self.board[1,2,4] =  1, 2 #positive workers for player 2
        
        #history recorder
        for i in range(self.history_len): self.record_state()

        return(self.get_board()) 
    
    def print_board(self, mode=0):
        '''
        mode 0: print numpy array
        
        mode 1: beautify the board using seaborn magic
        '''
        if mode == 0:
            print(f'Buildings:\n {self.board[0,:,:]}')
            print(f'Workers:\n {self.board[1,:,:]}')
            print(f'Parts:\n {self.board[2,:,:]}')
        elif mode == 1:
            fig,ax = plt.subplots(1, 3, figsize=(13, 4))
            if self.current_player == -1:
                _player = 'Player 1'
            else:
                _player = 'Player 2'
            fig.suptitle('Turn:{} ({} to move)'.format(self.turns, _player))
            ax[0].set_title('Buildings')
            sns.heatmap(self.board[0,:,:],annot=True,yticklabels=False,xticklabels=False,
                        annot_kws={'ha':'center','va':'center'},ax=ax[0],cbar=False,
                        linewidths=1,linecolor='white'
                       )
            ax[1].set_title('Workers')
            sns.heatmap(self.board[1,:,:],yticklabels=False,xticklabels=False,
                        annot=True,ax=ax[1],cbar=False,cmap='RdBu',
                       linewidths=1,linecolor='white'
                       )
            ax[2].set_title('Parts')
            sns.heatmap(self.board[2,:,:],yticklabels=False,xticklabels=False,
                        annot=True,ax=ax[2],cbar=False,
                       linewidths=1,linecolor='white'
                       )
            plt.show()
            
    def get_buildings_layer(self):
        buildings_layer = self.board[0,:,:].copy()
        return(buildings_layer)
    
    def get_worker_layer(self, worker):
        idx = np.where(self.board[1,:,:]==worker)
        worker_layer = np.zeros(self.board_dim)
        worker_layer[idx] = 1
        return(worker_layer)
    
    def record_state(self):
        self.buildings_layers.append(self.get_buildings_layer())
        self.minus_worker1_layers.append(self.get_worker_layer(-1))
        self.minus_worker2_layers.append(self.get_worker_layer(-2))
        self.plus_worker1_layers.append(self.get_worker_layer(1))
        self.plus_worker2_layers.append(self.get_worker_layer(2))
    
    def get_board(self, no_parts= True):
        '''
        return whole board in numpy
        '''
        return self.board
    
    def get_canonical_board(self, no_parts= False):
        '''
        canonical board
        '''
        #current player has negative workers; opposing player has positive workers
        sgn = -np.sign(self.current_player)
        state = self.board.copy()
        #if current player is -1, then don't invert the board
        state[1,:,:]*=sgn
        #if no_parts is True, remove the parts layer (last one)
        if no_parts: state = state[:2,:,:]

        return(state)
    
    def get_converted_board(self):
        '''
        This method converts dimension from [depth, row, col] to [row, col, depth] due to the reason that
        Conv2D layer currently supports only [row, col, depth] for non-gpu version.
        '''
        #get canonical board
        board = self.get_canonical_board()
        #convert from [depth, row, col] to [row, col, depth]
        converted_board = np.zeros(shape=board.T.shape)
        for i in range(board.T.shape[0]):
            converted_board[i] = board.T[:, i, :]

        return converted_board

    def score(self):
        #get position of current player's workers
        worker_idx = np.sign(self.get_canonical_board()[1,:,:]) == -1
        #check if workers at those positions are on top
        if (self.board[0,:,:][worker_idx] == self.winning_floor).any():
            reward = 1
        else:
            reward = 0
        return(reward)
        
    def move(self,worker,key):
        #worker is either -1, -2; pov of current player
        if worker not in [-1,-2]: raise ValueError('Wrong Worker')
        
        #get source and destinations
        state = self.get_canonical_board()
        worker_idx = np.where(state[1,:,:]==worker)
        src = (worker_idx[0][0],worker_idx[1][0])
        worker_num = self.board[1,src[0], src[1]]
        delta = self.ktoc[key]
        dest = (src[0]+delta[0],src[1]+delta[1])
        
        #check if correct turn
        if np.sign(self.board[1,src[0],src[1]])!=self.current_player:
            raise ValueError('Wrong Player')
        
        #check legality of the move; within the board, one level, no one standing
        inbound =  (-1 < dest[0] < self.board_dim[0]) & (-1 < dest[1] < self.board_dim[1])
        blank_tile = self.board[1,dest[0],dest[1]]==0
        one_level = (self.board[0,dest[0],dest[1]] - self.board[0,src[0],src[1]]) <= 1
        not_dome = self.board[0,dest[0],dest[1]] < 4
        
        if inbound & one_level & blank_tile & not_dome:
            self.board[1,src[0],src[1]] = 0
            self.board[1,dest[0],dest[1]] = worker_num
        else:
            #print(f'Illegal Move\n Inbound: {inbound}\n One Level: {one_level}\n Blank Tile: {blank_tile}')
            raise ValueError(f'Illegal Move')
    
    def build(self,worker,key):
        #worker is either -1, -2; pov of current player
        if worker not in [-1,-2]: raise ValueError('Wrong Worker')
        
        #get source and destinations
        state = self.get_canonical_board()
        worker_idx = np.where(state[1,:,:]==worker)
        src = (worker_idx[0][0],worker_idx[1][0])
        worker_num = self.board[1,src[0], src[1]]
        delta = self.ktoc[key]
        dest = (src[0]+delta[0],src[1]+delta[1]) #destination of the new building
        
        #check tower size legality
        to_build = self.board[0,dest[0],dest[1]] + 1
        if to_build <=4:
            parts_left = self.board[2,to_build,to_build]
        else:
            raise ValueError('Building too tall')
            
        #check if correct turn
        if np.sign(self.board[1,src[0],src[1]])!=self.current_player:
            raise ValueError('Wrong Player')
            
        #check legality of the build; within the board, enough parts, no one standing
        inbound =  (-1 < dest[0] < self.board_dim[0]) & (-1 < dest[1] < self.board_dim[1])
        enough_parts = parts_left > 0
        blank_tile = self.board[1,dest[0],dest[1]]==0
        if inbound & enough_parts & blank_tile:
            self.board[0,dest[0],dest[1]] = to_build
            self.board[2,to_build,to_build] -= 1
        elif inbound & (not enough_parts) & blank_tile:
            #if no parts left; do nothing 
            # EDIT -> raise error to return score -1 instead of doing nothing to consider current player as losing the game
            raise ValueError('Not enough parts')
        else:
            raise ValueError('Illegal Build')
                          
    def step(self, action_idx, switch_player=True , move_reward = 0):
        self.turns+=1
        reward = move_reward
        worker,move_key,build_key = self.itoa[action_idx]
        
        #try to move
        try:
            self.move(worker,move_key)
        except:
            self.record_state()
            next_state = self.get_board()
            reward += -1
            done = True
            if switch_player: self.current_player *= -1
            return(next_state,reward,done,self.current_player)
            
        #try to build
        try:
            self.build(worker,build_key)
        except:
            self.record_state()
            next_state = self.get_board()
            reward += -1
            done = True
            if switch_player: self.current_player *= -1 
            return(next_state,reward,done,self.current_player)
        
        #move on
        self.record_state()
        next_state = self.get_board()
        reward += self.score()
        done = True if (self.score()==1) else False
        if switch_player: self.current_player *= -1
        return(next_state,reward,done,self.current_player)
    
    def legal_moves(self):
        legals = []
        for i,j in enumerate(self.itoa):
            legal = True
            old_board = self.board.copy()
        
            #try to move
            try:
                self.move(j[0],j[1])
            except:
                #print(f'illegal move {i}')
                legal = False
            #try to build
            try:
                self.build(j[0],j[2])
            except:
                #print(f'illegal build {i}')
                legal = False
                
            if legal: 
                #print(f'legal {i}')
                legals.append(i)
                legal = True
                
            self.board = old_board
        return(legals)