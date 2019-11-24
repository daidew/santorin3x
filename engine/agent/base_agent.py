import numpy as np
from engine.network.neural_net import *
from collections import deque
from tensorflow.keras.optimizers import *
import datetime

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
    
    def __init__(self, current_player=-1, eps=0.3, min_eps=0.01, eps_decay=0.99, gamma=0.99, lr=1e-4, tau=1e-3, mode='dense', batch_size=32):
        self.mode = mode
        self.current_player = current_player
        if mode == 'dense':
            self.q_network = DenseNetwork([256])
            # self.q_network.build(input_shape=(1,75))
            self.q_target_network = DenseNetwork([256])
            self.q_target_network.set_weights(self.q_network.get_weights())
        # elif mode == 'conv':
            # self.q_network = ConvolutionNetwork()
            # self.
        self.lr = lr
        self.tau = tau
        self.eps = eps
        self.gamma = gamma
        self.min_eps = min_eps
        self.eps_decay = eps_decay
        self.memory = deque([], maxlen=50000)
        self.batch_size = batch_size
        self.optimizer = Adam(lr=self.lr)
        self.iter = 0
        print('======= configuration =======')
        print('current_player:',self.current_player)
        print('lr:{}\ngamma:{}\ntau:{}\nmode:{}\nbs:{}\nepsilon:{}, min_eps:{}, eps_decay:{}'.format(self.lr, self.gamma, self.tau,self.mode,self.batch_size,self.eps,self.min_eps, self.eps_decay))
        print('==============================')
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_dir = 'logs/' + current_time + '/'
        print('tensorboard log_dir at: ',self.log_dir)
        self.model_path = 'models/'
        print('save model path is: ',self.model_path)
        self.summary_writer = tf.summary.create_file_writer(self.log_dir)

    def step(self, env):
        all_legal_moves = env.legal_moves()
        s = env.get_canonical_board()
        #sample check epsilon boundary
        t = np.random.uniform(low=0,high=1)
        if t > self.eps:
            #find best action from neural network
            if self.mode == 'dense':
                #flatten the board and feed to neural network
                a = self.q_network(np.ravel(env.get_converted_board()).reshape(1, -1))[0]
            else:
                raise NotImplementedError()
            mask = np.zeros(shape=env.action_dim)
            #zero out probability of illegal moves
            mask[env.legal_moves()] = 1
            a *= mask
            #renormalize
            a /= np.sum(a)
            action_idx = np.argmax(a, axis=0)
            if action_idx not in all_legal_moves:
                try:
                    action_idx = np.random.choice(all_legal_moves)
                except:
                    #no legal actions available
                    action_idx = -1
        else:
            try:
                action_idx = np.random.choice(all_legal_moves)
            except:
                #no legal actions available
                action_idx = -1

        #step the env
        ss, r, done, next_player = env.step(action_idx, switch_player=True)

        #add action to the replay memory
        self.memory.append({'s':s,
                            'a':action_idx, 
                            'r':r, 
                            'ss':ss,
                            'done':done, 
                            })
        #decay epsilon
        self.decay_epsilon()

        return ss, action_idx, r, done, next_player

    def learn(self):
        
        if len(self.memory) < self.batch_size:
            # too few data
            return

        chosen_idx = np.random.choice(len(self.memory), self.batch_size, replace=False)

        s, a, r, ss, done = [], [], [], [], []
        for i, idx in enumerate(chosen_idx):
            #unpack
            _s, _a, _r, _ss, _done = [self.memory[idx][kw] for kw in ['s', 'a', 'r', 'ss', 'done']]
            s.append(_s)
            a.append(_a)
            r.append(_r)
            ss.append(_ss)
            done.append(_done)
            # s = tf.convert_to_tensor(s)
            # a = tf.convert_to_tensor(a)
            # r = tf.convert_to_tensor(r)
            # ss= tf.convert_to_tensor(ss)
            # done= tf.convert_to_tensor(done)
        grad = None 
        avg_td_error = 0
        avg_loss = 0
        for i in range(self.batch_size):
            with tf.GradientTape() as tape:
                with self.summary_writer.as_default():
                    if done[i]:
                        y = tf.convert_to_tensor([r[i]], dtype=tf.float32)
                    else:
                        if self.mode == 'dense':
                            # print(np.ravel(s[i]).reshape(1,-1))
                            _a = self.q_target_network(np.ravel(s[i]).reshape(1,-1))
                        else:
                            raise NotImplementedError()
                        y = tf.convert_to_tensor([r[i] + tf.math.multiply(self.gamma, tf.math.reduce_max(_a))], dtype=tf.float32)
                    __a = self.q_network(np.ravel(ss[i]).reshape(1,-1))
                    td_error = tf.math.subtract(y, tf.math.reduce_max(__a, keepdims=True))
                    loss = tf.math.pow(td_error, 2)
                    # print(td_error, loss)
                    avg_td_error += td_error[0, 0]
                    avg_loss += loss[0, 0]
                    tf.summary.scalar(name='td_error', data=td_error[0, 0], step=self.iter)
                    tf.summary.scalar(name='loss', data=loss[0, 0], step=self.iter)
                    # print('td error: {}, loss: {}'.format(td_error, loss))
                    if grad is None:
                        grad = tape.gradient(loss, self.q_network.trainable_weights) 
                    else:
                        grad += tape.gradient(loss, self.q_network.trainable_weights)
                    for g in grad:
                        g /= self.batch_size

                    for i, weight in enumerate(grad):
                        tf.summary.histogram(name='grad_{}'.format(i),data=weight,step=self.iter)
                    for weight in self.q_network.variables:
                        tf.summary.histogram(name=weight.name,data=weight,step=self.iter)
                    for weight in self.q_target_network.variables:
                        tf.summary.histogram(name=weight.name,data=weight,step=self.iter)

                self.iter += 1
            self.optimizer.apply_gradients(zip(grad, self.q_network.trainable_weights))
            #soft update weights
            _q_weights = self.q_network.get_weights()
            _q_target_weights = self.q_target_network.get_weights()
            for qw, qtw in zip(_q_weights, _q_target_weights):
                qtw = self.tau * qtw + (1 - self.tau) * qw
            self.q_target_network.set_weights(_q_target_weights)

        avg_loss /= self.batch_size
        avg_td_error /= self.batch_size
        tf.summary.scalar(name='avg_td_error', data=avg_td_error, step=self.iter)
        tf.summary.scalar(name='avg_loss', data=avg_loss, step=self.iter)

    def decay_epsilon(self):
        self.eps *= self.eps_decay
        if self.eps < self.min_eps:
            self.eps = self.min_eps

