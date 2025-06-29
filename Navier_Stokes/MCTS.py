import math
from wno_2d_time_NS import *
import pandas as pd
import operator
from copy import deepcopy
from functools import reduce
from mcts.base.base import BaseState, BaseAction
from mcts.searcher.mcts import MCTS

TRAIN_PATH = 'data/ns_V1e-3_N5000_T50.mat'
reader = MatReader(TRAIN_PATH)
data = reader.read_field('u')
train_a = data[:ntrain,::sub,::sub,:T_in]
train_u = data[:ntrain,::sub,::sub,T_in:T+T_in]

test_a = data[-ntest:,::sub,::sub,:T_in]
test_u = data[-ntest:,::sub,::sub,T_in:T+T_in]

print(train_u.shape)
print(test_u.shape)
assert (S == train_u.shape[-2])
assert (T == train_u.shape[-1])

train_a = train_a.reshape(ntrain,S,S,T_in)
test_a = test_a.reshape(ntest,S,S,T_in)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False)

def reward_function(architecture, epochs=100):
    model=train_loop(train_a, train_loader,test_loader, architecture, epochs)
    loss=test_loop(test_a, test_u, model)
    return (math.exp(-loss))

wavelets=['db4','coif6','bior6.8','rbio6.8','sym6']
activations=[F.gelu,F.elu,F.mish,F.selu,torch.sigmoid]

iter = 1
class Architecture_State(BaseState):
    def __init__(self):
        self.board = [-1,-1,-1,-1,-1,-1,-1,-1]
        self.currentPlayer = 1

    def get_current_player(self):
        return self.currentPlayer

    def get_possible_actions(self):
        for i in range(len(self.board)):
            if self.board[i]==-1:
                return [Action(self.currentPlayer, i, j) for j in range(len(wavelets))]
        return []

    def take_action(self, action):
        newState = deepcopy(self)
        newState.board[action.x] = action.y
        newState.currentPlayer = self.currentPlayer
        return newState

    def is_terminal(self):
        return self.board[-1]!=-1

    def get_reward(self):
        if self.board[-1]!=-1:
            rew = reward_function(self.board, epochs=100)
            global iter
            print(f'iteration = {iter}')
            iter += 1
            print(self.board)
            return rew
        return 0


class Action(BaseAction):
    def __init__(self, player, x, y):
        self.player = player
        self.x = x
        self.y = y

    def __str__(self):
        return str((self.x, self.y))

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.x == other.x and self.y == other.y and self.player == other.player

    def __hash__(self):
        return hash((self.x, self.y, self.player))
    
def main():
    initial_state = Architecture_State()
    searcher = MCTS(iteration_limit=100)
    best_action, reward = searcher.search(initial_state=initial_state, need_details=True)  
if __name__=='__main__':
    main()