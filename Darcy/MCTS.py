import math
from wno_2d_darcy import *
import pandas as pd
import operator
from copy import deepcopy
from functools import reduce
from mcts.base.base import BaseState, BaseAction
from mcts.searcher.mcts import MCTS

TRAIN_PATH = 'data/piececonst_r421_N1024_smooth1.mat'
TEST_PATH = 'data/piececonst_r421_N1024_smooth2.mat'
reader = MatReader(TRAIN_PATH)
x_train = reader.read_field('coeff')[:ntrain,::r,::r][:,:s,:s]
y_train = reader.read_field('sol')[:ntrain,::r,::r][:,:s,:s]

reader.load_file(TEST_PATH)
x_test = reader.read_field('coeff')[:ntest,::r,::r][:,:s,:s]
y_test = reader.read_field('sol')[:ntest,::r,::r][:,:s,:s]

x_normalizer = UnitGaussianNormalizer(x_train)
x_train = x_normalizer.encode(x_train)
x_test = x_normalizer.encode(x_test)

y_normalizer = UnitGaussianNormalizer(y_train)
y_train = y_normalizer.encode(y_train)

x_train = x_train.reshape(ntrain,s,s,1)
x_test = x_test.reshape(ntest,s,s,1)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

def reward_function(architecture, epochs=100):
    model=train_loop(x_train,y_normalizer, train_loader,test_loader, architecture, epochs=epochs)
    loss=test_loop(x_test, y_test,y_normalizer, model)
    return math.exp(-loss)

wavelets=['db4','coif6','bior6.8','rbio6.8','sym6']
activations=[F.gelu,F.elu,F.leaky_relu,F.selu,torch.sigmoid]

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
    searcher = MCTS(iteration_limit=500)
    best_action, reward = searcher.search(initial_state=initial_state, need_details=True)
if __name__=='__main__':
    main()