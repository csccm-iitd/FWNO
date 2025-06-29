import math
from wno_1d_Burgers import *
import pandas as pd

dataloader = MatReader('data/burgers_data_R10.mat')
x_data = dataloader.read_field('a')[:,::sub]
y_data = dataloader.read_field('u')[:,::sub]

ntrain = 1000
x_train = x_data[:ntrain,:]
y_train = y_data[:ntrain,:]
x_test = x_data[-ntest:,:]
y_test = y_data[-ntest:,:]

x_train = x_train.reshape(ntrain,s,1)
x_test = x_test.reshape(ntest,s,1)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

def reward_function(architecture, epochs = 100):
    model=train_loop(x_train, train_loader,test_loader, architecture, epochs, ntrain = 1000)
    loss=test_loop(x_test, y_test, model)
    return math.exp(-loss)

wavelets=['db6','coif6','bior6.8','rbio6.8','sym6']
activations=[F.gelu,F.elu,F.leaky_relu,F.selu,torch.sigmoid]

losses=[]
states=[]
for epochs in range(500):
    current_state=torch.tensor((-1,-1,-1,-1,-1,-1,-1,-1),dtype=torch.float32,device=device,requires_grad=True)
    for j in range(8):
        print('j='+str(j))
        print(current_state)
        action=torch.randint(0, len(wavelets), (1,))
        current_state_new=[]
        for k in range(8):
            if k==j:
                current_state_new.append(action)
            else:
                current_state_new.append(current_state[k])
        current_state_new=torch.tensor(current_state_new,dtype=torch.float32,device=device,requires_grad=True)
        if j==7:
            rew=reward_function(current_state_new)
            states.append(current_state_new)
            losses.append(-math.log(rew))
        current_state=current_state_new
    print(current_state)

sorted_pairs = sorted(zip(losses, states))
_, sorted_states = zip(*sorted_pairs) 
topk_states = list(sorted_states[:10])
min_loss = 1000
for state in topk_states:
    model=train_loop(x_train, train_loader,test_loader, state, epochs = 500, ntrain = 1000)
    loss=test_loop(x_test, y_test, model)
    if loss<min_loss:
        min_loss = loss
        best_architecture = state
        best_model = model
torch.save(best_model.state_dict(),'RandomSearch_Burgers.pth')
print(f'Best architecture searched with Random Search for Burgers Equation is {best_architecture}')