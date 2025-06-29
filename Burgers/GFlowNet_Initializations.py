import math
from wno_1d_Burgers import *

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

class GFlowNet_wavelets(nn.Module):
    def __init__(self, init = 'ku'):
        super().__init__()
        self.layers=nn.Sequential(nn.Linear(8,16),nn.LeakyReLU(),nn.Linear(16,len(wavelets)))
        self._initialize_weights(init)
    def _initialize_weights(self, init):
        if init=='kn':
            for m in self.layers:
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, a=0.01, mode='fan_in', nonlinearity='leaky_relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        elif init=='xu':
            for m in self.layers:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        elif init=='xn':
            for m in self.layers:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight) 
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self,architecture):
        return self.layers(architecture).exp()
    
class GFlowNet_activations(nn.Module):
    def __init__(self, init = 'ku'):
        super().__init__()
        self.layers=nn.Sequential(nn.Linear(8,16),nn.LeakyReLU(),nn.Linear(16,len(activations)))
        self._initialize_weights(init)

    def _initialize_weights(self, init):
        if init=='kn':
            for m in self.layers:
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, a=0.01, mode='fan_in', nonlinearity='leaky_relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        elif init=='xu':
            for m in self.layers:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        elif init=='xn':
            for m in self.layers:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight) 
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self,architecture):
        return self.layers(architecture).exp()

def main():
    init = 'ku'
    model_wavelets=GFlowNet_wavelets(init)
    model_activations=GFlowNet_activations(init)
    model_wavelets.to(device)
    model_activations.to(device)
    opt_wavelets=torch.optim.Adam(model_wavelets.parameters(),1e-3)
    opt_activations=torch.optim.Adam(model_activations.parameters(),1e-3)
    losses=[]
    states=[]
    mb_loss_l=[]
    def training():
        for i in range(500):
            print(i)
            current_state=torch.tensor((-1,-1,-1,-1,-1,-1,-1,-1),dtype=torch.float32,device=device,requires_grad=True)
            if i%2==0:
                flow_prediction=model_wavelets(current_state)
            else:
                flow_prediction=model_activations(current_state)
            mb_loss=0
            for j in range(8):
                print('j='+str(j))
                print(current_state)
                probabilities=flow_prediction/flow_prediction.sum()
                action=torch.distributions.Categorical(probs=probabilities,validate_args=False).sample()
                current_state_new=[]
                for k in range(8):
                    if k==j:
                        current_state_new.append(action)
                    else:
                        current_state_new.append(current_state[k])
                parent=current_state
                current_state_new=torch.tensor(current_state_new,dtype=torch.float32,device=device,requires_grad=True)
                if j%2==0:
                    flow_from_parent=model_wavelets(parent)[action]
                else:
                    flow_from_parent=model_activations(parent)[action]
                    
                if j==7:
                    rew=reward_function(current_state_new, epochs = 100)
                    flow_prediction=torch.zeros(1)
                    losses.append(-math.log(rew))
                    states.append(current_state_new)
                else:
                    rew=0
                    if j%2==0:
                        flow_prediction=model_activations(current_state_new)
                    else:
                        flow_prediction=model_wavelets(current_state_new)
                mb_loss+=(flow_from_parent-flow_prediction.sum()-rew).pow(2)
                current_state=current_state_new
            print(current_state)
            if i%1==0:
                print(mb_loss)
                mb_loss_l.append(mb_loss.item())
                mb_loss.backward()
                opt_wavelets.step()
                opt_activations.step()
                opt_wavelets.zero_grad()
                opt_activations.zero_grad()
                mb_loss=0

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
        if init=='ku':
            torch.save(best_model.state_dict(),'FWNO_Burgers_ku.pth')
            print(f'Best architecture searched for Burgers Equation with Kaiming Uniform initialization is {best_architecture}')
        elif init=='kn':
            torch.save(best_model.state_dict(),'FWNO_Burgers_kn.pth')
            print(f'Best architecture searched for Burgers Equation with Kaiming Normal initialization is {best_architecture}')
        elif init=='xu':
            torch.save(best_model.state_dict(),'FWNO_Burgers_xu.pth')
            print(f'Best architecture searched for Burgers Equation with Xavier Uniform initialization is {best_architecture}')
        elif init=='xn':
            torch.save(best_model.state_dict(),'FWNO_Burgers_xn.pth')
            print(f'Best architecture searched for Burgers Equation with Xavier Normal initialization is {best_architecture}')
    training()
    
if __name__=='__main__':
    main()