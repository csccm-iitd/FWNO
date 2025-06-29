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
    def __init__(self):
        super().__init__()
        self.layers=nn.Sequential(nn.Linear(8,16),nn.LeakyReLU(),nn.Linear(16,len(wavelets)))
        self.initialize_weights(5, -50)
    def initialize_weights(self,a,b):
        for m in self.layers:
            if isinstance(m, nn.Linear):
                if m == self.layers[0]:
                    m.weight.data[0, -1] = a
                elif m == self.layers[2]:
                    m.weight.data[0, 0] = b
                else:
                    nn.init.zeros_(m.weight)
                    nn.init.zeros_(m.bias)
    def forward(self,architecture):
        return self.layers(architecture).exp()
    def forward(self,architecture):
        return self.layers(architecture).exp()
    
class GFlowNet_activations(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=nn.Sequential(nn.Linear(8,16),nn.LeakyReLU(),nn.Linear(16,len(activations)))
        self.initialize_weights(5, -50)
    def initialize_weights(self,a,b):
        for m in self.layers:
            if isinstance(m, nn.Linear):
                if m == self.layers[0]:
                    m.weight.data[0, -1] = a
                elif m == self.layers[2]:
                    m.weight.data[0, 0] = b
                else:
                    nn.init.zeros_(m.weight)
                    nn.init.zeros_(m.bias)
    def forward(self,architecture):
        return self.layers(architecture).exp()

def main():
    model_wavelets=GFlowNet_wavelets()
    model_activations=GFlowNet_activations()
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
        torch.save(best_model.state_dict(),'FWNO_Burgers.pth')
        print(f'Best architecture searched for Burgers Equation is {best_architecture}')

        """ For plotting the figure"""
        # pred = torch.zeros(y_test.shape)
        # index = 0
        # test_e = torch.zeros(y_test.shape[0])
        # test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=1, shuffle=False)
        # with torch.no_grad():
        #     for x, y in test_loader:
        #         test_l2 = 0
        #         x, y = x.to(device), y.to(device)
            
        #         out = best_model(x).view(-1)
        #         pred[index] = out
            
        #         test_l2 += myloss(out.view(1, -1), y.view(1, -1)).item()
        #         test_e[index] = test_l2
        #         index = index + 1
        #     print('Mean Error:', 100*torch.mean(test_e))
        
        # state = [0,0,0,0,0,0,0,0]
        # model_wno = train_loop(x_train, train_loader,test_loader, state, epochs = 500, ntrain = 1000)
        # torch.save(model_wno.state_dict(),'WNO_Burgers.pth')
        # pred1 = torch.zeros(y_test.shape)
        # index = 0
        # test_e = torch.zeros(y_test.shape[0])
        # test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=1, shuffle=False)
        # with torch.no_grad():
        #     for x, y in test_loader:
        #         test_l2 = 0
        #         x, y = x.to(device), y.to(device)
            
        #         out = model_wno(x).view(-1)
        #         pred1[index] = out
            
        #         test_l2 += myloss(out.view(1, -1), y.view(1, -1)).item()
        #         test_e[index] = test_l2
        #         index = index + 1
        #     print('Mean Error:', 100*torch.mean(test_e))

        # plt.rcParams["font.family"] = "Times New Roman"
        # plt.rcParams['font.size'] = 16
        # figure1 = plt.figure(figsize = (12, 12))
        # plt.subplots_adjust(hspace=0.4)
        # colors = plt.cm.viridis(np.linspace(0, 1, y_test.shape[0]))
        # plt.rcParams["font.family"] = "DejaVu Serif"
        # plt.rcParams['font.size'] = 16
        # for i in range(y_test.shape[0]):
        #     if i % 30 == 1:
        #         plt.subplot(3, 1, 1)
        #         plt.plot(np.linspace(0, 1, 1024), x_test[i, :].numpy(), color=colors[i])  # Different color for each plot
        #         plt.title('(a) I.C.')
        #         plt.xlabel('x', fontsize=20, fontweight='bold')
        #         plt.ylabel('u(x,0)', fontsize=20, fontweight='bold')
        #         plt.grid(True)
        #         plt.xticks(fontweight='bold')
        #         plt.yticks(fontweight='bold')
        #         plt.margins(0)

        #         plt.subplot(3, 1, 2)
        #         plt.plot(np.linspace(0, 1, 1024), y_test[i, :].numpy(), color=colors[i])  # Use the same color for ground truth
        #         plt.plot(np.linspace(0, 1, 1024), pred1[i, :], ':k')  # Use black for prediction line
        #         plt.title('(b) WNO')
        #         plt.legend(['Truth', 'Prediction'], ncol=2, loc='best', fontsize=20)
        #         plt.xlabel('x', fontsize=20, fontweight='bold')
        #         plt.ylabel('u(x,1)', fontsize=20, fontweight='bold')
        #         plt.grid(True)
        #         plt.xticks(fontweight='bold')
        #         plt.yticks(fontweight='bold')
        #         plt.margins(0)
                
        #         plt.subplot(3, 1, 3)
        #         plt.plot(np.linspace(0, 1, 1024), y_test[i, :].numpy(), color=colors[i])  # Use the same color for ground truth
        #         plt.plot(np.linspace(0, 1, 1024), pred[i, :], ':k')  # Use black for prediction line
        #         plt.title('(c) FWNO')
        #         plt.legend(['Truth', 'Prediction'], ncol=2, loc='best', fontsize=20)
        #         plt.xlabel('x', fontsize=20, fontweight='bold')
        #         plt.ylabel('u(x,1)', fontsize=20, fontweight='bold')
        #         plt.grid(True)
        #         plt.xticks(fontweight='bold')
        #         plt.yticks(fontweight='bold')
        #         plt.margins(0)

        # plt.show()
        # figure1.patch.set_facecolor('white')
        # figure1.savefig('Prediction_Burgers.pdf', bbox_inches='tight')
    training()
    
if __name__=='__main__':
    main()