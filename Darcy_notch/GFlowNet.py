import math
from wno_2d_Darcy_notch import *
import pandas as pd
from matplotlib.patches import Rectangle

PATH = 'data/Darcy_Triangular_FNO.mat'
reader = MatReader(PATH)
x_train = reader.read_field('boundCoeff')[:ntrain,::r,::r][:,:s,:s]
y_train = reader.read_field('sol')[:ntrain,::r,::r][:,:s,:s]

x_test = reader.read_field('boundCoeff')[-ntest:,::r,::r][:,:s,:s]
y_test = reader.read_field('sol')[-ntest:,::r,::r][:,:s,:s]

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


wavelets=['db6','coif6','bior6.8','rbio6.8','sym6']
activations=[F.gelu,F.elu,F.leaky_relu,F.selu,torch.sigmoid]

class GFlowNet_wavelets(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=nn.Sequential(nn.Linear(8,16),nn.LeakyReLU(),nn.Linear(16,len(wavelets)))
        self.initialize_weights(4, -40)
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
    
class GFlowNet_activations(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=nn.Sequential(nn.Linear(8,16),nn.LeakyReLU(),nn.Linear(16,len(activations)))
        self.initialize_weights(4, -40)
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
    mb_loss_l=[]
    def training():
        states=[]
        losses=[]
        for i in range(500):
            print(i)
            current_state=torch.tensor((-1,-1,-1,-1,-1,-1,-1,-1),dtype=torch.float32,device=device,requires_grad=True)
            flow_prediction=model_wavelets(current_state)
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
                    rew=reward_function(current_state_new)
                    losses.append(-math.log(rew))
                    states.append(current_state_new)
                    flow_prediction=torch.zeros(1)
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
            model=train_loop(x_train, train_loader,test_loader, state, epochs = 500)
            loss=test_loop(x_test, y_test, model)
            if loss<min_loss:
                min_loss = loss
                best_architecture = state
                best_model = model
        torch.save(best_model.state_dict(),'FWNO_Darcy_Notch.pth')
        print(f'Best architecture searched for Darcy Equation with triangular domain and a notch is {best_architecture}')

        """ For plotting the figure"""
        # pred_fwno = torch.zeros(y_test.shape)
        # index = 0
        # test_e = torch.zeros(y_test.shape)
        # test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=1, shuffle=False)
        # with torch.no_grad():
        #     y_normalizer.cuda()
        #     for x, y in test_loader:
        #         test_l2 = 0
        #         x, y = x.to(device), y.to(device)

        #         out = best_model(x).reshape(s, s)
        #         out = y_normalizer.decode(out)
        #         pred_fwno[index] = out

        #         test_l2 += myloss(out.reshape(1, s, s), y.reshape(1, s, s)).item()
        #         test_e[index] = test_l2
        #         print(index, test_l2)
        #         index = index + 1

        # state = [0,0,0,0,0,0,0,0]
        # model_wno = train_loop(x_train, y_normalizer, train_loader, test_loader, state, epochs=500)
        # torch.save(model_wno.state_dict(), 'WNO_Darcy_Notch.pth')
        # pred_wno = torch.zeros(y_test.shape)
        # index = 0
        # test_e = torch.zeros(y_test.shape)
        # test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=1, shuffle=False)
        # with torch.no_grad():
        #     y_normalizer.cuda()
        #     for x, y in test_loader:
        #         test_l2 = 0
        #         x, y = x.to(device), y.to(device)

        #         out = model_wno(x).reshape(s, s)
        #         out = y_normalizer.decode(out)
        #         pred_wno[index] = out

        #         test_l2 += myloss(out.reshape(1, s, s), y.reshape(1, s, s)).item()
        #         test_e[index] = test_l2
        #         print(index, test_l2)
        #         index = index + 1
        
        # s = 1
        # xmax = s
        # ymax = s-8/51

        # plt.rcParams['font.family'] = 'Times New Roman' 
        # plt.rcParams['font.size'] = 16
        # plt.rcParams['mathtext.fontset'] = 'dejavuserif'

        # figure1, ax = plt.subplots(nrows=4, ncols=4, figsize=(16, 10), dpi=100)
        # plt.subplots_adjust(hspace=0.30, wspace=0.30)

        # value = 1

        # im = ax[0,0].imshow(x_test[value,:,:,0], origin='lower', extent=[0,1,0,1], interpolation='Gaussian', cmap='inferno')
        # xf = np.array([0., xmax/2]); yf = xf*(ymax/(xmax/2)); 
        # ax[0,0].fill_between(xf, yf, ymax, color = [1, 1, 1])
        # xf = np.array([xmax/2, xmax]); yf = (xf-xmax)*(ymax/((xmax/2)-xmax)); 
        # ax[0,0].fill_between(xf, yf, ymax, color = [1, 1, 1])
        # xf = np.array([0, xmax]); 
        # ax[0,0].fill_between(xf, ymax, s, color = [1, 1, 1])       
        # ax[0,0].add_patch(Rectangle((0.5,0),0.02,0.4, facecolor='white'))
        # plt.colorbar(im, ax=ax[0,0], fraction=0.2)
        # ax[0,0].set_ylabel('BC', color='r', fontsize=18)

        # ax[1,0].set_axis_off()

        # im = ax[0,1].imshow(y_test[value,...], origin='lower', extent=[0,1,0,1], interpolation='Gaussian', cmap='inferno')
        # xf = np.array([0., xmax/2]); yf = xf*(ymax/(xmax/2)); 
        # ax[0,1].fill_between(xf, yf, ymax, color = [1, 1, 1])
        # xf = np.array([xmax/2, xmax]); yf = (xf-xmax)*(ymax/((xmax/2)-xmax)); 
        # ax[0,1].fill_between(xf, yf, ymax, color = [1, 1, 1])
        # xf = np.array([0, xmax]); 
        # ax[0,1].fill_between(xf, ymax, s, color = [1, 1, 1])        
        # ax[0,1].add_patch(Rectangle((0.5,0),0.02,0.4, facecolor='white'))
        # plt.colorbar(im, ax=ax[0,1], fraction=0.2)
        # ax[0,1].set_ylabel('Truth', color='green', fontsize=18)

        # ax[1,1].set_axis_off()

        # im = ax[0,2].imshow(pred_wno[value, ...], origin='lower', extent=[0,1,0,1], interpolation='Gaussian', cmap='inferno')
        # xf = np.array([0., xmax/2]); yf = xf*(ymax/(xmax/2)); 
        # ax[0,2].fill_between(xf, yf, ymax, color = [1, 1, 1])
        # xf = np.array([xmax/2, xmax]); yf = (xf-xmax)*(ymax/((xmax/2)-xmax)); 
        # ax[0,2].fill_between(xf, yf, ymax, color = [1, 1, 1])
        # xf = np.array([0, xmax]); 
        # ax[0,2].fill_between(xf, ymax, s, color = [1, 1, 1])        
        # ax[0,2].add_patch(Rectangle((0.5,0),0.02,0.4, facecolor='white'))
        # plt.colorbar(im, ax=ax[0,2], fraction=0.2)
        # ax[0,2].set_ylabel('Prediction', color='m', fontsize=18)
        # ax[0,2].set_title('WNO', color='m', fontsize=18)

        # im = ax[1,2].imshow(np.abs(y_test[value, ...] - pred_wno[value, ...]), origin='lower',
        #                 extent=[0,1,0,1], interpolation='Gaussian', cmap='inferno', vmin=0, vmax=0.02)
        # xf = np.array([0., xmax/2]); yf = xf*(ymax/(xmax/2)); 
        # ax[1,2].fill_between(xf, yf, ymax, color = [1, 1, 1])
        # xf = np.array([xmax/2, xmax]); yf = (xf-xmax)*(ymax/((xmax/2)-xmax)); 
        # ax[1,2].fill_between(xf, yf, ymax, color = [1, 1, 1])
        # xf = np.array([0, xmax]); 
        # ax[1,2].fill_between(xf, ymax, s, color = [1, 1, 1])        
        # ax[1,2].add_patch(Rectangle((0.5,0),0.02,0.4, facecolor='white'))
        # plt.colorbar(im, ax=ax[1,2], fraction=0.2)
        # ax[1,2].set_ylabel('Error', color='m', fontsize=18)

        # im = ax[0,3].imshow(pred_fwno[value, ...], origin='lower', extent=[0,1,0,1], interpolation='Gaussian', cmap='inferno')
        # xf = np.array([0., xmax/2]); yf = xf*(ymax/(xmax/2)); 
        # ax[0,3].fill_between(xf, yf, ymax, color = [1, 1, 1])
        # xf = np.array([xmax/2, xmax]); yf = (xf-xmax)*(ymax/((xmax/2)-xmax)); 
        # ax[0,3].fill_between(xf, yf, ymax, color = [1, 1, 1])
        # xf = np.array([0, xmax]); 
        # ax[0,3].fill_between(xf, ymax, s, color = [1, 1, 1])        
        # ax[0,3].add_patch(Rectangle((0.5,0),0.02,0.4, facecolor='white'))
        # plt.colorbar(im, ax=ax[0,3], fraction=0.2)
        # ax[0,3].set_ylabel('Prediction', color='blue', fontsize=18)
        # ax[0,3].set_title('FWNO', color='blue', fontsize=18)

        # im = ax[1,3].imshow(np.abs(y_test[value, ...] - pred_fwno[value, ...]), origin='lower',
        #                 extent=[0,1,0,1], interpolation='Gaussian', cmap='inferno', vmin=0, vmax=0.02)
        # xf = np.array([0., xmax/2]); yf = xf*(ymax/(xmax/2)); 
        # ax[1,3].fill_between(xf, yf, ymax, color = [1, 1, 1])
        # xf = np.array([xmax/2, xmax]); yf = (xf-xmax)*(ymax/((xmax/2)-xmax)); 
        # ax[1,3].fill_between(xf, yf, ymax, color = [1, 1, 1])
        # xf = np.array([0, xmax]); 
        # ax[1,3].fill_between(xf, ymax, s, color = [1, 1, 1])        
        # ax[1,3].add_patch(Rectangle((0.5,0),0.02,0.4, facecolor='white'))
        # plt.colorbar(im, ax=ax[1,3], fraction=0.2)
        # ax[1,3].set_ylabel('Error', color='blue', fontsize=18)

        # value = 30

        # im = ax[2,0].imshow(x_test[value,:,:,0], origin='lower', extent=[0,1,0,1], interpolation='Gaussian', cmap='inferno')
        # xf = np.array([0., xmax/2]); yf = xf*(ymax/(xmax/2)); 
        # ax[2,0].fill_between(xf, yf, ymax, color = [1, 1, 1])
        # xf = np.array([xmax/2, xmax]); yf = (xf-xmax)*(ymax/((xmax/2)-xmax)); 
        # ax[2,0].fill_between(xf, yf, ymax, color = [1, 1, 1])
        # xf = np.array([0, xmax]); 
        # ax[2,0].fill_between(xf, ymax, s, color = [1, 1, 1])       
        # ax[2,0].add_patch(Rectangle((0.5,0),0.02,0.4, facecolor='white'))
        # plt.colorbar(im, ax=ax[2,0], fraction=0.2)
        # ax[2,0].set_ylabel('BC', color='r', fontsize=18)

        # ax[3,0].set_axis_off()

        # im = ax[2,1].imshow(y_test[value,...], origin='lower', extent=[0,1,0,1], interpolation='Gaussian', cmap='inferno')
        # xf = np.array([0., xmax/2]); yf = xf*(ymax/(xmax/2)); 
        # ax[2,1].fill_between(xf, yf, ymax, color = [1, 1, 1])
        # xf = np.array([xmax/2, xmax]); yf = (xf-xmax)*(ymax/((xmax/2)-xmax)); 
        # ax[2,1].fill_between(xf, yf, ymax, color = [1, 1, 1])
        # xf = np.array([0, xmax]); 
        # ax[2,1].fill_between(xf, ymax, s, color = [1, 1, 1])        
        # ax[2,1].add_patch(Rectangle((0.5,0),0.02,0.4, facecolor='white'))
        # plt.colorbar(im, ax=ax[2,1], fraction=0.2)
        # ax[2,1].set_ylabel('Truth', color='green', fontsize=18)

        # ax[3,1].set_axis_off()

        # im = ax[2,2].imshow(pred_wno[value, ...], origin='lower', extent=[0,1,0,1], interpolation='Gaussian', cmap='inferno')
        # xf = np.array([0., xmax/2]); yf = xf*(ymax/(xmax/2)); 
        # ax[2,2].fill_between(xf, yf, ymax, color = [1, 1, 1])
        # xf = np.array([xmax/2, xmax]); yf = (xf-xmax)*(ymax/((xmax/2)-xmax)); 
        # ax[2,2].fill_between(xf, yf, ymax, color = [1, 1, 1])
        # xf = np.array([0, xmax]); 
        # ax[2,2].fill_between(xf, ymax, s, color = [1, 1, 1])        
        # ax[2,2].add_patch(Rectangle((0.5,0),0.02,0.4, facecolor='white'))
        # plt.colorbar(im, ax=ax[2,2], fraction=0.2)
        # ax[2,2].set_ylabel('Prediction', color='m', fontsize=18)

        # im = ax[3,2].imshow(np.abs(y_test[value, ...] - pred_wno[value, ...]), origin='lower',
        #                 extent=[0,1,0,1], interpolation='Gaussian', cmap='inferno', vmin=0, vmax=0.01)
        # xf = np.array([0., xmax/2]); yf = xf*(ymax/(xmax/2)); 
        # ax[3,2].fill_between(xf, yf, ymax, color = [1, 1, 1])
        # xf = np.array([xmax/2, xmax]); yf = (xf-xmax)*(ymax/((xmax/2)-xmax)); 
        # ax[3,2].fill_between(xf, yf, ymax, color = [1, 1, 1])
        # xf = np.array([0, xmax]); 
        # ax[3,2].fill_between(xf, ymax, s, color = [1, 1, 1])        
        # ax[3,2].add_patch(Rectangle((0.5,0),0.02,0.4, facecolor='white'))
        # plt.colorbar(im, ax=ax[3,2], fraction=0.2)
        # ax[3,2].set_ylabel('Error', color='m', fontsize=18)

        # im = ax[2,3].imshow(pred_fwno[value, ...], origin='lower', extent=[0,1,0,1], interpolation='Gaussian', cmap='inferno')
        # xf = np.array([0., xmax/2]); yf = xf*(ymax/(xmax/2)); 
        # ax[2,3].fill_between(xf, yf, ymax, color = [1, 1, 1])
        # xf = np.array([xmax/2, xmax]); yf = (xf-xmax)*(ymax/((xmax/2)-xmax)); 
        # ax[2,3].fill_between(xf, yf, ymax, color = [1, 1, 1])
        # xf = np.array([0, xmax]); 
        # ax[2,3].fill_between(xf, ymax, s, color = [1, 1, 1])        
        # ax[2,3].add_patch(Rectangle((0.5,0),0.02,0.4, facecolor='white'))
        # plt.colorbar(im, ax=ax[2,3], fraction=0.2)
        # ax[2,3].set_ylabel('Prediction', color='blue', fontsize=18)

        # im = ax[3,3].imshow(np.abs(y_test[value, ...] - pred_fwno[value, ...]), origin='lower',
        #                 extent=[0,1,0,1], interpolation='Gaussian', cmap='inferno', vmin=0, vmax=0.01)
        # xf = np.array([0., xmax/2]); yf = xf*(ymax/(xmax/2)); 
        # ax[3,3].fill_between(xf, yf, ymax, color = [1, 1, 1])
        # xf = np.array([xmax/2, xmax]); yf = (xf-xmax)*(ymax/((xmax/2)-xmax)); 
        # ax[3,3].fill_between(xf, yf, ymax, color = [1, 1, 1])
        # xf = np.array([0, xmax]); 
        # ax[3,3].fill_between(xf, ymax, s, color = [1, 1, 1])        
        # ax[3,3].add_patch(Rectangle((0.5,0),0.02,0.4, facecolor='white'))
        # plt.colorbar(im, ax=ax[3,3], fraction=0.2)
        # ax[3,3].set_ylabel('Error', color='blue', fontsize=18)

        # plt.show()

        # figure1.savefig('Prediction_Darcy_Notch.pdf', format='pdf', dpi=300, bbox_inches='tight')  
    training()
    
if __name__=='__main__':
    main()