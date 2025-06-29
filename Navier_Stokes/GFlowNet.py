import math
from wno_2d_time_NS import *
import pandas as pd
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import MaxNLocator

np.random.seed(0)
torch.manual_seed(0)

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

class GFlowNet_wavelets(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=nn.Sequential(nn.Linear(8,16),nn.LeakyReLU(),nn.Linear(16,len(wavelets)))
    
    def forward(self,architecture):
        return self.layers(architecture).exp()
    
class GFlowNet_activations(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=nn.Sequential(nn.Linear(8,16),nn.LeakyReLU(),nn.Linear(16,len(activations)))

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
    losses=[]
    states=[]
    def training():
        for i in range(100):
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
                    rew=reward_function(current_state_new, epochs=100)
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
            model=train_loop(train_a, train_loader,test_loader, state, 1000)
            loss=test_loop(test_a, test_u, model)
            if loss<min_loss:
                min_loss = loss
                best_architecture = state
                best_model = model
        torch.save(best_model.state_dict(),'FWNO_NS.pth')
        print(f'Best architecture searched for Navier-Stokes Equation is {best_architecture}')

        """ For plotting the figure"""
        # pred_fwno = torch.zeros(test_u.shape)
        # index = 0
        # test_e = torch.zeros(test_u.shape)        
        # test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=1, shuffle=False)

        # with torch.no_grad():
        #         for xx, yy in test_loader:
        #             test_l2_step = 0
        #             test_l2_full = 0
        #             loss = 0
        #             xx = xx.to(device)
        #             yy = yy.to(device)

        #             for t in range(0, T, step):
        #                 y = yy[..., t:t + step]
        #                 im = best_model(xx)
        #                 loss += myloss(im.reshape(1, y.size()[-3], y.size()[-2]), y.reshape(1, y.size()[-3], y.size()[-2]))

        #                 if t == 0:
        #                     pred = im
        #                 else:
        #                     pred = torch.cat((pred, im), -1)

        #                 xx = torch.cat((xx[..., step:], im), dim=-1)

        #             pred_fwno[index] = pred
        #             test_l2_step += loss.item()
        #             test_l2_full += myloss(pred.reshape(1, -1), yy.reshape(1, -1)).item()
        #             test_e[index] = test_l2_step

        #             print(index, test_l2_step/ ntest/ (T/step), test_l2_full/ ntest)
        #             index = index + 1
        #         print('Mean Testing Error:', 100*torch.mean(test_e).numpy()/ (T/step), '%')
        # state = [0, 0, 0, 0, 0, 0, 0, 0]
        # model_wno = train_loop(train_a, train_loader, test_loader, state, 1000)
        # pred_wno = torch.zeros(test_u.shape)
        # index = 0
        # test_e = torch.zeros(test_u.shape)        
        # test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=1, shuffle=False)

        # with torch.no_grad():
        #         for xx, yy in test_loader:
        #             test_l2_step = 0
        #             test_l2_full = 0
        #             loss = 0
        #             xx = xx.to(device)
        #             yy = yy.to(device)

        #             for t in range(0, T, step):
        #                 y = yy[..., t:t + step]
        #                 im = model_wno(xx)
        #                 loss += myloss(im.reshape(1, y.size()[-3], y.size()[-2]), y.reshape(1, y.size()[-3], y.size()[-2]))

        #                 if t == 0:
        #                     pred = im
        #                 else:
        #                     pred = torch.cat((pred, im), -1)

        #                 xx = torch.cat((xx[..., step:], im), dim=-1)

        #             pred_wno[index] = pred
        #             test_l2_step += loss.item()
        #             test_l2_full += myloss(pred.reshape(1, -1), yy.reshape(1, -1)).item()
        #             test_e[index] = test_l2_step

        #             print(index, test_l2_step/ (T/step), test_l2_full/ ntest)
        #             index = index + 1
        #         print('Mean Testing Error:', 100*torch.mean(test_e).numpy()/ (T/step), '%')
        # plt.rcParams["font.family"] = "dejavu serif"
        # plt.rcParams['font.size'] = 10


        # figure1 = plt.figure(figsize = (17, 13))

        # plt.subplots_adjust(wspace=0.6, hspace=0.01)

        # vmin_list = [0,0,0,0]
        # vmax_list = [3e-1,2e-1,3e-1,3e-1]
        # index = 0
        # ax = plt.subplot(5,5,1)
        # img = plt.imshow(test_a.numpy()[15,:,:,0], cmap='turbo', extent=[0,1,0,1], interpolation='Gaussian')
        # ax.set_ylabel('Initial Condition', color='m', fontweight='bold', fontsize=18)
        # plt.xticks([])
        # plt.yticks([])
        # for value in range(test_u.shape[-1]):
        #     if value % 3 == 0:
        #         ax = plt.subplot(5,5, index+2)
        #         img = plt.imshow(test_u[15,:,:,value], cmap='turbo', extent=[0,1,0,1], interpolation='Gaussian')
        #         ax.set_title('t={}s'.format(value+10), color='b', fontsize=18, fontweight='bold')
        #         cbar = plt.colorbar(img, ax=ax, fraction=0.045)
        #         cbar.ax.tick_params(labelsize=18) 
        #         cbar.locator = MaxNLocator(nbins=3) 
        #         if value==0:
        #             ax.set_ylabel('Truth',color= 'r', fontweight='bold', fontsize=18)
        #         plt.xticks([])
        #         plt.yticks([])
                
        #         ax = plt.subplot(5,5, index+2+5)
        #         img = plt.imshow(pred_wno[15,:,:,value], cmap='turbo', extent=[0,1,0,1], interpolation='Gaussian')
        #         cbar = plt.colorbar(img, ax=ax, fraction=0.045)
        #         cbar.ax.tick_params(labelsize=18) 
        #         cbar.locator = MaxNLocator(nbins=3)
        #         if value==0:
        #             ax.set_ylabel('WNO\n Prediction', color='green', fontweight='bold', fontsize=18)
        #         plt.xticks([])
        #         plt.yticks([])
                
        #         ax = plt.subplot(5,5, index+2+10)
        #         img = plt.imshow(pred_fwno[15,:,:,value], cmap='turbo', extent=[0,1,0,1], interpolation='Gaussian')
        #         cbar = plt.colorbar(img, ax=ax, fraction=0.045)
        #         cbar.ax.tick_params(labelsize=18) 
        #         cbar.locator = MaxNLocator(nbins=3) 
        #         if value==0:
        #             ax.set_ylabel('FWNO\n Prediction', color='green', fontweight='bold', fontsize=18)
        #         plt.xticks([])
        #         plt.yticks([])
                
        #         ax = plt.subplot(5, 5, index + 2 + 15)
        #         img1 = plt.imshow(np.abs(test_u[15, :, :, value] - pred_wno[15, :, :, value]), cmap='viridis', extent=[0, 1, 0, 1], interpolation='Gaussian', vmin=vmin_list[index], vmax=vmax_list[index])
        #         if value==0:
        #             ax.set_ylabel('WNO\n Error', color='blue', fontweight='bold', fontsize=18)
        #         cbar = plt.colorbar(img1, ax=ax, fraction=0.045, format='%.0e')
        #         cbar.ax.tick_params(labelsize=18) 
        #         cbar.locator = MaxNLocator(nbins=3)  
        #         cbar.update_ticks()
        #         plt.xticks([])
        #         plt.yticks([])

        #         ax = plt.subplot(5, 5, index + 2 + 20)
        #         img2 = plt.imshow(np.abs(test_u[15, :, :, value] - pred_fwno[15, :, :, value]), cmap='viridis', extent=[0, 1, 0, 1], interpolation='Gaussian', vmin=vmin_list[index], vmax=vmax_list[index])
        #         if value==0:
        #             ax.set_ylabel('FWNO\n Error', color='blue', fontweight='bold', fontsize=18)
        #         cbar = plt.colorbar(img2, ax=ax, fraction=0.045, format='%.0e')
        #         cbar.ax.tick_params(labelsize=18)  
        #         cbar.locator = MaxNLocator(nbins=3)  
        #         cbar.update_ticks()
        #         plt.xticks([])
        #         plt.yticks([])
                
        #         plt.margins(0)
        #         index = index + 1
        # plt.show()
        # figure1.patch.set_facecolor('white')
        # figure1.savefig('Prediction_NS.pdf', bbox_inches='tight')
    training()
    
if __name__=='__main__':
    main()