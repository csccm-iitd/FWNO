import pywt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from timeit import default_timer
from utilities3 import *
from pytorch_wavelets import DWT1D, IDWT1D

torch.manual_seed(0)
np.random.seed(0)

wavelets=['db6','coif6','bior6.8','rbio6.8','sym6']
activations=[F.gelu,F.elu,F.leaky_relu,F.selu,torch.sigmoid]
myloss = LpLoss(size_average=False)


# %%
""" Def: 1d Wavelet layer """
class WaveConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, level, dummy,mother_wavelet):
        super(WaveConv1d, self).__init__()

        """
        1D Wavelet layer. It does Wavelet Transform, linear transform, and
        Inverse Wavelet Transform.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.level = level  
        self.dwt_ = DWT1D(wave=mother_wavelet, J=self.level, mode='symmetric').to(dummy.device)
        self.mode_data, _ = self.dwt_(dummy)
        self.modes1 = self.mode_data.shape[-1]
        
        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1))
        self.mother_wavelet=mother_wavelet

    # Convolution
    def mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute single tree Discrete Wavelet coefficients using some wavelet     
        dwt = DWT1D(wave=self.mother_wavelet, J=self.level, mode='symmetric').to(device)
        x_ft, x_coeff = dwt(x)
        
        # Multiply the final low pass and high pass coefficients
        out_ft = torch.zeros(batchsize, self.out_channels, x_ft.shape[-1],  device=x.device)
        out_ft = self.mul1d(x_ft, self.weights1)
        x_coeff[-1] = self.mul1d(x_coeff[-1], self.weights2)
        
        # Reconstruct the signal
        idwt = IDWT1D(wave=self.mother_wavelet, mode='symmetric').to(device)
        x = idwt((out_ft, x_coeff))        
        return x

""" The forward operation """
class WNO1d(nn.Module):
    def __init__(self, width, level, dummy_data, architecture):
        super(WNO1d, self).__init__()

        """
        The WNO network. It contains 4 layers of the Wavelet integral layer.
        1. Lift the input using v(x) = self.fc0 .
        2. 4 layers of the integral operators v(+1) = g(K(.) + W)(v).
            W is defined by self.w_; K is defined by self.conv_.
        3. Project the output of last layer using self.fc1 and self.fc2.
        
        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.level = level
        self.width = width
        self.dummy_data = dummy_data
        self.padding = 2 # pad the domain when required
        self.fc0 = nn.Linear(2, self.width) # input channel is 2: (a(x), x)
        self.architecture=architecture
        
        
        self.conv=nn.ModuleList([WaveConv1d(self.width, self.width, self.level, self.dummy_data,wavelets[int(architecture[2*index])]) for index in range(len(architecture)//2)])
        self.w=nn.ModuleList([nn.Conv1d(self.width, self.width, 1) for index in range(len(architecture)//2)])
    

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        
        for index in range(len(self.architecture)//2):
            x=x.to(device)
            x1=self.conv[index](x)
            x2=self.w[index](x)
            x=x1+x2
            if index==len(self.architecture)//2-1 :
                x=x.permute(0,2,1)
                x=self.fc1(x)
                x=activations[int(self.architecture[2*index+1])](x)
                #x=F.gelu(x)
                x=self.fc2(x)
            else:
                x=activations[int(self.architecture[2*index+1])](x)
        return x

    def get_grid(self, shape, device):
        # The grid of the solution
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)


# %%
""" Model configurations """

ntest = 100

sub = 2**3 # subsampling rate
h = 2**13 // sub # total grid size divided by the subsampling rate
s = h

batch_size = 10
learning_rate = 0.001

step_size = 100
gamma = 0.5

level = 8 
width = 64


# %%
""" The model definition """
def train_loop(x_train, train_loader,test_loader, architecture, epochs, ntrain):
    model = WNO1d(width, level, x_train.permute(0,2,1),architecture).to(device)
    print(count_params(model))
    
    """ Training and testing """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    train_loss = torch.zeros(epochs)
    test_loss = torch.zeros(epochs)
    myloss = LpLoss(size_average=False)
    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_mse = 0
        train_l2 = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
    
            optimizer.zero_grad()
            out = model(x)
            
            mse = F.mse_loss(out.view(batch_size, -1), y.view(batch_size, -1), reduction='mean')
            l2 = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
            l2.backward() # l2 relative loss
    
            optimizer.step()
            train_mse += mse.item()
            train_l2 += l2.item()
    
        scheduler.step()
        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
    
                out = model(x)
                test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()
    
        train_mse /= len(train_loader)
        train_l2 /= ntrain
        test_l2 /= ntest
        
        train_loss[ep] = train_l2
        test_loss[ep] = test_l2
    
        t2 = default_timer()
        print(ep, t2-t1, train_mse, train_l2, test_l2)
    return model

# %%
""" Prediction """
def test_loop(x_test, y_test, model):
    pred = torch.zeros(y_test.shape)
    index = 0
    test_e = torch.zeros(y_test.shape[0])
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=1, shuffle=False)
    with torch.no_grad():
        for x, y in test_loader:
            test_l2 = 0
            x, y = x.to(device), y.to(device)
    
            out = model(x).view(-1)
            pred[index] = out
    
            test_l2 += myloss(out.view(1, -1), y.view(1, -1)).item()
            test_e[index] = test_l2
            print(index, test_l2)
            index = index + 1
    
    print('Mean Error:', 100*torch.mean(test_e))
    return 100*torch.mean(test_e)
