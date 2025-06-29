import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt

from timeit import default_timer
from utilities3 import *
from pytorch_wavelets import DWT, IDWT # (or import DWT, IDWT)

torch.manual_seed(0)
np.random.seed(0)

wavelets=['db4','coif6','bior6.8','rbio6.8','sym6']
activations=[F.gelu,F.elu,F.leaky_relu,F.selu,torch.sigmoid]
myloss = LpLoss(size_average=False)

# %%
""" Def: 2d Wavelet layer """
class WaveConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, level, dummy,mother_wavelet):
        super(WaveConv2d, self).__init__()

        """
        2D Wavelet layer. It does DWT, linear transform, and Inverse dWT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.level = level
        self.dwt_ = DWT(J=self.level, mode='symmetric', wave=mother_wavelet).to(dummy.device)
        self.mode_data, _ = self.dwt_(dummy)
        self.modes1 = self.mode_data.shape[-2]
        self.modes2 = self.mode_data.shape[-1]

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))
        self.mother_wavelet=mother_wavelet

    # Convolution
    def mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute single tree Discrete Wavelet coefficients using some wavelet
        dwt = DWT(J=self.level, mode='symmetric', wave=self.mother_wavelet).to(device)
        x_ft, x_coeff = dwt(x)

        # Multiply relevant Wavelet modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x_ft.shape[-2], x_ft.shape[-1], device=x.device)
        out_ft = self.mul2d(x_ft, self.weights1)
        # Multiply the finer wavelet coefficients
        x_coeff[-1][:,:,0,:,:] = self.mul2d(x_coeff[-1][:,:,0,:,:].clone(), self.weights2)
        x_coeff[-1][:,:,1,:,:] = self.mul2d(x_coeff[-1][:,:,1,:,:].clone(), self.weights3)
        x_coeff[-1][:,:,2,:,:] = self.mul2d(x_coeff[-1][:,:,2,:,:].clone(), self.weights4)
        
        # Return to physical space        
        idwt = IDWT(mode='symmetric', wave=self.mother_wavelet).to(device)
        x = idwt((out_ft, x_coeff))
        return x

""" The forward operation """
class WNO2d(nn.Module):
    def __init__(self, width, level, dummy_data,architecture):
        super(WNO2d, self).__init__()

        """
        The WNO network. It contains 4 layers of the Wavelet integral layer.
        1. Lift the input using v(x) = self.fc0 .
        2. 4 layers of the integral operators v(+1) = g(K(.) + W)(v).
            W is defined by self.w_; K is defined by self.conv_.
        3. Project the output of last layer using self.fc1 and self.fc2.
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.level = level
        self.dummy_data = dummy_data
        self.width = width
        self.padding = 1 # pad the domain when required
        self.fc0 = nn.Linear(3, self.width) # input channel is 3: (a(x, y), x, y)
        self.architecture=architecture
        
        self.conv=nn.ModuleList([WaveConv2d(self.width, self.width, self.level, self.dummy_data,wavelets[int(architecture[2*index])]) for index in range(len(architecture)//2)])
        self.w=nn.ModuleList([nn.Conv2d(self.width, self.width, 1) for index in range(len(architecture)//2)])

        self.fc1 = nn.Linear(self.width, 192)
        self.fc2 = nn.Linear(192, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)

        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, [0,self.padding, 0,self.padding])

        for index in range(len(self.architecture)//2):
            x=x.to(device)
            x1=self.conv[index](x)
            x2=self.w[index](x)
            x=x1+x2
            if index==len(self.architecture)//2-1 :
                x = x[..., :-self.padding, :-self.padding]
                x=x.permute(0,2,3,1)
                x=self.fc1(x)
                x=activations[int(self.architecture[2*index+1])](x)
                x=self.fc2(x)
            else:
                x=activations[int(self.architecture[2*index+1])](x)
        return x
    
    def get_grid(self, shape, device):
        # The grid of the solution
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

# %%
""" Model configurations """

TRAIN_PATH = 'data/piececonst_r421_N1024_smooth1.mat'
TEST_PATH = 'data/piececonst_r421_N1024_smooth2.mat'
ntrain = 1000
ntest = 100

batch_size = 20
learning_rate = 0.001

step_size = 50
gamma = 0.75

level = 4
width = 64

r = 5
h = int(((421 - 1)/r) + 1)
s = h

# %%
""" The model definition """
def train_loop(x_train,y_normalizer,train_loader,test_loader,architecture, epochs):
    model = WNO2d(width, level, x_train.permute(0,3,1,2),architecture).to(device)
    print(count_params(model))

    """ Training and testing """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    train_loss = torch.zeros(epochs)
    test_loss = torch.zeros(epochs)
    y_normalizer.cuda()
    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_l2 = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x).reshape(batch_size, s, s)
            out = y_normalizer.decode(out)
            y = y_normalizer.decode(y)

            loss = myloss(out.view(batch_size,-1), y.view(batch_size,-1))
            loss.backward()
            optimizer.step()
            train_l2 += loss.item()

        scheduler.step()
        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)

                out = model(x).reshape(batch_size, s, s)
                out = y_normalizer.decode(out)

                test_l2 += myloss(out.view(batch_size,-1), y.view(batch_size,-1)).item()

        train_l2/= ntrain
        test_l2 /= ntest

        train_loss[ep] = train_l2
        test_loss[ep] = test_l2

        t2 = default_timer()
        print(ep, t2-t1, train_l2, test_l2)
    return model
    # %%
    """ Prediction """
def test_loop(x_test,y_test,y_normalizer,model):
    pred = torch.zeros(y_test.shape)
    index = 0
    test_e = torch.zeros(y_test.shape[0])
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=1, shuffle=False)
    with torch.no_grad():
        for x, y in test_loader:
            test_l2 = 0
            x, y = x.to(device), y.to(device)

            out = model(x).reshape(s, s)
            out = y_normalizer.decode(out)
            pred[index] = out

            test_l2 += myloss(out.reshape(1, s, s), y.reshape(1, s, s)).item()
            test_e[index] = test_l2
            print(index, test_l2)
            index = index + 1

    print('Mean Testing Error:', 100*torch.mean(test_e).numpy(), '%')
    return 100*torch.mean(test_e).numpy()