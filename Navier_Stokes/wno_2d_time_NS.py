import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utilities3 import *
from timeit import default_timer
from pytorch_wavelets import DWT, IDWT # (or import DWT, IDWT)

torch.manual_seed(0)
np.random.seed(0)
myloss = LpLoss(size_average=False)
wavelets=['db4','coif6','bior6.8','rbio6.8','sym6']
activations=[F.gelu,F.elu,F.mish,F.selu,torch.sigmoid]

# %%
""" Def: 2d Wavelet layer """
class WaveConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, level, dummy, mother_wavelet):
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
        self.mother_wavelet=mother_wavelet

    # Convolution
    def mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        
        #Compute single tree Discrete Wavelet coefficients using some wavelet
        dwt = DWT(J=self.level, mode='symmetric', wave=self.mother_wavelet).to(device)
        x_ft, x_coeff = dwt(x)

        # Multiply relevant Wavelet modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x_ft.shape[-2], x_ft.shape[-1], device=x.device)
        out_ft = self.mul2d(x_ft, self.weights1)
        # Multiply the finer wavelet coefficients
        x_coeff[-1][:,:,0,:,:] = self.mul2d(x_coeff[-1][:,:,0,:,:].clone(), self.weights2)
        # x_coeff[-1][:,:,1,:,:] = self.mul2d(x_coeff[-1][:,:,1,:,:].clone(), self.weights3)
        # x_coeff[-1][:,:,2,:,:] = self.mul2d(x_coeff[-1][:,:,2,:,:].clone(), self.weights4)
        
        # Return to physical space        
        idwt = IDWT(mode='symmetric', wave=self.mother_wavelet).to(device)
        x = idwt((out_ft, x_coeff))
        return x

""" The forward operation """
class WNO2d(nn.Module):
    def __init__(self, width, level, dummy_data, architecture):
        super(WNO2d, self).__init__()

        """
        The WNO network. It contains 4 layers of the Wavelet integral layer.
        1. Lift the input using v(x) = self.fc0 .
        2. 4 layers of the integral operators v(+1) = g(K(.) + W)(v).
            W is defined by self.w_; K is defined by self.conv_.
        3. Project the output of last layer using self.fc1 and self.fc2.
        
        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x=S, y=S, c=12)
        output: the solution of the next timestep
        output shape: (batchsize, x=S, y=S, c=1)
        """

        self.level = level
        self.dummy_data = dummy_data
        self.width = width
        self.padding = 1 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(12, self.width) 
        self.architecture=architecture
        
        self.conv=nn.ModuleList([WaveConv2d(self.width, self.width, self.level, self.dummy_data,wavelets[int(architecture[2*index])]) for index in range(len(architecture)//2)])
        self.w=nn.ModuleList([nn.Conv2d(self.width, self.width, 1) for index in range(len(architecture)//2)])
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        for index in range(len(self.architecture)//2):
            x=x.to(device)
            x1=self.conv[index](x)
            x2=self.w[index](x)
            x=x1+x2
            if index==len(self.architecture)//2-1 :
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

ntrain = 1000
ntest = 100

level = 3
width = 26

batch_size = 20
batch_size2 = batch_size

learning_rate = 0.001
scheduler_step = 50
scheduler_gamma = 0.75

sub = 1
S = 64//sub
T_in = 10
T = 10
step = 1


# %%
""" The model definition """
def train_loop(train_a,train_loader,test_loader,architecture, epochs):
    model = WNO2d(width, level, train_a.permute(0,3,1,2), architecture).to(device)
    print(count_params(model))

    """ Training and testing """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

    train_loss = torch.zeros(epochs)
    test_loss = torch.zeros(epochs)
    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_l2_step = 0
        train_l2_full = 0
        for xx, yy in train_loader:
            loss = 0
            xx = xx.to(device)
            yy = yy.to(device)

            for t in range(0, T, step):
                y = yy[..., t:t + step] # t:t+step, retains the third dimension,

                im = model(xx)            
                loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

                if t == 0:
                    pred = im
                else:
                    pred = torch.cat((pred, im), -1)

                xx = torch.cat((xx[..., step:], im), dim=-1)

            train_l2_step += loss.item()
            l2_full = myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1))
            train_l2_full += l2_full.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        test_l2_step = 0
        test_l2_full = 0
        with torch.no_grad():
            for xx, yy in test_loader:
                loss = 0
                xx = xx.to(device)
                yy = yy.to(device)

                for t in range(0, T, step):
                    y = yy[..., t:t + step]
                    im = model(xx)
                    loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

                    if t == 0:
                        pred = im
                    else:
                        pred = torch.cat((pred, im), -1)

                    xx = torch.cat((xx[..., step:], im), dim=-1)

                test_l2_step += loss.item()
                test_l2_full += myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1)).item()

        train_loss[ep] = train_l2_step/ntrain/(T/step)
        test_loss[ep] = test_l2_step/ntest/(T/step)

        t2 = default_timer()
        scheduler.step()
        print(ep, t2 - t1, train_l2_step / ntrain / (T / step), train_l2_full / ntrain, test_l2_step / ntest / (T / step),
              test_l2_full / ntest)
    return model

# %%
""" Prediction """
def test_loop(test_a,test_u,model):
    pred0 = torch.zeros(test_u.shape)
    index = 0
    test_e = torch.zeros(test_u.shape)        
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=1, shuffle=False)

    with torch.no_grad():
        for xx, yy in test_loader:
            test_l2_step = 0
            test_l2_full = 0
            loss = 0
            xx = xx.to(device)
            yy = yy.to(device)

            for t in range(0, T, step):
                y = yy[..., t:t + step]
                im = model(xx)
                loss += myloss(im.reshape(1, y.size()[-3], y.size()[-2]), y.reshape(1, y.size()[-3], y.size()[-2]))

                if t == 0:
                    pred = im
                else:
                    pred = torch.cat((pred, im), -1)

                xx = torch.cat((xx[..., step:], im), dim=-1)

            pred0[index] = pred
            test_l2_step += loss.item()
            test_l2_full += myloss(pred.reshape(1, -1), yy.reshape(1, -1)).item()
            test_e[index] = test_l2_step

            print(index, test_l2_step/ ntest/ (T/step), test_l2_full/ ntest)
            index = index + 1

    print('Mean Testing Error:', 100*torch.mean(test_e).numpy() / (T/step), '%')
    return 100*torch.mean(test_e).numpy() / (T/step)