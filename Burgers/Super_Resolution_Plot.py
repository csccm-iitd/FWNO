from wno_1d_Burgers_super import *

ntrain = 1000
ntest = 100

sub = 2**3 # subsampling rate
sub_test1 = 2**3
sub_test2 = 2**2
sub_test3 = 2**1
sub_test4 = 2**0

h = 2**13 // sub # total grid size divided by the subsampling rate
s = h
s_test1 = 2**13//sub_test1
s_test2 = 2**13//sub_test2
s_test3 = 2**13//sub_test3
s_test4 = 2**13//sub_test4

batch_size = 10
learning_rate = 0.001

epochs = 500
step_size = 100
gamma = 0.5

level = 8 
width = 64

dataloader = MatReader('data/burgers_data_R10.mat')
x_data = dataloader.read_field('a')[:,::sub]
y_data = dataloader.read_field('u')[:,::sub]
x_data_test1 = dataloader.read_field('a')[:,::sub_test1]
y_data_test1 = dataloader.read_field('u')[:,::sub_test1]
x_data_test2 = dataloader.read_field('a')[:,::sub_test2]
y_data_test2 = dataloader.read_field('u')[:,::sub_test2]
x_data_test3 = dataloader.read_field('a')[:,::sub_test3]
y_data_test3 = dataloader.read_field('u')[:,::sub_test3]
x_data_test4 = dataloader.read_field('a')[:,::sub_test4]
y_data_test4 = dataloader.read_field('u')[:,::sub_test4]

x_train = x_data[:ntrain,:]
y_train = y_data[:ntrain,:]
x_test1 = x_data_test1[-ntest:,:]
y_test1 = y_data_test1[-ntest:,:]
x_test2 = x_data_test2[-ntest:,:]
y_test2 = y_data_test2[-ntest:,:]
x_test3 = x_data_test3[-ntest:,:]
y_test3 = y_data_test3[-ntest:,:]
x_test4 = x_data_test4[-ntest:,:]
y_test4 = y_data_test4[-ntest:,:]

x_train = x_train.reshape(ntrain,s,1)
x_test1 = x_test1.reshape(ntest,s_test1,1)
x_test2 = x_test2.reshape(ntest,s_test2,1)
x_test3 = x_test3.reshape(ntest,s_test3,1)
x_test4 = x_test4.reshape(ntest,s_test4,1)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)

model_dict=torch.load('FWNO_Burgers.pth')
State=[2,0,0,0,3,0,0,1] # Best Architecture searched by FWNO
model=WNO1d(width, level, x_train.permute(0,2,1),State)
model.load_state_dict(model_dict)
model.to(device)
opt=torch.optim.Adam(model.parameters(),1e-3)

pred1 = torch.zeros(y_test1.shape)
index = 0
test_e = torch.zeros(y_test1.shape[0])
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test1, y_test1), batch_size=1, shuffle=False)
with torch.no_grad():
    for x, y in test_loader:
        test_l2 = 0
        x, y = x.to(device), y.to(device)
    
        out = model(x).view(-1)
        pred1[index] = out
    
        test_l2 += myloss(out.view(1, -1), y.view(1, -1)).item()
        test_e[index] = test_l2
        index = index + 1
    print('Mean Error:', 100*torch.mean(test_e))
    
model=WNO1d(width, level, x_train.permute(0,2,1),State)
model.load_state_dict(model_dict)
model.to(device)
opt=torch.optim.Adam(model.parameters(),1e-3)

pred2 = torch.zeros(y_test2.shape)
index = 0
test_e = torch.zeros(y_test2.shape[0])
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test2, y_test2), batch_size=1, shuffle=False)
with torch.no_grad():
    for x, y in test_loader:
        test_l2 = 0
        x, y = x.to(device), y.to(device)
    
        out = model(x).view(-1)
        pred2[index] = out
    
        test_l2 += myloss(out.view(1, -1), y.view(1, -1)).item()
        test_e[index] = test_l2
        index = index + 1
    print('Mean Error:', 100*torch.mean(test_e))

model=WNO1d(width, level, x_train.permute(0,2,1),State)
model.load_state_dict(model_dict)
model.to(device)
opt=torch.optim.Adam(model.parameters(),1e-3)

pred3 = torch.zeros(y_test3.shape)
index = 0
test_e = torch.zeros(y_test3.shape[0])
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test3, y_test3), batch_size=1, shuffle=False)
with torch.no_grad():
    for x, y in test_loader:
        test_l2 = 0
        x, y = x.to(device), y.to(device)
    
        out = model(x).view(-1)
        pred3[index] = out
    
        test_l2 += myloss(out.view(1, -1), y.view(1, -1)).item()
        test_e[index] = test_l2
        index = index + 1
    print('Mean Error:', 100*torch.mean(test_e))
    
model=WNO1d(width, level, x_train.permute(0,2,1),State)
model.load_state_dict(model_dict)
model.to(device)
opt=torch.optim.Adam(model.parameters(),1e-3)

pred4 = torch.zeros(y_test4.shape)
index = 0
test_e = torch.zeros(y_test4.shape[0])
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test4, y_test4), batch_size=1, shuffle=False)
with torch.no_grad():
    for x, y in test_loader:
        test_l2 = 0
        x, y = x.to(device), y.to(device)
    
        out = model(x).view(-1)
        pred4[index] = out
    
        test_l2 += myloss(out.view(1, -1), y.view(1, -1)).item()
        test_e[index] = test_l2
        index = index + 1
    print('Mean Error:', 100*torch.mean(test_e))

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 16
figure1 = plt.figure(figsize = (12, 8))
plt.subplots_adjust(hspace=0.4)
plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams['font.size'] = 16

for i in range(y_test1.shape[0]):
    if i % 30 == 1:
        plt.subplot(2, 1, 1)
        plt.plot(np.linspace(0, 1, s_test4), x_test4[i, :].numpy(), color='k') 
        plt.title('(a) I.C.')
        plt.xlabel('x', fontsize=20, fontweight='bold')
        plt.ylabel('u(x,0)', fontsize=20, fontweight='bold')
        plt.grid(True)
        plt.xticks(fontweight='bold')
        plt.yticks(fontweight='bold')
        plt.margins(0)

        plt.subplot(2, 1, 2)
        plt.plot(np.linspace(0, 1, s_test4), y_test4[i, :].numpy(), color='k') 
        plt.plot(np.linspace(0, 1, s_test2), pred2[i, :], ':g')  
        plt.plot(np.linspace(0, 1, s_test3), pred3[i, :], ':b')  
        plt.plot(np.linspace(0, 1, s_test4), pred4[i, :], ':m')  
        plt.title('(b) Solution')
        plt.legend(['Truth','Prediction res=2048','Prediction res=4096','Prediction res=8192'], ncol=2, bbox_to_anchor=(0.45, 0.3), fontsize=10)
        plt.xlabel('x', fontsize=20, fontweight='bold')
        plt.ylabel('u(x,1)', fontsize=20, fontweight='bold')
        plt.grid(True)
        plt.xticks(fontweight='bold')
        plt.yticks(fontweight='bold')
        plt.margins(0)

plt.show()
figure1.patch.set_facecolor('white')
figure1.savefig('Prediction_Burgers_Super.pdf', bbox_inches='tight')