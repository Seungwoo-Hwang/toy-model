"""

    In this toy model
    
    1. Use position descriptor consist of 144 atoms, total 432 descriptors(Not symmetry function descriptor)
    2. Train just energy term (Not include force/stress)
    3. 

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

hyperparam = {'activation':'tanh', 'regularization':'l2', 'batch_size':20, 'learning_rate':1e-4, 'valid_rate':0.1}

# Extract atomic position and free energy from OUTCAR
atom_num=144

f=open('OUTCAR','r')

pos_data=[]
e_data=[]

line=f.readline()
while(line != ''):
    if 'POSITION' in line:
        pos_v = []
        f.readline()
        for i in range(atom_num):
            pos =f.readline().split()
            pos_v.append(float(pos[0]))
            pos_v.append(float(pos[1]))
            pos_v.append(float(pos[2]))
        pos_data.append(pos_v)
    if 'free  energy' in line:
        e_data.append(float(line.split()[4]))
    line=f.readline()


# split train/test set
x_train = torch.FloatTensor(pos_data[::10][:270])
y_train = torch.FloatTensor(e_data[::10][:270])
x_test = torch.FloatTensor(pos_data[::10][270:])
y_test = torch.FloatTensor(e_data[::10][270:])

# make neural network (432-30-30-1)
class toySimpleNN(nn.Module):
    def __init__(self, act):
        super().__init__()
        self.activations = nn.ModuleDict([
                ['relu', nn.ReLU()],
                ['sigmoid', nn.Sigmoid()],
                ['tanh', nn.Tanh()]
        ])
        self.act = act
        self.linear1 = nn.Linear(432,30, bias=True)
        self.linear2 = nn.Linear(30,1, bias=True)
        #model = torch.nn.Sequential(linear1, self.activations[self.act], linear2)
    def forward(self, x):
        return torch.nn.Sequential(self.linear1, self.act, self.linear2)
model = toySimpleNN(act=hyperparam['activation'])
linear1 = nn.Linear(432,30, bias=True)
linear2 = nn.Linear(30,1, bias=True)
sigmoid = nn.Sigmoid()
model = torch.nn.Sequential(linear1, sigmoid, linear2)

# set optimizer
optimizer = optim.SGD(model.parameters(), lr=1e-4)

# training
for epoch in range(10001):
    prediction = model(x_train)
    
    cost = F.mse_loss(prediction, y_train)
    
    l2_reg = 0
    for param in model.parameters():
        l2_reg += torch.norm(param)
    
    cost += l2_reg
    
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        params = list(model.parameters())
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(epoch, 1000, cost.item()))

# validation
with torch.no_grad():
    hypothesis = model(x_test)
    mse = F.mse_loss(hypothesis, y_test)
    rmse = torch.sqrt(mse)
    print('\nRMSE: ', rmse.item()/atom_num)
