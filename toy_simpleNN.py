import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# make neural network (432-30-30-1)
class ToySimpleNN(nn.Module):
    def __init__(self, act='sigmoid', lr=1e-4, train_step=10000):
        super().__init__()
        self.activations = nn.ModuleDict([
                ['relu', nn.ReLU()],
                ['sigmoid', nn.Sigmoid()],
                ['tanh', nn.Tanh()]
        ])
        self.act = act
        self.lr = lr
        self.train_step = train_step
        self.linear1 = nn.Linear(432,30, bias=True)
        self.linear2 = nn.Linear(30,1, bias=True)
        self.model = torch.nn.Sequential(self.linear1, self.activations[self.act], self.linear2)
        self.data_set = None

    def set_optimizer(self, optimizer):
        if optimizer == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        else:
            print('Not implemented')
    def load_data(self, data_set):
        self.data_set = data_set
    def train(self):
        if self.data_set == None:
            print('Must load "Data class" data set first')
            raise Exception('Use "ToySimpleNN.load_data(data_set)" method')
        if self.data_set.x_train == None or self.data_set.y_train == None:
            print('Must split data set first')
            raise Exception('Use "Data.train_test_split()" method')

        for epoch in range(self.train_step):
            prediction = self.model(self.data_set.x_train)
            
            cost = F.mse_loss(prediction, self.data_set.y_train)
            
            l2_reg = 0
            for param in self.model.parameters():
                l2_reg += torch.norm(param)
            
            cost += l2_reg
            
            self.optimizer.zero_grad()
            cost.backward()
            self.optimizer.step()
            
            if epoch % 100 == 0:
                #params = list(self.model.parameters())
                print('Epoch {:4d}/{} Cost: {:.6f}'.format(epoch, 1000, cost.item()))

    def test(self):
        with torch.no_grad():
            hypothesis = self.model(self.data_set.x_test)
            mse = F.mse_loss(hypothesis, self.data_set.y_test)
            rmse = torch.sqrt(mse)
            print('\nRMSE: ', rmse.item()/self.data_set.atom_num)
