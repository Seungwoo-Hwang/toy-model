import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ToySimpleNN(nn.Module):
    """ Neural Network class that perform both training and test

    First, initialize with hyperparameter input
    Second, load data set that form of "Data()" class
    Third, initialize optimizer
    And then, training Neural Network with loaded data set
    Also test Neural Network performance

    Attributes:
        activations: Activation function dictionary that can be used
        act:         Choosen activation function that will be used
        lr:          Learning rate
        train_step:  Total train steps
        linear1:     First hidden layer
        linear2:     Second hidden layer
        model:       Full Neural Network model
        data_set:    Data set that will be used to train and test
    
    Methods:
        set_optimizer(optimizer)
        load_data(data_set)
        train()
        test()       
    """
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
        """ Initialize optimizer with input hyperparameters

        Set "self.optimizer" 

        Args:
            optimizer:(str) optimizer type name to used
        """
        if optimizer == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        else:
            print('Not implemented')

    def load_data(self, data_set):
        """ Import data set that form of "Data" class

        Set "self.data_set" to Data class type

        Args:
            data_set:(class Data) Data class that contain x_train, y_train, x_test, y_test
        """
        self.data_set = data_set

    def train(self):
        """ Train Neural Network with loaded data set

        Train "self.model" with self.data_set
        Updata self.model.parameters()
        """
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
        """ Test trained Neural Network with loaded data set

        Show RMSE
        """
        with torch.no_grad():
            hypothesis = self.model(self.data_set.x_test)
            mse = F.mse_loss(hypothesis, self.data_set.y_test)
            rmse = torch.sqrt(mse)
            print('\nRMSE: ', rmse.item()/self.data_set.atom_num)
