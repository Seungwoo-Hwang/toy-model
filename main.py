"""

    In this toy model
    
    1. Use position descriptor consist of 144 atoms, total 432 descriptors(Not symmetry function descriptor)
    2. Train just energy term (Not include force/stress)
    3. 

"""

from toy_simpleNN import ToySimpleNN
from preprocess import Data


hyperparam = {'activation':'tanh', 'regularization':'l2', 'batch_size':20, 'learning_rate':1e-4, 'valid_rate':0.1, 'optimizer':'SGD','train_step':1000}



data_set = Data()
data_set.load_OUTCAR("C:/Users/Seungwoo Hwang/Desktop/toy-simpleNN/OUTCAR")
#data_set = Data("OUTCAR")

x_train, y_train, x_test, y_test = data_set.train_test_split()

model = ToySimpleNN(act=hyperparam['activation'], train_step=hyperparam['train_step'])
model.set_optimizer(optimizer=hyperparam['optimizer'])
model.load_data(data_set)

model.train()

model.test()
