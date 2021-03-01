import torch

class Data:
    """ Data class that used in ToySimpleNN model

    First, initialize with hyperparameter input
    Second, load data set that form of "Data()" class
    Third, initialize optimizer
    And then, training Neural Network with loaded data set
    Also test Neural Network performance

    Attributes:
        atom_num:(int)               Total atom number in each snapshot
        f_name:(str)                 File name to open
        x_train:(torch.FloatTensor)  Torch tensor that used as x_train
        y_train:(torch.FloatTensor)  Torch tensor that used as y_train
        x_test:(torch.FloatTensor)   Torch tensor that used as x_test
        y_test:(torch.FloatTensor)   Torch tensor that used as y_test
    
    Methods:
        load_OUTCAR(FILE)
        train_test_split()    
    """
    def __init__(self):
        self.atom_num = 144
        self.f_name = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

    def load_OUTCAR(self, FILE):
        """ Set OUTCAR format file name to be used

        Args:
            FILE:(str) File names to be used (OUTCAR format)
        """
        self.f_name = FILE

    def train_test_split(self):
        """ Set OUTCAR format file name to be used

        First, open OUTCAR file that name self.f_name
        Read position and conver to 432 dimension vector type and used as x_train/x_test
        Read energy and used as y_train/y_test
        Split position data and energy data to train/test
        > After split, set self.x_train, self.x_test, self.y_train, self.y_test

        Returns:
            Splited train/test x data and y data to be used in other Class or Method (ex> ToySimpleNN class)
        """
        if self.f_name == None:
            print('Must load OUTCAR first')
            raise Exception('Use "Data.load_OUTCAR(FILE)" method')
        f=open(self.f_name, 'r')

        pos_data=[]
        e_data=[]

        line=f.readline()
        while(line != ''):
            if 'POSITION' in line:
                pos_v = []
                f.readline()
                for i in range(self.atom_num):
                    pos =f.readline().split()
                    pos_v.append(float(pos[0]))
                    pos_v.append(float(pos[1]))
                    pos_v.append(float(pos[2]))
                pos_data.append(pos_v)
            if 'free  energy' in line:
                e_data.append(float(line.split()[4]))
            line=f.readline()
        f.close()

        # split train/test set
        self.x_train = torch.FloatTensor(pos_data[::10][:270])
        self.y_train = torch.FloatTensor(e_data[::10][:270])
        self.x_test = torch.FloatTensor(pos_data[::10][270:])
        self.y_test = torch.FloatTensor(e_data[::10][270:])

        return self.x_train, self.y_train, self.x_test, self.y_test