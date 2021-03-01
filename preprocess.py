import torch

class Data:
    def __init__(self):
        self.atom_num = 144
        self.f_name = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

    def load_OUTCAR(self, FILE):
        self.f_name = FILE

    def train_test_split(self):
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