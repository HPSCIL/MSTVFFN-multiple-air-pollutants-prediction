from torch.utils.data import Dataset
import numpy as np
import torch



# 创建一个自定义的Dataset类
class MyDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]



def pro_data(data,args):
    trainX,trainY= [],[]
    for i in range(0,len(data)-args.lag-args.pre_len):
        x_train = data[i:i+args.lag:,:,:]   #[48,35,6]
        target_train = data[i+args.lag:i+args.lag+args.pre_len,:,:]  #[24,35,1]
        trainX.append(x_train)
        trainY.append(target_train)
    trainX = np.array(trainX).transpose(0, 2, 3, 1)  #[8614,35,6,24]
    trainY = np.array(trainY).transpose(0,2,3,1)   # [8614,35,6,24]
    return trainX,trainY


def data_loader(X, Y, batch_size, shuffle=True, drop_last=True):
    cuda = True if torch.cuda.is_available() else False
    # cuda = False
    TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    X,Y= TensorFloat(X),TensorFloat(Y)
    data = torch.utils.data.TensorDataset(X,Y)
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                             shuffle=shuffle, drop_last=drop_last)

    return dataloader