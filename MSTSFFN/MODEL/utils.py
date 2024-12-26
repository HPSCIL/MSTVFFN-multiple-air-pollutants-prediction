import torch
import os
import logging
import random

def get_logger(root, name=None, debug=True):
    #when debug is true, show DEBUG and INFO in screen
    #when debug is false, show DEBUG in file and info in both screen&file
    #INFO will always be in screen
    # create a logger
    logger = logging.getLogger(name)
    logger.handlers.clear()
    #critical > error > warning > info > debug > notset
    logger.setLevel(logging.DEBUG)

    # define the formate
    formatter = logging.Formatter('%(asctime)s: %(message)s', "%Y-%m-%d %H:%M")
    # create another handler for output log to console
    console_handler = logging.StreamHandler()
    if debug:
        console_handler.setLevel(logging.DEBUG)
    else:
        console_handler.setLevel(logging.INFO)
        # create a handler for write log to file
        logfile = os.path.join(root, 'run.log')
        print('Creat Log File in: ', logfile)
        file_handler = logging.FileHandler(logfile, mode='w')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    # add Handler to logger
    logger.addHandler(console_handler)
    if not debug:
        logger.addHandler(file_handler)
    return logger



def All_Metrics(pred, true):
   # print(pred.shape,true.shape)
    mae = MAE_torch(pred, true)
    rmse = RMSE_torch(pred, true)
    mape = MAPE_torch(pred, true)
    r2 = r2_torch(pred, true)
    return mae, rmse, mape,r2


def MAE_torch(pred, true):
    return torch.mean(torch.abs(true-pred))

def MSE_torch(pred, true):
    return torch.mean((pred - true) ** 2)

def RMSE_torch(pred, true):
    return torch.sqrt(torch.mean((pred - true) ** 2))

def MAPE_torch(pred, true):
    return torch.mean(torch.abs(torch.div((true - pred), true)))

def r2_torch(pred, true):
    target_mean = torch.mean(true)
    ss_tot = torch.sum((true - target_mean) ** 2)
    ss_res = torch.sum((true - pred) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2


import numpy as np
from scipy.sparse.linalg import eigs
import torch

def scaled_Laplacian(W):
    '''
    compute \tilde{L}

    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    scaled_Laplacian: np.ndarray, shape (N, N)

    '''

    assert W.shape[0] == W.shape[1]

    D = np.diag(np.sum(W, axis=1))  #每行的和 分布在对角线上

    L = D - W

    lambda_max = eigs(L, k=1, which='LR')[0].real

    return (2 * L) / lambda_max - np.identity(W.shape[0])

def cheb_polynomial(L_tilde, K):
    '''
    compute a list of chebyshev polynomials from T_0 to T_{K-1}

    Parameters
    ----------
    L_tilde: scaled Laplacian, np.ndarray, shape (N, N)

    K: the maximum order of chebyshev polynomials

    Returns
    ----------
    cheb_polynomials: list(np.ndarray), length: K, from T_0 to T_{K-1}

    '''

    N = L_tilde.shape[0]  #35

    cheb_polynomials = [np.identity(N), L_tilde.copy()]  #单位矩阵和L~拼接 构成切比雪夫的前两项

    for i in range(2, K):
        cheb_polynomials.append(2 * L_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])  #添加切比雪夫的第三项

    return cheb_polynomials


class StandardScaler:
    """
    Standard the input
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        #if len(data.shape) != 4:
            #print(len(data.shape))
            #exit()
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        #if len(data.shape) != 4:
            #print(len(data.shape))
            #exit()
        return (data * self.std) + self.mean




def init_seed(seed):
    '''
    Disable cudnn to maximize reproducibility
    '''
    torch.cuda.cudnn_enabled = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)