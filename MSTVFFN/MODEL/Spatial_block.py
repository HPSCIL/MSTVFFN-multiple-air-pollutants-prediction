import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
from visualizer import get_local

class Chevby_GCN(nn.Module):
    '''
    K-order chebyshev graph convolution
    '''

    def __init__(self,device,K,num_nodes,in_channels, out_channels):
        '''
        :param K: int
        :param in_channles: int, num of channels in the input sequence
        :param out_channels: int, num of channels in the output sequence
        '''
        super(Chevby_GCN, self).__init__()
        self.K = K
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_nodes = num_nodes
        self.device =device
        self.Theta = nn.ParameterList([nn.Parameter(torch.randn(in_channels, out_channels).to(self.device)) for _ in range(K)])


    def forward(self, x,STDG,cheb_polynomials):

        batch_size, num_nodes, in_channels, num_of_timesteps = x.shape
        outputs = []
        for time_step in range(num_of_timesteps):
            graph_signal = x[:, :, :, time_step]  # (b, N, F_in,1)
            output = torch.zeros(batch_size, num_nodes, self.out_channels).to(self.device)  # (b, N, F_out)
            # STDG-chebyshev graph convolution
            for k in range(self.K):
                T_k = cheb_polynomials[k]
                # add STDG
                T_k_with_at = T_k.mul(STDG)
                theta_k = self.Theta[k]
                rhs = T_k_with_at.permute(0,2,1).matmul(graph_signal)
                output = output + rhs.matmul(theta_k)
            # output final hiden state ht
            outputs.append(output.unsqueeze(-1))  # (b, N, F_out, 1)
        return F.relu(torch.cat(outputs, dim=-1))  # (b, N, F_out, T)



class Spatial_Attention_layer(nn.Module):
    '''
    compute spatial attention scores
    '''
    def __init__(self, device, in_channels, num_of_vertices, num_of_timesteps):
        super(Spatial_Attention_layer, self).__init__()
        self.W1 = nn.Parameter(torch.randn(num_of_timesteps).to(device))
        self.W2 = nn.Parameter(torch.randn(in_channels, num_of_timesteps).to(device))
        self.W3 = nn.Parameter(torch.randn(in_channels).to(device))
        self.bs = nn.Parameter(torch.randn(1, num_of_vertices, num_of_vertices).to(device))
        self.Vs = nn.Parameter(torch.randn(num_of_vertices, num_of_vertices).to(device))


    def forward(self, x):
        '''
        :param x: (batch_size, N, F_in, T)
        :return: (B,N,N)
        '''
        lhs = torch.matmul(torch.matmul(x, self.W1), self.W2)  # (b,N,F,T)(T)->(b,N,F)(F,T)->(b,N,T)
        rhs = torch.matmul(self.W3, x).transpose(-1, -2)  # (F)(b,N,F,T)->(b,N,T)->(b,T,N)
        product = torch.matmul(lhs, rhs)  # (b,N,T)(b,T,N) -> (B, N, N)
        S = torch.matmul(self.Vs, torch.sigmoid(product + self.bs))  # (N,N)(B, N, N)->(B,N,N)
        S_normalized = F.softmax(S, dim=1)

        return S_normalized





class SpaFEM(nn.Module):
    def __init__(self,device, in_channels, out_channels, num_nodes, num_of_timesteps,dropout,K = 3):
        super(SpaFEM,self).__init__()

        self.Bilinear_Spatial_Attention = Spatial_Attention_layer(device, in_channels, num_nodes, num_of_timesteps)
        self.GCN =Chevby_GCN(device,K,num_nodes,in_channels, out_channels)
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1,1), stride=(1,1))
        self.ln = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.Conv = nn.Conv2d(out_channels,out_channels,kernel_size=(3,1),padding=(1,0))


    def forward(self,x,cheb_polynomials):
        # x:[B,N,F,T]
        x =x.permute(0,1,3,2)
        # spatial dependence graph
        SDG = self.Bilinear_Spatial_Attention(x)  # [B,N,N]
        # Chebyshev graph convolution
        x_s = self.GCN(x,SDG,cheb_polynomials)
        # residual_layer
        x_residual = self.residual_conv(x.permute(0,2,1,3))
        x_residual = self.dropout(self.ln(F.relu(x_residual + x_s.permute(0,2,1,3)).permute(0,2,3,1)).permute(0,3,1,2))
        output = self.dropout(F.relu(self.Conv(x_residual)))
        return output.permute(0,2,3,1)
