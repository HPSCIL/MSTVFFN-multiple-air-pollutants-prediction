import torch.nn as nn
from Temporal_block import TemFEM
from Spatial_block import SpaFEM
from Spectral_block import Spectral_Attention
from Embedding import DataEmbedding


class EncoderLayer(nn.Module):
    def __init__(self,var_dim,hid_dim,num_nodes,lag,d_k,d_model,n_heads,device,dropout):
        super(EncoderLayer,self).__init__()

        self.data_embedding = DataEmbedding(var_dim,d_model)
        # TemFEM
        self.temporal_layer = TemFEM(device, embed_size=d_model, num_nodes=num_nodes, heads=n_heads, time_num=lag, dropout=dropout, forward_expansion=4)
        # SpaFEM
        self.spatial_layer = SpaFEM(device, var_dim, hid_dim, num_nodes, lag, dropout)
        # SpeFEM
        self.spectral_layer = Spectral_Attention(n_heads, num_nodes, lag, d_k, dropout)
        self.spectral_conv = nn.Linear(var_dim,d_model)

    def forward(self,x,cheb_polynomials):
        # x:B,N,T,D
        x1 = x
        x2 = x
        x3 = x

        x1 = self.data_embedding(x1)
        x1 = self.temporal_layer(x1)

        x2 = self.spatial_layer(x2, cheb_polynomials)

        x3 = self.spectral_layer(x3)
        x3 = self.spectral_conv(x3)

        return x1,x2,x3



