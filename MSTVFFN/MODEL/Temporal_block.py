import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from Embedding import DataEmbedding


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V):
        '''
        Q: [batch_size, n_heads, T(Spatial) or N(Temporal), N(Spatial) or T(Temporal), d_k]
        K: [batch_size, n_heads, T(Spatial) or N(Temporal), N(Spatial) or T(Temporal), d_k]
        V: [batch_size, n_heads, T(Spatial) or N(Temporal), N(Spatial) or T(Temporal), d_k]
        '''
        B, n_heads, len1, len2, d_k = Q.shape
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn,V)  # [batch_size, n_heads, T(Spatial) or N(Temporal), N(Spatial) or T(Temporal), d_k]]
        return context


class TMultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(TMultiHeadAttention, self).__init__()

        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        assert (
                self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.W_V = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_K = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_Q = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, input_Q, input_K, input_V):
        '''
        input_Q: [batch_size, N, T, C]
        input_K: [batch_size, N, T, C]
        input_V: [batch_size, N, T, C]
        '''
        B, N, T, C = input_Q.shape
        # [B, N, T, C] --> [B, N, T, h * d_k] --> [B, N, T, h, d_k] --> [B, h, N, T, d_k]
        Q = self.W_Q(input_Q).view(B, N, T, self.heads, self.head_dim).permute(0, 3, 1, 2, 4)  # Q: [B, h, N, T, d_k]
        K = self.W_K(input_K).view(B, N, T, self.heads, self.head_dim).permute(0, 3, 1, 2, 4)  # K: [B, h, N, T, d_k]
        V = self.W_V(input_V).view(B, N, T, self.heads, self.head_dim).permute(0, 3, 1, 2, 4)  # V: [B, h, N, T, d_k]

        context = ScaledDotProductAttention()(Q, K, V)  # [B, h, N, T, d_k]
        context = context.permute(0, 2, 3, 1, 4)  # [B, N, T, h, d_k]
        context = context.reshape(B, N, T, self.heads * self.head_dim)  # [B, N, T, C]
        output = self.fc_out(context)  # [batch_size, len_q, d_model]
        return output


class Temporal_Attention_layer(nn.Module):
    def __init__(self, device, in_channels, num_of_vertices, num_of_timesteps):
        super(Temporal_Attention_layer, self).__init__()
        self.U1 = nn.Parameter(torch.randn(num_of_vertices).to(device))
        self.U2 = nn.Parameter(torch.randn(in_channels, num_of_vertices).to(device))
        self.U3 = nn.Parameter(torch.randn(in_channels).to(device))
        self.be = nn.Parameter(torch.randn(1, num_of_timesteps, num_of_timesteps).to(device))
        self.Ve = nn.Parameter(torch.randn(num_of_timesteps, num_of_timesteps).to(device))

    def forward(self, x):
        '''
        :param x: (batch_size, N, F_in, T)
        :return: (B, T, T)
        '''
        _, num_of_vertices, num_of_features, num_of_timesteps = x.shape
        lhs = torch.matmul(torch.matmul(x.permute(0,3,2,1), self.U1), self.U2)
        # x:(B, N, F_in, T) -> (B, T, F_in, N)
        # (B, T, F_in, N)(N) -> (B,T,F_in)
        # (B,T,F_in)(F_in,N)->(B,T,N)
        rhs = torch.matmul(self.U3, x)  # (F)(B,N,F,T)->(B, N, T)
        product = torch.matmul(lhs, rhs)  # (B,T,N)(B,N,T)->(B,T,T)
        E = torch.matmul(self.Ve, torch.sigmoid(product + self.be))  # (B, T, T)
        E_normalized = F.softmax(E, dim=1)   # normalization
        return E_normalized


class TTransformer(nn.Module):
    def __init__(self, embed_size, heads, time_num, device, dropout, forward_expansion):
        super(TTransformer, self).__init__()

        self.time_num = time_num
        self.temporal_embedding = nn.Embedding(time_num, embed_size)
        self.attention = TMultiHeadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.dropout = nn.Dropout(dropout)
        self.device = device

    def forward(self,query):
        B, N, T, C = query.shape
        D_T = self.temporal_embedding(torch.arange(0, T).to(self.device))
        D_T = D_T.expand(B, N, T, C)
        query = query + D_T
        attention = self.attention(query, query, query)

        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class TemFEM(nn.Module):
    def __init__(self,device,embed_size,num_nodes, heads, time_num, dropout, forward_expansion=4):
        super(TemFEM, self).__init__()
        self.Multi_station_Transformer = TTransformer(embed_size, heads, time_num, device, dropout, forward_expansion)
        self.Bilinear_Temporal_Attention = Temporal_Attention_layer(device,embed_size,num_nodes,time_num)
        self.Local_Temporal_Convolution = nn.Conv2d(embed_size,embed_size,kernel_size=(3,1),padding=(1,0))
        self.norm1 = nn.LayerNorm(embed_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)


    def forward(self, x):
        # x:[B,N,T,D]
        batch_size, num_of_vertices, num_of_timesteps, num_of_features = x.shape
        x = self.Multi_station_Transformer(x) + x
        y = x.permute(0,1,3,2)

        T_attention = self.Bilinear_Temporal_Attention(y)
        x_TAt = torch.matmul(x.permute(0,1,3,2).reshape(batch_size, -1, num_of_timesteps), T_attention).reshape(batch_size, num_of_vertices, -1,num_of_timesteps)
        x_TAt = self.dropout1(self.norm1((x_TAt+y).permute(0,1,3,2))).permute(0,3,2,1)
        output = self.dropout2(F.relu(self.Local_Temporal_Convolution(x_TAt)))
        return output.permute(0,3,2,1)






