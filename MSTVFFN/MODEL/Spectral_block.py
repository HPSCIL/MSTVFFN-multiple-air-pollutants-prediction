import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from visualizer import get_local

class Spectral_Attention(nn.Module):
    def __init__(self, n_head, num_nods, lag, d_k, dropout):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k

        self.w_qs = nn.Linear(num_nods*lag, n_head * d_k)
        self.w_ks = nn.Linear(num_nods*lag, n_head * d_k)
        self.w_vs = nn.Linear(num_nods*lag, n_head * d_k)

        self.fc = nn.Linear(n_head * d_k, num_nods*lag)

        self.layer_norm = nn.LayerNorm(num_nods*lag)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)


    def forward(self, x):
        d_k,  n_head = self.d_k, self.n_head

        B,N,T,D = x.size()

        # Reshape spatio-temporal vector
        x = x.permute(0,3,1,2).reshape(B,D,-1)   # Represent the variables as vectors of T*D

        residual = x
        q = self.w_qs(x).view(B, D, n_head, d_k)
        k = self.w_ks(x).view(B, D, n_head, d_k)
        v = self.w_vs(x).view(B, D, n_head, d_k)

        q = q.permute(0,2,1,3)
        k = k.permute(0,2,1,3)
        v = v.permute(0,2,1,3)

        attn = torch.matmul(q, k.transpose(-2, -1))
        attn = attn / math.sqrt(d_k)
        scores = self.dropout1(F.softmax(attn, dim = -1))
        output = torch.matmul(scores, v)

        output = output.transpose(1, 2).contiguous().view(B, -1, n_head * d_k)
        output = self.fc(output)
        output = self.dropout2(F.relu(output))
        output = self.layer_norm(output + residual)
        output = output.view(B, D, N, T).permute(0,2,3,1)

        return output