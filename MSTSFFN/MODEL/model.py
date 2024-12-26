import torch.nn as nn
from Layers import EncoderLayer
from CrossAttention import CrossAttention



class Encoder(nn.Module):
    def __init__(self,var_dim,hid_dim,num_nodes,lag,d_k,d_model,n_heads,device,dropout):
        super(Encoder,self).__init__()
        self.layer = EncoderLayer(var_dim,hid_dim,num_nodes,lag,d_k,d_model,n_heads,device,dropout)

    def forward(self,x,cheb_polynomials):
        x = x.permute(0,1,3,2)
        temporal_att,spatial_att,variety_att = self.layer(x,cheb_polynomials)
        return temporal_att,spatial_att,variety_att


class Decoder(nn.Module):
    def __init__(self,n_head, num_nods,d_model, d_k, dropout):
        super(Decoder,self).__init__()
        self.Crossatt_layer1 = CrossAttention(n_head, d_model, d_k, dropout)
        self.prediction = nn.Conv2d(num_nods,num_nods,kernel_size=(13,d_model-3))


    def forward(self, x1,x2,x3):
        attention12 = self.Crossatt_layer1(x2,x1,x1)
        attention23 = self.Crossatt_layer1(x3,attention12,attention12)
        output = self.prediction(attention23)

        return output.permute(0,1,3,2)


class MSTSFF(nn.Module):
    def __init__(self,args):
        super(MSTSFF, self).__init__()
        self.var_dim = args.var_dim
        self.hid_dim = args.hid_dim
        self.num_nodes =args.num_nodes
        self.lag =args.lag
        self.d_k =args.d_k
        self.d_model =args.d_model
        self.d_ff =args.d_ff
        self.n_heads =args.n_heads
        self.device = args.device
        self.pre_len = args.pre_len
        self.dropout = args.dropout

        self.encoder = Encoder(self.var_dim,self.hid_dim,self.num_nodes,self.lag,self.d_k,self.d_model,self.n_heads,self.device,self.dropout)
        self.decoder = Decoder(self.n_heads, self.num_nodes,self.d_model, self.d_k,self.dropout)

    def forward(self,x,cheb_polynomials):
      # x: B,N,D,T
      temporal_att,spatial_att,variety_att = self.encoder(x,cheb_polynomials)
      output = self.decoder(temporal_att,spatial_att,variety_att)

      return output

