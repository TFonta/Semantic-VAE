import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math
from torch import Tensor
from torch.autograd import Variable

from inspect import isfunction
import math
from einops import rearrange, repeat

from models.networks.modules import res,att
from torchvision import transforms as T

class MaskEncoder(nn.Module):
    def __init__(self, latent_size, size = 128):
        super(MaskEncoder, self).__init__()  
        self.size = size 
        self.latent_size = latent_size

        self.encs = nn.Sequential(
            nn.Linear(self.size**2, self.latent_size*4),
            nn.ReLU(),
            nn.Linear(self.latent_size*4, self.latent_size*2),
            nn.ReLU(),
            nn.Linear(self.latent_size*2, self.latent_size),
            nn.ReLU()
        )
        self.mu = nn.Linear(self.latent_size, self.latent_size)
        self.log_var = nn.Linear(self.latent_size, self.latent_size)


    def forward(self, x):
        x = nn.functional.interpolate(x, size = (self.size, self.size))
        x = x.view(x.size(0), x.size(1), -1)
        x = self.encs(x)
        return self.mu(x), self.log_var(x)

class MaskDecoder(nn.Module):
    def __init__(self, latent_size, size = 256):
        super(MaskDecoder, self).__init__()  
        self.size = size 
        self.latent_size = latent_size

        # self.decs = nn.Sequential(
        #     nn.Linear(self.latent_size, self.size**2)
        #     # Activation function?
        # )
        # self.decs = nn.Sequential(
        #     nn.Linear(self.latent_size, self.latent_size*2),
        #     nn.LeakyReLU(),
        #     nn.Linear(self.latent_size*2, self.latent_size*4),
        #     nn.LeakyReLU(),
        #     nn.Linear(self.latent_size*4, self.size**2)
        # )

        self.decs = nn.Sequential(
            nn.Linear(self.latent_size*2, self.latent_size*4),
            nn.GELU(),
            nn.Linear(self.latent_size*4, self.latent_size)
        )
    
    def forward(self, x):
        
        x = self.decs(x)
        #x = x.view(x.size(0), 18, self.size, self.size)
        #x = nn.functional.interpolate(x, size = (self.size*2, self.size*2), mode='nearest')
        return x


class MaskDecoder_Conv(nn.Module):
    def __init__(self, latent_size, size = 256):
        super(MaskDecoder_Conv, self).__init__()  
        self.size = size 
        self.latent_size = latent_size
        ngf = 64

        self.in_conv = nn.Conv2d(18, ngf*8,1,1)
        #self.in_conv = nn.Conv2d(1, ngf*8,1,1)

        self.decs = nn.Sequential(
            res.ResBlock(ngf*8, dropout=0, out_channels=ngf*4, dims=2, up=True), #32x32
            res.ResBlock(ngf*4, dropout=0, out_channels=ngf*2, dims=2, up=True),
            res.ResBlock(ngf*2, dropout=0, out_channels=ngf, dims=2, up=True),
            res.ResBlock(ngf, dropout=0, out_channels=ngf//2, dims=2, up=True)
        )

        self.out_conv = nn.Conv2d(ngf//2, 18,1,1)

    
    def forward(self, x):
        x = x.view(x.size(0), 18, 16, 16)
        # x = x.view(x.size(0), 1, 16, 16)
        x = self.in_conv(x)
        x = self.decs(x)
        x = self.out_conv(x)
        #x = nn.functional.interpolate(x, size = (self.size*2, self.size*2), mode='nearest')
        return x

class VAE_net(nn.Module):
    def __init__(self, opt, nc, latent_variable_size):
        super(VAE_net, self).__init__()
        #self.cuda = True
        self.opt = opt
        if len(opt.gpu_ids) > 0:
            self.device = 'cuda:0'
        else:
            self.device = 'cpu'

        self.parts_for_dec = nc

        if self.opt.contain_dontcare_label:
            self.nc = nc+1
        else:
            self.nc = nc

        self.latent_variable_size = latent_variable_size


        self.encs = MaskEncoder(self.latent_variable_size)
        self.decs = MaskDecoder(self.latent_variable_size)
        self.decs_conv = MaskDecoder_Conv(self.latent_variable_size)

        if self.opt.no_T == False:
            print("Transformer Initialized")
            self.pos_enc = nn.Parameter(torch.zeros(self.nc, self.latent_variable_size))
            #self.pos_enc = PositionalEncoding(d_model = self.latent_variable_size)
            self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.latent_variable_size, nhead=8, batch_first=True)
            self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6, norm=nn.LayerNorm(self.latent_variable_size))
       
        if not self.opt.no_LSTM:
            if self.opt.bidir:
                self.opt.lstm_num = self.opt.lstm_num
                self.resh_lin = nn.Linear(self.latent_variable_size*2,self.latent_variable_size)
                self.lstm = nn.LSTM(self.latent_variable_size, self.latent_variable_size, num_layers = self.opt.lstm_num, batch_first=True, bidirectional=True)
            else:
                self.lstm = nn.LSTM(self.latent_variable_size, self.latent_variable_size, num_layers = self.opt.lstm_num, batch_first=True)            

        if self.opt.do_perm:
            self.permutations = torch.LongTensor([1,0,2] + list(range(3,18))).to(self.device)

        

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def encode_mask_parts(self, x):
        mu, log_var = self.encs(x)       
        z = self.reparametrize(mu, log_var)
        return mu, log_var, z
    
    def transformer_pass(self, z):
        z = z + self.pos_enc
        z = self.transformer_encoder(z)
        return z

    def encode(self, x):
        
        mu, log_var, z = self.encode_mask_parts(x)
        return mu, log_var, z
    
    def cross_att(self, out, s, c_layer):
        return c_layer(out,s)
    
    def kl_loss(self, mu, log_var):
        kld_loss = 0 #torch.Tensor([0]).to(self.device)
        
        # for i in range(mu.size(1)):
        #     mu_el = mu[:,i]
        #     lv_el = log_var[:,i]

        #     #kld_loss += (-0.5 * (1 + lv_el - mu_el ** 2 - lv_el.exp())).mean() #, dim = 1) torch.mean(, dim = 0)
        #     kld_loss += torch.mean(-0.5 * torch.sum(1 + lv_el - mu_el ** 2 - lv_el.exp(), dim = 1), dim = 0)

        # return torch.mean(kld_loss)

        mu_cat = mu.reshape(-1,self.nc*self.latent_variable_size)
        log_var_cat = log_var.reshape(-1,self.nc*self.latent_variable_size)

        kld_loss += torch.mean(-0.5 * torch.sum(1 + log_var_cat - mu_cat ** 2 - log_var_cat.exp(), dim = 1), dim = 0)
        
        return kld_loss #torch.mean(kld_loss)
    
    def decode(self, z):
        if self.opt.no_T == False:
            z = self.transformer_pass(z)
            return self.decs(z)
        #z = z.reshape(z.size(0), -1)

        if self.opt.do_perm:
            z_ = z.clone()
            # z_[:,2] = z[:,-1]
            # z_[:,-1] = z[:,2]

            z_ = z[:,self.permutations].clone()
        else:
            z_ = z.clone()
            
        if not self.opt.no_LSTM:
            if self.opt.bidir:
                hidden_size = self.opt.lstm_num*2
            else:
                hidden_size = self.opt.lstm_num
            hidden = (torch.randn(hidden_size, z.size(0), self.latent_variable_size).to(self.device), 
                    torch.randn(hidden_size, z.size(0), self.latent_variable_size).to(self.device))
            out, hidden = self.lstm(z_, hidden)
        else:
            out = z_
        if self.opt.bidir:  
            out = self.decs(out) + self.resh_lin(out)
        else:
            out = self.decs(out) + out

        return self.decs_conv(out)
        #return self.decs(z)

    def forward(self, x_org):
        x = x_org.clone()
        #remove bg
        if self.opt.exclude_bg:
            x = x[:,1:]
        mu, log_var, z = self.encode(x)
        return mu, log_var, self.decode(z)