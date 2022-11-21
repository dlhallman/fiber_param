import torch
import torch.nn as nn
import numpy as np



class Encoder(nn.Module):
    def __init__(self, latent_dim, obs_dim, hidden_units, hidden_layers):
        super(Encoder, self).__init__()
        # self.rnn = nn.GRU(obs_dim, hidden_units, hidden_layers, dtype=torch.float64)
        self.h2o = nn.Linear(obs_dim, latent_dim * 2 ,dtype=torch.float64)
        
    def forward(self, x):
        # y, _ = self.rnn(x)
        # y = y[:, -1, :]
        y = self.h2o(x)
        return y

class odeFunc(nn.Module):
    ' dh/dt = A h'
    def __init__(self, input_dim, output_dim):
        super(odeFunc, self).__init__()
        self.mu = np.array([0,0])
        self.A = nn.Linear(input_dim, output_dim, bias=False, dtype=torch.float64)
        self.b = nn.Linear(2, output_dim, bias=False, dtype=torch.float64)
        self.nfe = 0 #number of function evaluations
    def forward(self, t, x):
        self.nfe += 1
        return self.A(x)
        


class Decoder(nn.Module):
    def __init__(self, latent_dim, obs_dim, hidden_units, hidden_layers):
        super(Decoder, self).__init__()
        self.act = nn.Tanh()
        # self.rnn = nn.GRU(latent_dim, hidden_units, hidden_layers, dtype=torch.float64)
        # self.h1 = nn.Linear(hidden_units, hidden_units - 5, dtype=torch.float64)
        self.h2 = nn.Linear(latent_dim, obs_dim, dtype=torch.float64)

    def forward(self, x):
        # y, _ = self.rnn(x)
        # y = self.h1(y)
        # y = self.act(y)
        y = self.h2(x)
        return y

"""
NEURAL ODE NETWORKS
    - Neural ODE: updates the hidden state h from h'=LatentODE
    - Heavy-Ball Neural ODE: learns hidden state h from h'+gamma m=LatentODE

"""
class NODE(nn.Module):
    def __init__(self, df=None, **kwargs):
        super(NODE, self).__init__()
        self.__dict__.update(kwargs)
        self.df = df
        self.nfe = 0
    def forward(self, t, x):#dy/dt = f(t,x) 
        self.nfe += 1
        return self.df(t, x) 
    def getStiff(self):
        eigvals = torch.linalg.eigvals(self.df.A.weight)
        eigvals = torch.abs(eigvals)
        maxeig = torch.max(eigvals)
        mineig = torch.min(eigvals)
        return maxeig/mineig


class HBNODE(NODE):
    def __init__(self, df, actv_h=None, gamma_guess=-3.0, gamma_act='sigmoid', corr=-100, corrf=True):
        super().__init__(df)
        # Momentum parameter gamma
        self.gamma = nn.Parameter(torch.tensor(gamma_guess,dtype=torch.float64))
        self.gammaact = nn.Sigmoid() if gamma_act == 'sigmoid' else gamma_act
        self.corr = nn.Parameter(torch.tensor(corr,dtype=torch.float64))
        self.sp = nn.Softplus()
        # Activation for dh, GHBNODE only
        self.actv_h = nn.Identity() if actv_h is None else actv_h
    def forward(self, t, x): 
        self.nfe += 1
        h, m = torch.split(x, x.shape[-1]//2, dim=1)
        dh = self.actv_h(- m) #dh/dt = m
        dm = self.df(t, h) - self.gammaact(self.gamma) * m #dm/dt = -gamm *m + f(t, h)!(network)!
        dm = dm + self.sp(self.corr) * h
        out = torch.cat((dh, dm), dim=1)
        return out
    def getStiff(self):
        In = torch.eye(self.df.A.weight.shape[0])
        Zeros = torch.zeros_like(self.df.A.weight)
        Matrix1 = torch.hstack([Zeros, In])
        Matrix2 = torch.hstack([self.df.A.weight, self.gamma*In])
        Matrix = torch.vstack([Matrix1, Matrix2])
        eigvals = torch.linalg.eigvals(Matrix)
        eigvals = torch.abs(eigvals)
        maxeig = torch.max(eigvals)
        mineig = torch.min(eigvals)
        return maxeig/mineig