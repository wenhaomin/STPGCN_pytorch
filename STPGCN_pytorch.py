# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 22:30:55 2021


@author: Haomin Wen

"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class STPGConv(nn.Module):
    def __init__(self, config):
        super(STPGConv, self).__init__()
        
        self.C = config.C
        self.d = config.d
        self.V = config.V
        self.t_size = config.t_size


        self.W = nn.Parameter(torch.randn(self.C, self.C*2))
        self.b = nn.Parameter(torch.randn(1, self.C*2))
        self.Ws = nn.Parameter(torch.randn(self.d, self.C*2))
        self.Wt = nn.Parameter(torch.randn(self.d, self.C*2))
        # self.ln  = nn.LayerNorm() #todo: add back the layer norm

    def forward(self, x, S, sape, tape):
        # x:B,t,V,C
        # S:B,V,tV
        # SE: V,d
        # TE: B,1,1,d
        
        # aggregation
        # B,t,V,C -> B,tV,C
        x = x.reshape((-1,self.t_size*self.V,self.C))
        # B,(V,tV x tV,C) -> B,V,C
        x = torch.dot(S, x)
        x = torch.dot(x, self.W)
        x += self.b

        
        # STPGAU
        # V,d x d,C -> V,C 
        SE = torch.dot(sape, self.Ws)
        # B,1,1,d -> B,1,1,d -> B,1,d
        TE = tape.rashpe((-1, 1, self.d))
        # B,1,d x d,C -> B,1,C 
        TE = torch.dot(TE,self.Wt)
        x += SE  # x +=  SE
        x += TE
        # x = self.ln(x)  # todo: add back the layernorm
        lhs, rhs = torch.chunk(x, chunks=2, dim=1)
        return lhs * F.sigmoid(rhs)


class Gaussian_component(nn.Module):
    def __init__(self, config):
        super(Gaussian_component, self).__init__()
        self.d = config.d
        self.device = config.device
        self.mu = nn.Parameter(torch.randn(1, self.d)).to(self.device)
        self.inv_sigma = nn.Parameter(torch.randn(1,self.d)).to(self.device)

    def forward(self, emb):
        # -1/2(emb - mu)**2/sigma**2
        # e = -0.5 * F.power(F.broadcast_sub(emb, mu),2)
        # e = F.broadcast_mul(e, F.power(inv_sigma,2))

        e =  -0.5 * torch.pow(emb -  self.mu.expand_as(emb),2)
        e =  e * torch.pow(self.inv_sigma, 2).expand_as(emb)
        # return F.sum(e,axis=-1,keepdims=True)

        return torch.sum(e, dim=-1, keepdim=True)


class STPRI(nn.Module):
    """Spatial-Temporal Position-aware Relation Inference"""
    def __init__(self, config):
        super(STPRI, self).__init__()
        
        self.d = config.d
        self.V = config.V
        self.t_size = config.t_size

        self.gc_lst = []
        for i in range(6):
            self.gc_lst.append(Gaussian_component(config))
            # self.register_child(self.gc_lst[-1])

    def forward(self,  sape, tape_i, tape_j, srpe, trpe):
        """
        sape:V,d
        tape:B,T,1,d
        srpe:V,V,d
        trpe:t,1,d
        """  

        # V,d -> V,1
        sapei = self.gc_lst[0](sape)
        # V,d -> V,1 -> 1,V
        sapej = self.gc_lst[1](sape)
        sapej = sapej.transpose(1,0)

        # V,1 + 1,V -> V,V
        # gaussian = F.broadcast_add(sapei, sapej)
        gaussian = sapei.expand(self.V, self.V) + sapej.expand(self.V, self.V)


        # B,t,1,d -> B,t,1,1
        tapei = self.gc_lst[2](tape_i)
        # B,t,1,1 + V,V -> B,t,V,V
        gaussian = gaussian + tapei
        # B,t,1,d -> B,t,1,1
        tapej = self.gc_lst[3](tape_j)
        # B,t,1,1 + V,V -> B,t,V,V
        gaussian = F.broadcast_add(gaussian, tapej)
        
        # V,V,d -> V,V,1 -> V,V
        srpe = F.squeeze(self.gc_lst[4](srpe))
        # B,t,V,V + V,V -> B,t,V,V
        gaussian = F.broadcast_add(gaussian, srpe)
        
        # t,1,d -> t,1,1
        trpe = self.gc_lst[5](trpe)
        # B,t,V,V + t,1,1 -> B,t,V,V
        gaussian = F.broadcast_add(gaussian, trpe)
        
        # B,t,V,V -> B,tV,V -> B,V,tV
        gaussian = F.reshape(gaussian, (-1,self.t_size*self.V,self.V))
        gaussian = F.transpose(gaussian,(0,2,1))
        
        return F.exp(gaussian)
    


class GLU(nn.Module):
    def __init__(self, dim, **kwargs):
        super(GLU, self).__init__(**kwargs)
        self.linear = nn.Conv2d(in_channels=dim, out_channels=dim * 2, kernel_size=(1, 1))


    def forward(self, x):
        # B,C,V,T
        x = self.linear(x)
        lhs, rhs = torch.chunk(x, chunks=2, dim=1)
        return lhs * F.sigmoid(rhs)

class GFS(nn.Module):
    """gated feature selection module"""
    def __init__(self, config):
        super(GFS, self).__init__()
        self.fc  = nn.Conv2d(in_channels=config.C, out_channels=config.C, kernel_size=(1, config.C)) #todo: has problem, #todo@:shixiong
        self.glu = GLU(config.C)
            
    def forward(self,  x):

        x = self.fc(x)
        x = self.glu(x)
        return x




class OutputLayer(nn.Module):
    def __init__(self, config, **kwargs):
        super(OutputLayer, self).__init__(**kwargs)
        self.V = config.V
        self.D = config.num_features
        self.P = config.num_prediction
        self.C = config.C
        self.fc = nn.Conv2d(in_channels=self.C, out_channels=self.P*self.D, kernel_size=(1, 1)) #todo@:shixiong

    def forward(self, x):
        # x:B,C',V,1 -> B,PD,V,1 -> B,P,V,D
        x = self.fc(x)
        if self.D>1:
            # x = F.reshape(x,(0,self.P,self.D,self.V))
            # x = F.transpose(x, (0,1,3,2))
            #
            x = x.reshape((0,self.P,self.D,self.V))
            x = x.transpose( (0,1,3,2))
        return x
    


class InputLayer(nn.Module):
    def __init__(self, config, **kwargs):
        super(InputLayer, self).__init__(**kwargs)
        self.fc = nn.Linear(in_features=config.D, out_features=config.C)

    def forward(self, x):
        x = self.fc(x)
        return x
        # x:B,T,V,D -> B,T,V,C




class STPGCNs(nn.Module):
    """Spatial-Temporal Position-aware Graph Convolution"""
    def __init__(self, config, **kwargs):
        super(STPGCNs, self).__init__(**kwargs)

        self.config = config
        self.L = config.L
        self.d = config.d
        self.C = config.C
        self.V = config.V
        self.T = config.T
        self.t_size = config.t_size


        self.input_layer = InputLayer(self.config)
        self.fs = GFS(self.config)

        self.ri_lst = []
        self.gc_lst = []
        self.fs_lst = []
        for i in range(self.L):
            self.ri_lst.append(STPRI(self.config))
            # self.register_child(self.ri_lst[-1])

            self.gc_lst.append(STPGConv(self.config))
            # self.register_child(self.gc_lst[-1])

            self.fs_lst.append(GFS(self.config))
            # self.register_child(self.fs_lst[-1])

        self.glu = GLU(self.C*4)
        self.output_layer = OutputLayer(self.config)

    def forward(self, x, sape, tape, srpe, trpe, zeros_x, zeros_tape, range_mask):
        """
        x:B,T,V,D
        sape:V,d
        tape:B,T,1,d
        srpe:V,V,d
        trpe:t,1,d
        zeros_x:B,beta,V,D
        zeros_tape:B,beta,1,d
        range_mask:B,V,tV
        """          
        
        # x:B,T,V,D -> B,T,V,C
        x = self.input_layer(x)
        # padding: B,T+beta,1,d
        tape_pad = torch.cat((zeros_tape, tape), dim=1)

        # skip = [self.fs(x)] #todo: add back
        skip = []
        for i in range(self.L):
            # padding: B,T+beta,V,C
            x = torch.cat((zeros_x, x), dim=1)
            
            xs = []
            for t in range(self.T):
                # B,t,V,C
                # xj  = F.slice_axis(x,  axis=1, begin=t, end=t+self.t_size)
                xj = x[:,t:t+self.t_size,:,:]
                # B,1,1,C
                # tape_i = F.slice_axis(tape, axis=1, begin=t, end=t+1)
                tape_i = tape[:,t:t+1,:,:]
                # B,t,1,C
                # tape_j = F.slice_axis(tape_pad, axis=1, begin=t, end=t+self.t_size)
                tape_j = tape_pad[:,t+self.t_size,:,:]
                
                # Inferring spatial-temporal relations
                S = self.ri_lst[i](sape, tape_i, tape_j, srpe, trpe)
                # S = F.broadcast_mul(S,range_mask)
                S = S * range_mask
                
                # STPGConv
                xs.append(self.gc_lst[i](xj, S, sape, tape_i))
                
            x = torch.stack(*xs, dim=1)
            #B,T,V,C->B,C',V,1
            skip.append(self.fs_lst[i](x))
        
        # B,T,V,C->B,LD,V,1
        x = torch.cat(*skip,dim=1)
        
        # B,LD,V,1 -> B,C,V,1
        x = self.glu(x)
        
        # B,C,V,1 -> B,PF,V,1 -> B,P,V,D
        x = self.output_layer(x)
        return x


class SAPE(nn.Module):
    """Spatial Absolute-Position Embedding"""
    def __init__(self, config, **kwargs):
        super(SAPE, self).__init__(**kwargs)

        # self.sape = self.params.get('SE', shape=(config.V, config.d))
        self.sape = nn.Embedding(config.V, config.d)
            
    def forward(self):
        # return self.sape.data()
        return self.sape.weight


class TAPE(nn.Module):
    """Temporal Absolute-Position Embedding"""

    def __init__(self, config, **kwargs):
        super(TAPE, self).__init__(**kwargs)

        self.dow_emb = nn.Embedding(config.week_len, config.d)
        self.tod_emb = nn.Embedding(config.day_len, config.d)


    def forward(self, pos_w, pos_d):
        # B,T,i -> B,T,1,C
        dow = self.dow_emb(pos_w).unsqueeze(2)
        tod = self.tod_emb(pos_d).unsqueeze(2)
        return dow + tod
    
class SRPE(nn.Module):
    """Spatial Relative-Position Embedding"""
    def __init__(self, config, **kwargs):
        super(SRPE, self).__init__(**kwargs)


        # self.SDist = torch.Tensor(config.spatial_distance, dtype=torch.int).to(config.device)
        self.SDist = torch.from_numpy(config.spatial_distance).to(config.device).long()
        self.srpe = nn.Parameter(torch.randn(config.alpha+1, config.d))
            
    def forward(self):
        # return self.srpe.data()[self.SDist]
        return self.srpe[self.SDist]

class TRPE(nn.Module):
    """Temporal Relative-Position Embedding"""
    def __init__(self, config, **kwargs):
        super(TRPE, self).__init__(**kwargs)


        # self.TDist = torch.Tensor(np.expand_dims(range(config.t_size),-1), dtype=torch.int, device=config.device)
        self.TDist = torch.from_numpy(np.expand_dims(range(config.t_size), -1)).to(config.device).long()

        self.trpe = nn.Parameter(torch.randn(config.t_size,  config.d))


            
    def forward(self):
        # return self.trpe.data()[self.TDist]
        return self.trpe[self.TDist]


class GeneratePad(nn.Module):
    def __init__(self, config,  **kwargs):
        super(GeneratePad, self).__init__(**kwargs)
        self.device = config.device
        self.C = config.C
        self.V = config.V
        self.d = config.d
        self.pad_size = config.beta
        
    def forward(self, x):
        B = x.shape[0]
        return torch.zeros((B,self.pad_size,self.V,self.C), device=self.device),torch.zeros((B,self.pad_size,1,self.d), device=self.device)



class Model(nn.Module):
    def __init__(self, config,  **kwargs):
        super(Model, self).__init__(**kwargs)

        self.config = config
        self.T = config.T
        self.V = config.V
        self.C = config.C
        self.L = config.L
        self.range_mask = torch.Tensor(config.range_mask).to(config.device)

        
        self.PAD  = GeneratePad(self.config)
        self.SAPE = SAPE(self.config)
        self.TAPE = TAPE(self.config)
        self.SRPE = SRPE(self.config)
        self.TRPE = TRPE(self.config)
        self.net = STPGCNs(self.config)

    def forward(self, x, pos_w, pos_d):
        # x:B,T,V,D
        # pos_w:B,t,1,1
        # pos_d:B,t,1,1
        sape = self.SAPE()
        tape = self.TAPE(pos_w, pos_d)
        srpe = self.SRPE()
        trpe = self.TRPE()
        zeros_x, zeros_tape = self.PAD(x)
        
        x = self.net(x, sape, tape, srpe, trpe, zeros_x, zeros_tape, self.range_mask)
        return x

