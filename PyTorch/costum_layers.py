#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 15:22:38 2018

@author: danny
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

############################# Costum implementation of Recurrent Highway Networks #####################


# implementation of recurrent highway networks using existing PyTorch layers (GRU and linear)
class RHN(nn.Module):
    def __init__(self, in_size, hidden_size, n_steps):
        super(RHN, self).__init__()
        self.n_steps = n_steps
        # create a GRU layer. The first step of an RHN is basically a GRU, only 
        # after applying the GRU operations we apply n microsteps. 
        self.GRU = nn.GRU(in_size, hidden_size)
        # create 3 linear layers serving as the hidden, transform and carry gate,
        # one each for each microstep. 
        self.H, self.T, self.C = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        layer_list = [self.H, self.T, self.C]
        for steps in range(self.n_steps):
            for layers, lists in zip(self.init_microstep(hidden_size), layer_list):
                lists.append(layers)
                
    # initialise linear layers for the microsteps
    def init_microstep(self, n_nodes):       
        H = nn.Linear(n_nodes,n_nodes)
        T = nn.Linear(n_nodes,n_nodes)
        C = nn.Linear(n_nodes,n_nodes)
        return(H,T,C)
    
    def perform_microstep(self, input, step):
        output = nn.functional.tanh(self.H[step](input)) * nn.functional.sigmoid(self.T[step](input)) + input * (1 - nn.functional.sigmoid(self.C[step](input)))
        return(output)
        
    def forward(self, input):
        # list to append the output of each time step to
        output = []
        # loop through all time steps
        for x in input:
            # apply the GRU
            x , hx = self.GRU(x.view(-1, x.size(0), x.size(1)))
            #hx = hx.squeeze()
            # apply the microsteps to the hidden state of the GRU
            for step in range(self.n_steps):            
                x = self.perform_microstep(x, step)
            # append the hidden state of time step n to the output. 
            output.append(x)
        return torch.cat(output)

# implementation of recurrent highway networks using existing PyTorch layers (GRU and linear)
class RHN_2(nn.Module):
    def __init__(self, in_size, hidden_size, n_steps):
        super(RHN_2, self).__init__()
        self.n_steps = n_steps
        self.initial_state = torch.autograd.Variable(torch.rand(1, 2, hidden_size))
        # create 3 linear layers serving as the hidden, transform and carry gate,
        # one each for each microstep. 
        self.H, self.T, self.C = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        self.init_h = (nn.Linear(in_size, hidden_size))
        self.init_t = (nn.Linear(in_size, hidden_size))
        self.init_c = (nn.Linear(in_size, hidden_size))
        self.tan = nn.Tanh()
        self.sig = nn.Sigmoid()
        
        layer_list = [self.H, self.T, self.C]
        for steps in range(self.n_steps):
            for layers, lists in zip(self.create_microstep(hidden_size), layer_list):
                lists.append(layers)
                
    # initialise linear layers for the microsteps
    def create_microstep(self, n_nodes):       
        H = nn.Linear(n_nodes,n_nodes)
        T = nn.Linear(n_nodes,n_nodes)
        C = nn.Linear(n_nodes,n_nodes)
        return(H,T,C)
    
    def calc_h(self, x, hx, step):
        return self.tan(((step+1)//1 * self.init_h(x)) + self.H[step](hx))
    
    def calc_t(self, x, hx, step):
        return self.sig(((step+1)//1 * self.init_t(x)) + self.T[step](hx))
    
    def calc_c(self, x, hx, step):
        return self.sig(((step+1)//1 * self.init_c(x)) + self.C[step](hx))
    
    def perform_microstep(self, x, hx, step):
        output = self.calc_h(x, hx, step) * self.calc_t(x, hx, step) + hx * (1 - self.calc_c(x, hx, step))
        return(output)
        
    def forward(self, input):
        # list to append the output of each time step to
        output = []
        hx = self.initial_state
        # loop through all time steps
        for x in input:
            # apply the microsteps to the hidden state of the GRU
            for step in range(self.n_steps):
                hx = self.perform_microstep(x, hx, step)
            # append the hidden state of time step n to the output. 
            output.append(hx)
        return torch.cat(output)

# attention layer for the RHN audio encoder
class attention(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super(attention, self).__init__()
        self.hidden = nn.Linear(in_size, hidden_size)
        self.out = nn.Linear(hidden_size, out_size)
        self.softmax = nn.Softmax(dim = 0)
    def forward(self, input):
        # calculate the attention weights
        att_weights = self.out(nn.functional.tanh(self.hidden(input)))
        # apply the weights to the input and sum over all timesteps
        x = torch.sum(self.softmax(att_weights) * input, 0)
        return x
           
        