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

# implementation of recurrent highway networks using existing PyTorch layers
class RHN(nn.Module):
    def __init__(self, in_size, hidden_size, n_steps, batch_size):
        super(RHN, self).__init__()
        self.n_steps = n_steps
        self.initial_state = torch.autograd.Variable(torch.rand(1, batch_size, hidden_size)).gpu()
        # create 3 linear layers serving as the hidden, transform and carry gate,
        # one each for each microstep. 
        self.H, self.T, self.C = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        # linear layers for the input in the first microstep
        self.init_h = (nn.Linear(in_size, hidden_size))
        self.init_t = (nn.Linear(in_size, hidden_size))
        self.init_c = (nn.Linear(in_size, hidden_size))
        self.tan = nn.Tanh()
        self.sig = nn.Sigmoid()
        # linear layers for the history in the microsteps
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
    # the input is only used in the first microstep. For the first step the history is a random initial state.
    def calc_htc(self, x, hx, step, non_linearity):
        if step == 0:
            return non_linearity(((step+1)//1 * self.init_h(x)) + self.H[step](hx))
        else:
            return non_linearity(self.H[step](hx))
                                 
    def perform_microstep(self, x, hx, step):
        output = self.calc_htc(x, hx, step, self.tan) * self.calc_htc(x, hx, step, self.sig) + hx * (1 - self.calc_htc(x, hx, step,self.sig))
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
    def __init__(self, in_size, hidden_size):
        super(attention, self).__init__()
        self.hidden = nn.Linear(in_size, hidden_size)
        self.out = nn.Linear(hidden_size, 1)
        self.softmax = nn.Softmax(dim = 1)
    def forward(self, input):
        # calculate the attention weights
        alpha = self.softmax(self.out(nn.functional.tanh(self.hidden(input))))
        # apply the weights to the input and sum over all timesteps
        x = torch.sum(alpha.expand_as(input) * input, 1)
        return x
           
        
