#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 14:40:29 2020

@author: danny
"""
import torch.nn as nn
from encoders import *
img_net = img_encoder(image_config)
cap_net = quantized_encoder(token_config)
    
trainer = flickr_trainer(img_net, cap_net, args.visual, args.cap)
trainer.set_loss(batch_hinge_loss)
trainer.set_optimizer(optimizer)
trainer.set_token_batcher()
trainer.set_dict_loc(args.dict_loc)
trainer.set_lr_scheduler(cyclic_scheduler, 'cyclic')
trainer.set_att_loss(attention_loss)

img, cap, lengths = next(trainer.batcher(train, 5, 64, shuffle = False))

trainer.cap_embedder.att.register_backward_hook(track_grads)
#trainer.cap_embedder.quant.register_backward_hook(track_grads)
trainer.cap_embedder.RNN.register_backward_hook(track_grads)

img, cap = trainer.dtype(img), trainer.dtype(cap)

img_embedding = trainer.img_embedder(img)
cap_embedding = trainer.cap_embedder(cap, lengths)

loss = trainer.loss(img_embedding, cap_embedding, trainer.dtype)

loss.backward()


def track_grads(self, grad_input, grad_output):
    #print(grad_output[0])
    for g in grad_input:
        if not type(g) == type(None):
            print(f'g-in {g.size()}')
    for g in grad_output:
        print(f'g-out {g.size()}')
        
def skip_layer(self, grad_input, grad_output):
    #print(grad_output[0].size()
    #print(type(grad_input))
    #print(tuple(grad_output[0],))
    print(grad_output[0].permute(0,1,2).size())
    print(grad_input[0].size())
    return (grad_output[0],)


input = torch.randint(10,[32,100]).float()

lin = nn.Linear(100,1000)
#lin.register_backward_hook(track_grads)
lin2 = nn.Linear(1000,1500)
lin2.register_backward_hook(track_grads)
lin_skip = nn.Linear(1500,1500)
lin_skip.register_backward_hook(track_grads)
lin3 = nn.Linear(1500, 2)
#lin3.register_backward_hook(track_grads)

output = lin3(lin_skip(lin2(lin(input))))

loss = output.mean()
loss.backward()