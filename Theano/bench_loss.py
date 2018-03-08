#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 11:11:56 2018

@author: danny
"""

import time
import numpy as np

from costum_loss import cosine_hinge_loss, dot_hinge_loss, l2norm_hinge_loss, batch_hinge_loss

start_time = time.time()

for x in range(0,1000):
    embeddings_1 = np.random.rand(128, 1024)
    embeddings_2 = np.random.rand(128, 1024)
    
    loss = cosine_hinge_loss(embeddings_1, embeddings_2)

print("cosine hinge took {:.3f}s".format(time.time() - start_time))

start_time = time.time()

for x in range(0,1000):
    embeddings_1 = np.random.rand(128, 1024)
    embeddings_2 = np.random.rand(128, 1024)
    
    loss = dot_hinge_loss(embeddings_1, embeddings_2)

print("dot hinge took {:.3f}s".format(time.time() - start_time))

start_time = time.time()

for x in range(0,1000):
    embeddings_1 = np.random.rand(128, 1024)
    embeddings_2 = np.random.rand(128, 1024)
    
    loss = l2norm_hinge_loss(embeddings_1, embeddings_2)

print("l2 norm hinge took {:.3f}s".format(time.time() - start_time))

start_time = time.time()

for x in range(0,1000):
    embeddings_1 = np.random.rand(128, 1024)
    embeddings_2 = np.random.rand(128, 1024)
    
    loss = batch_hinge_loss(embeddings_1, embeddings_2, [False, False])

print("batch dot hinge took {:.3f}s".format(time.time() - start_time))

start_time = time.time()

for x in range(0,1000):
    embeddings_1 = np.random.rand(128, 1024)
    embeddings_2 = np.random.rand(128, 1024)
    
    loss = batch_hinge_loss(embeddings_1, embeddings_2, [True,True])
    
print("batch cosine hinge took {:.3f}s".format(time.time() - start_time))

start_time = time.time()

for x in range(0,1000):
    embeddings_1 = np.random.rand(128, 1024)
    embeddings_2 = np.random.rand(128, 1024)
    
    loss = batch_hinge_loss(embeddings_1, embeddings_2, [True,False])
print("batch l2norm hinge took {:.3f}s".format(time.time() - start_time))
