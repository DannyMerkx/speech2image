#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 13:34:33 2018

@author: danny
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 10:36:08 2018

@author: danny

evaluation functions. contains only recall@n now. convenience functions for embedding
the data and calculating the recall@n to keep your NN training script clean
"""
import numpy as np
import time
from evaluate import recall_at_n

embeddings_1 = [np.random.rand(2**8, 1024)]
embeddings_2 = [np.random.rand(2**8, 1024)]

start_time = time.time()
 
recall1, avg_rank1 = recall_at_n(embeddings_1,embeddings_2, [1,5,10], mode = 'full')

print("full mode took {:.3f}s".format( time.time() - start_time))

start_time = time.time()
 
recall2, avg_rank2 = recall_at_n(embeddings_1,embeddings_2, [1,5,10], mode = 'array')

print("array mode took {:.3f}s".format( time.time() - start_time))

start_time = time.time()
 
recall3, avg_rank3 = recall_at_n(embeddings_1, embeddings_2, [1,5,10], mode = 'big')

print("big mode took {:.3f}s".format( time.time() - start_time))