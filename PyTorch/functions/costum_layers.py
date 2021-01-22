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
import numpy as np

import torch.autograd
#########################attention layer for rnns##############################
        
# class for making multi headed attention layers. This is not attention as used
# in Transformer architectures, I use this layer as an alternative to mean/max
# pooling over time to create sentence embeddings.  
class multi_attention(nn.Module):
    def __init__(self, in_size, hidden_size, n_heads):
        super(multi_attention, self).__init__()
        self.att_heads = nn.ModuleList()
        for x in range(n_heads):
            self.att_heads.append(attention(in_size, hidden_size))
    def forward(self, input):
        out, self.alpha = [], []
        for head in self.att_heads:
            o = head(input)
            out.append(o) 
            # save the attention matrices to be able to use them in a loss 
            # function
            self.alpha.append(head.alpha)
        # return the resulting embedding 
        return torch.cat(out, 1)
    
# attention layer for audio encoders
class attention(nn.Module):
    def __init__(self, in_size, hidden_size):
        super(attention, self).__init__()
        self.hidden = nn.Linear(in_size, hidden_size)
        nn.init.orthogonal_(self.hidden.weight.data)
        self.out = nn.Linear(hidden_size, in_size)
        nn.init.orthogonal_(self.hidden.weight.data)
        self.softmax = nn.Softmax(dim = 1)
    def forward(self, input):
        # calculate the attention weights
        self.alpha = self.softmax(self.out(torch.tanh(self.hidden(input))))
        # apply the weights to the input and sum over all timesteps
        x = torch.sum(self.alpha * input, 1)
        # return the resulting embedding
        return x   
###############################################################################

####################### Vector Quantization layer #############################

class VQ_EMA_layer(nn.Module):
    def __init__(self, num_emb, emb_dim):
        super(VQ_EMA_layer, self).__init__()
        self.num_emb = num_emb
        # create the embedding layer and initialise with uniform distribution
        self.embed = nn.Embedding(num_emb, emb_dim)
        self.embed.weight.data.uniform_(-1/num_emb, 1/num_emb)
        
        # set the exponential moving average 
        self.register_buffer('_ema_cluster_size', torch.zeros(num_emb))
        self._ema_w = nn.Parameter(self.embed.weight.clone())
        self._ema_w.requires_grad_(False)
        self._decay = 0.99
        self._epsilon = 1e-5
        # set the costum autograd function for quantization
        self.quant_emb = quantization_emb.apply
                
    # expected input dimensions is Batch/Signal-length/Channels
    def forward(self, input):
        input_shape = input.shape
        # flatten accross B/SL dims
        flat_input = input.view(-1, input_shape[-1])
        # retrieve the quantized input and encoding indices
        quantized, enc_idx, idx = self.quant_emb(flat_input, self.embed)
        
        self.idx = idx
        quantized = quantized.view(input_shape)
        # update the exponential moving average
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(enc_idx, 0)
    
            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = ((self._ema_cluster_size + self._epsilon)
                                      / (n + self.num_emb * self._epsilon) * n
                                      )
    
            dw = torch.matmul(enc_idx.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + 
                                       (1 - self._decay) * dw)
            
            self.embed.weight = nn.Parameter(self._ema_w / 
                                             self._ema_cluster_size.unsqueeze(1)
                                             )
        # loss based on distance between input and embeddings
        e_latent_loss = torch.nn.functional.mse_loss(input, quantized.detach())
        loss = 0.25 * e_latent_loss

        #quantized = input + (quantized - input).detach()
        avg_probs = torch.mean(enc_idx, dim = 0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs 
                                                                + 1e-10
                                                                )
                                          )
                               )
        return quantized, loss
 
# autograd function for the embedding mapping which also implements the 
# skipping the gradient for this layer. 
class quantization_emb(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, emb):
        # calculate input and embedding norms and the input-emb distance
        i_norm = (input**2).sum(1).view(-1, 1)
        w_norm = (emb.weight**2).sum(1).view(1, -1)

        dist = i_norm + w_norm - 2.0 * \
               torch.mm(input, torch.transpose(emb.weight, 0, 1)) 
        # retrieve closest embedding and create onehot encoding vector   
        idx = dist.argmin(-1)   
        #smooth(idx)
        one_hot_idx = nn.functional.one_hot(idx, emb.num_embeddings).float()
        quantized = torch.mm(one_hot_idx, emb.weight)
        # return quantized input and the one_hot_idx
        return quantized, one_hot_idx, idx

    @staticmethod
    def backward(ctx, grad_quant, grad_one_hot_idx, grad_idx):
        # simply pass the gradient of the quantized embeddings
        return grad_quant, None

###################### residual convolutional layer ###########################
# residual convolution block as used by Harwath et al. in DAVE-net. it is a
# residual convolution block with 4 conv layers and 2 residual connections that
# should, depending on your kernel and stride size, correctly resize the dims.
class res_conv(nn.Module):
    def __init__(self, in_ch, out_ch, ks, stride):
        super(res_conv, self).__init__()   
        # the first layer can optionally downsample the time dimension and 
        # change the number of channels. If this happens we need to downsample
        # and up/downscale the residuals as well.
        self.req_downsample = stride != 1
        # if the number of channels is changed, residual needs an up/downscale
        self.req_rescale = in_ch != out_ch
        # only works for uneven kernel sizes and even strides
        pad = int((ks -1)/ 2)
        self.conv_1 = nn.Sequential(nn.Conv1d(in_ch, out_ch, kernel_size = ks, 
                                              stride = stride, padding = pad
                                              ),
                                    nn.BatchNorm1d(out_ch)
                                    )
        
        self.conv_2 = nn.Sequential(nn.Conv1d(out_ch, out_ch, kernel_size = ks,
                                              padding = pad
                                              ),
                                    nn.BatchNorm1d(out_ch)
                                    )
        
        self.conv_3 = nn.Sequential(nn.Conv1d(out_ch, out_ch, kernel_size = ks,
                                              padding = pad
                                              ),
                                    nn.BatchNorm1d(out_ch)
                                    )
        
        self.conv_4 = nn.Sequential(nn.Conv1d(out_ch, out_ch, kernel_size = ks, 
                                              padding = pad
                                              ),
                                    nn.BatchNorm1d(out_ch)
                                    )
        
        self.relu = nn.functional.relu
        # layer to downsample the residual using maxpool
        self.downsample = nn.MaxPool1d(stride, stride, padding = 0)
        # layer to up or downscale the residual channels
        self.rescale = nn.Conv1d(in_ch, out_ch, kernel_size = 1, stride = 1)
           
    def forward(self, input):
        residual = input
        conv_out_1 = self.conv_2(self.relu(self.conv_1(input)))
        # if input is downsampled or n_channels increases, transform input to
        # match conv_out
        if self.req_downsample:
            # if the input size was uneven, a onesided 0 pad is required.
            if not input.size(-1) % 2 == 0:
                residual = nn.functional.pad(residual, [0, 1])
            residual = self.downsample(residual)
        if self.req_rescale:
            residual = self.rescale(residual)
            
        residual = self.relu(conv_out_1 + residual)
        
        conv_out_2 = self.conv_3(self.relu(self.conv_4(residual)))
        output = self.relu(conv_out_2 + residual)
        
        return output
################################ Transformer Layers ###########################
# costum implementation of the Transformer. Implements the Decoder and Encoder
# cells, transformer attention and costum forward functions for training 
# transformer models. Encoders should inherit the Transformer superclass 
# for acces to useful functions such as creating the masks and positional 
# embeddings

# single encoder transformer cell with h attention heads fully connected layer
# block and residual connections
class transformer_encoder_cell(nn.Module):
    def __init__(self, in_size, fc_size, h):
        super(transformer_encoder_cell, self).__init__()
        # assert input size is compatible with the number of attention heads
        assert in_size % h == 0
        # create the attention layer
        self.att_heads = transformer_att(in_size, h)
        # create the linear layer block
        self.ff = transformer_ff(in_size, fc_size)
        # the layernorm and dropout functions      
        self.norm_att = nn.LayerNorm(in_size)
        self.norm_ff = nn.LayerNorm(in_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input, mask = None):
        # apply the attention block to the input. In an encoder q=k=v
        att = self.att_heads(input, input, input, mask)
        # apply the residual connection and layer normalisation
        norm_att = self.norm_att(self.dropout(att) + input)        
        # apply the linear layer block
        lin = self.ff(norm_att)
        # apply  residual connection and layer normalisation
        out = self.norm_ff(self.dropout(lin) + norm_att)
        return out

# decoder cell, has an extra attention block which looks at encoder cell output
class transformer_decoder_cell(nn.Module):
    def __init__(self, in_size, fc_size, h):
        super(transformer_decoder_cell, self).__init__()
        # assert input size is compatible with the number of attention heads
        assert in_size % h == 0
        # create the two attention layers
        self.att_one = transformer_att(in_size, h)
        self.att_two = transformer_att(in_size, h)
        # linear layer block
        self.ff = transformer_ff(in_size, fc_size)
        # the layernorm and dropout functions     
        self.norm_att_one = nn.LayerNorm(in_size)
        self.norm_att_two = nn.LayerNorm(in_size)
        self.norm_ff = nn.LayerNorm(in_size)
        self.dropout = nn.Dropout(0.1)
    # decoders have a mask for each attention layer i.e. for translation tasks
    # to allow full access to encoder output but restricted access to decoder 
    # input 
    def forward(self, input, mask_1 = None, mask_2 = None, enc_output = None):
        # apply first attention block, in the first layer q=k=v
        att = self.att_one(input, input, input, mask_1)
        # apply residual connection and layer normalisation
        norm_att = self.norm_att_one(self.dropout(att) + input)
        # apply second attention block. q is the intermediate decoder output
        # k and v are the encoder output. If endocer output is not given act 
        # like an encoder with 2 attention layers.  
        if enc_output is None:
            enc_output = norm_att
            mask_2 = mask_1
        att_2 = self.att_two(norm_att, enc_output, enc_output, mask_2)

        # apply residual connection and layer normalisation
        norm_att_2 = self.norm_att_two(self.dropout(att_2) + norm_att)
        # apply the linear layer block
        lin = self.ff(norm_att_2)
        # apply residual connection and layer normalisation
        out = self.norm_ff(self.dropout(lin) + norm_att_2)
        return out
    
# the linear layer block of the transformer
class transformer_ff(nn.Module):
    def __init__(self, in_size, fc_size):
        super(transformer_ff, self).__init__()
        # the linear layers of feed forward block
        self.ff_1 = nn.Linear(in_size, fc_size)
        self.ff_2 = nn.Linear(fc_size, in_size)
        # rectified linear unit activation function
        self.relu = nn.ReLU()
    def forward(self, input):
        output = self.ff_2(self.relu(self.ff_1(input)))
        return output
    
# transformer attention head with in_size equal to transformer input size and
# hidden size equal to in_size/h (number of attention heads) 
class transformer_att(nn.Module):
    def __init__(self, in_size, h):
        super(transformer_att, self).__init__()
        self.att_size = int(in_size/h)
        # create the Q, K and V parts of the attention head
        self.Q = nn.Linear(in_size, in_size, bias = False)
        self.K = nn.Linear(in_size, in_size, bias = False)
        self.V = nn.Linear(in_size, in_size, bias = False)
        # att block linear output layer
        self.fc = nn.Linear(in_size, in_size, bias = False)
        self.softmax = nn.Softmax(dim = -1)
        self.h = h
        self.dropout = nn.Dropout(0.1)
    # in encoding q=k=v . In decoding, the second attention layer, k=v (encoder 
    # output) and q is the decoder intermediate output
    def forward(self, q, k, v, mask = None):
        # scaling factor for the attention scores
        scale = torch.sqrt(torch.FloatTensor([self.h])).item() 
        batch_size = q.size(0)
        # apply the linear transform to the query, key and value and reshape 
        # the result into h attention heads
        Q = self.Q(q).view(batch_size, -1, self.h, 
                           self.att_size).transpose(1,2)
        K = self.K(k).view(batch_size, -1, self.h, 
                           self.att_size).transpose(1,2)
        V = self.V(v).view(batch_size, -1, self.h, 
                           self.att_size).transpose(1,2)
        # multiply and scale q and v to get the attention scores
        self.alpha = torch.matmul(Q,K.transpose(-2,-1))/scale
        # apply mask if needed
        if mask is not None:
            mask = mask.unsqueeze(1)
            self.alpha = self.alpha.masked_fill(mask == 0, -1e9)
        # apply softmax to the attention scores
        self.alpha = self.softmax(self.alpha)
        # apply the att scores to the value v
        att_applied = torch.matmul(self.dropout(self.alpha), V)    
        # reshape the attention heads and finally pass through a fully 
        # connected layer
        att = att_applied.transpose(1, 2).reshape(batch_size, -1, 
                                                  self.att_size * self.h)
        output = self.fc(att)   
        return output

# the transformer encoder layer for stacking multiple encoder cells. 
class transformer_encoder(nn.Module):
    def __init__(self, in_size, fc_size, n_layers, h):
        super(transformer_encoder, self).__init__()
        # create one or more multi-head attention layers
        self.transformers = nn.ModuleList()
        self.dropout = nn.Dropout(0.1)
        for x in range(n_layers):
            self.transformers.append(transformer_encoder_cell(in_size, fc_size, 
                                                              h
                                                              )
                                     )
    def forward(self, input, mask = None):
        # apply the (stacked) transformer
        for tf in self.transformers:
            input = tf(self.dropout(input), mask)
        return(input)

# the transformer decoder layer for stacking multiple decoder cells. 
class transformer_decoder(nn.Module):
    def __init__(self, in_size, fc_size, n_layers, h):
        super(transformer_decoder, self).__init__()
        # create one or more multi-head attention layers
        self.transformers = nn.ModuleList()
        self.dropout = nn.Dropout(0.1)
        for x in range(n_layers):
            self.transformers.append(transformer_decoder_cell(in_size, 
                                                              fc_size, h
                                                              )
                                     )
    def forward(self, input, dec_mask = None, enc_mask = None, 
                enc_input = None):
        # apply the (stacked) transformer
        for tf in self.transformers:
            input = tf(self.dropout(input), dec_mask, enc_mask, enc_input)
        return(input)

# super class with some functions that are useful for Transformers 
class transformer(nn.Module):
    def __init__(self):
        super(transformer, self).__init__()
        pass    

    # function to create the positional embeddings
    def pos_embedding(self, sent_len, d_model):
        pos_emb = torch.zeros(int(sent_len), d_model)
        for x in range(0, sent_len):
            for y in range (0, d_model, 2):
                pos_emb[x, y] = torch.sin(torch.Tensor([x / (10000 ** ((2 * y) / d_model))]))
                pos_emb[x, y + 1] = torch.cos(torch.Tensor([x / (10000 ** ((2 * (y + 1)) / d_model))]))
        if self.is_cuda == True:
            pos_emb = pos_emb.cuda()
        return pos_emb
    
    # create the encoder mask, which masks only the padding indices 
    def create_enc_mask(self, input):
        return (input != 0).unsqueeze(1)
    
    # create the decoder mask, which masks the padding and for each timestep
    # all future timesteps
    def create_dec_mask(self, input):
        seq_len = input.size(1)
        # create a mask which masks the padding indices
        mask = (input != 0).unsqueeze(1).byte()
        # create a mask which masks for each time-step the future time-steps
        triu = (np.triu(np.ones([1, seq_len, seq_len]), k = 1) == 0).astype('uint8')
        # combine the two masks
        if self.is_cuda == True:
            dtype = torch.cuda.ByteTensor
        else:
            dtype = torch.ByteTensor
        return dtype(triu) & dtype(mask)
            
    # Function used for training a transformer encoder-decoder, use 
    # instead of forward
    def encoder_decoder_train(self, enc_input, dec_input):
        # create the targets for the loss function (decoder input, shifted to 
        # the left, padded with a zero)
        targs = torch.nn.functional.pad(dec_input[:, 1:], [0, 1]).long()

        # create the encoder mask which is 0 where the input is padded along 
        # the time dimension
        e_mask = self.create_enc_mask(enc_input)

        # retrieve embeddings for the sentence and scale the embeddings 
        # importance relative to the positional embeddings
        e_emb = self.embed(enc_input.long()) * \
                np.sqrt(self.embed.embedding_dim)

        # apply the (stacked) encoder transformer
        encoded = self.TF_enc(e_emb + self.pos_emb[:enc_input.size(1), :], 
                              mask = e_mask)  

        # create the decoder mask for padding, which also prevents the decoder 
        # from looking into the future.
        d_mask = self.create_dec_mask(dec_input)

        # retrieve embeddings for the sentence and scale the embeddings 
        # importance relative to the positional embeddings
        d_emb = self.embed(dec_input.long()) * \
                np.sqrt(self.embed.embedding_dim)

        # apply the (stacked) decoder transformer
        decoded = self.TF_dec(d_emb + self.pos_emb[:dec_input.size(1), :],
                              dec_mask = d_mask, enc_mask = e_mask, 
                              enc_input = encoded
                              )

        return decoded, targs

    # Function used for a transformer with an encoder only without additional 
    # context input e.g. for next word prediction. Use instead of forward
    def encoder_train(self, enc_input):
        # create the targets (decoder input, shifted to the left, padded with 
        #a zero)
        targs = torch.nn.functional.pad(enc_input[:,1:], [0,1]).long()     
        # create a mask for the padding, which also prevents the encoder from 
        # looking into the future.
        e_mask = self.create_dec_mask(enc_input)       
        # retrieve embeddings for the sentence and scale the embeddings 
        # importance relative to the pos embeddings
        e_emb = self.embed(enc_input.long()) *\
                np.sqrt(self.embed.embedding_dim)
        
        # apply the (stacked) encoder transformer
        encoded = self.TF_enc(e_emb + self.pos_emb[:enc_input.size(1), :], 
                              mask = e_mask)
        
        return encoded, targs 
    
    # training function for caption2image encoders. Use instead of forward
    def cap2im_train(self, enc_input):
        mask = self.create_dec_mask(enc_input)
        # for encoders with an embedding layer, convert input to longtensor
        if hasattr(self, 'embed'):
            enc_input = self.embed(enc_input.long()) 
        # scale the inputs importance relative to the pos embeddings
        enc_input = enc_input * np.sqrt(self.embed.embedding_dim) 
        encoded = self.TF_enc(enc_input + self.pos_emb[:enc_input.size(1), :],
                              mask = mask)
        
        return encoded
    
    # function to generate translations from an encoded sentence. if 
    # translations are available they can be used as targets for evaluating 
    # but also works for unknown sentences. Works only if batch size is set to 
    # 1 in the test loop.
    def translate(self, enc_input, dec_input = None, max_len = 64, 
                             beam_width = 1):
        if self.is_cuda == True:
            dtype = torch.cuda.FloatTensor
        else:
            dtype = torch.FloatTensor
        # create the targets if dec_input is given, decoder input is only used
        # to create targets (e.g. for calculating a loss or comparing 
        # translation to golden standard)
        if not dec_input is None:
            targs = torch.nn.functional.pad(dec_input[:, 1:], 
                                            [0, max_len - dec_input[:, 1:].size()[-1]]).long()
        else:
            targs = dtype([0])
        # create the encoder mask which is 0 where the input is padded along 
        # the time dimension
        e_mask = self.create_enc_mask(enc_input)
        # retrieve embeddings for the sentence and scale the embeddings 
        # importance relative to the pos embeddings
        emb = self.embed(enc_input.long()) * np.sqrt(self.embed.embedding_dim)
        # apply the (stacked) encoder transformer
        encoded = self.TF_enc(emb + self.pos_emb[:enc_input.size(1), :], 
                              mask = e_mask)  
        # set the decoder input to the <bos> token (i.e. predict the 
        # tranlation using only the encoder output and a <bos> token)      
        dec_input = enc_input[:,0:1]
        # create the initial candidate consisting of <bos> and 0
        candidates = [[dec_input, 0]]
        # perform beam search
        for x in range(1, max_len + 1):
            candidates = self.beam_search(candidates, encoded, e_mask, 
                                          beam_width, dtype)
        # create label predictions for the top candidate (e.g. to calculate a 
        # cross entropy loss)
        d_mask = self.create_dec_mask(candidates[0][0][:, :-1])
        # convert data to embeddings
        emb = self.embed(candidates[0][0].long()) * \
              np.sqrt(self.embed.embedding_dim)
        # pass the data through the decoder
        decoded = self.TF_dec(emb[:, :-1, :] + 
                              self.pos_emb[:candidates[0][0].size(1), :], 
                              dec_mask = d_mask, enc_mask = e_mask, 
                              enc_input = encoded
                              )
        top_pred = self.linear(decoded)
        return candidates, top_pred, targs   
    
    # beam search algorithm for finding translation candidates
    def beam_search(self, candidates, encoded, e_mask, beam_width, dtype):
        new_candidates = []
        for input, score in candidates:
            # create the decoder mask
            d_mask = self.create_dec_mask(input)
            # convert data to embeddings
            emb = self.embed(input.long()) * np.sqrt(self.embed.embedding_dim)
            # pass the data through the decoder
            decoded = self.TF_dec(emb + self.pos_emb[:input.size(1), :], 
                                  dec_mask = d_mask, enc_mask = e_mask, 
                                  enc_input = encoded)
            # pass the data through the prediction layer
            pred = torch.nn.functional.softmax(self.linear(decoded), 
                                               dim = -1).squeeze(0)
            # get the top k predictions for the next word, calculate the new
            # probability of the sentence and append to the list of new 
            # candidates
            for value, idx in zip(*pred[-1, :].cpu().data.topk(beam_width)):
                new_candidates.append([torch.cat([input, dtype([idx.item()]).unsqueeze(0)], 1), -np.log(value) + score])
        sorted(new_candidates, key=lambda s: s[1])
        return new_candidates[:beam_width]       


###############################################################################

# this implementation is taken from zalando research on github. It also has
# Exponential moving average updating.
class VQ_EMA_zalando(nn.Module):
    def __init__(self, num_emb, emb_dim):
        super(VQ_EMA_zalando, self).__init__()
        self.num_emb = num_emb
        # create the embedding layer and initialise with uniform distribution
        self.embed = nn.Embedding(num_emb, emb_dim)
        self.embed.weight.data.uniform_(-1/num_emb, 1/num_emb)

        self.register_buffer('_ema_cluster_size', torch.zeros(num_emb))
        # set the exponential moving average 
        self._ema_w = nn.Parameter(self.embed.weight.clone())
        self._ema_w.requires_grad_(False)
        self._decay = 0.99
        self._epsilon = 1e-5
    # expected input dimensions is Batch/Signal-length/Channels
    def forward(self, input):
        input_shape = input.shape
        # flatten accross B/SL dims
        flat_input = input.detach().view(-1, input_shape[-1])

        # matrix of distances between inputs and embeddings
        distances = (torch.sum(flat_input**2, dim = 1, keepdim = True)
                    + torch.sum(self.embed.weight**2, dim = 1)
                    - 2.0 * torch.matmul(flat_input, self.embed.weight.t()))

        # Retrieve for each input the index of the closest embedding
        encoding_indices = torch.argmin(distances, dim = -1)
        encodings = nn.functional.one_hot(encoding_indices, self.num_emb).float()
        # Quantize and unflatten
        quantized = torch.matmul(encodings,
                                 self.embed.weight).view(input_shape)
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)
    
            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = ((self._ema_cluster_size + self._epsilon)
                                      / (n + self.num_emb * self._epsilon) * n
                                      )
    
            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + 
                                       (1 - self._decay) * dw)
            
            self.embed.weight = nn.Parameter(self._ema_w / 
                                             self._ema_cluster_size.unsqueeze(1)
                                             )
        # Loss
        e_latent_loss = torch.nn.functional.mse_loss(input, quantized.detach())
        #q_latent_loss = torch.nn.functional.mse_loss(quantized, input.detach())
        loss = 0.25 * e_latent_loss

        quantized = input + (quantized - input).detach()
        avg_probs = torch.mean(encodings, dim = 0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs 
                                                                + 1e-10
                                                                )
                                          )
                               )
        return quantized, loss