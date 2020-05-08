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
        
# class for making multi headed attenders. 
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

# quantization layer is an embedding layer which uses a costum embedding
# function. It takes input activations from the previous layer, and maps them
# to the closes embedding. upon calling backward, the output gradient for this
# layer is directly passed to the layer beneath.
class quantization_layer(nn.Module):
    def __init__(self, num_emb, emb_dim):
        super(quantization_layer, self).__init__()

        self.embed = nn.Parameter(torch.zeros(num_emb, emb_dim))
        torch.nn.init.uniform_(self.embed, -1/1024, 1/1024)
        self.quant_emb = quantization_emb.apply
    def forward(self, input):
        # get the distance and the index of the closest embedding
        dims = input.size()
        embs, one_hot = self.quant_emb(input.reshape(-1, dims[-1]), 
                                     self.embed)
        #embs = embs.view(dims[0], dims[1], -1)
        embeddings = torch.mm(one_hot, self.embed)
        input = input.reshape(-1, dims[-1])
        
        l1 = torch.nn.functional.mse_loss(input.detach(), embeddings)
        l2 = 0.25 * torch.nn.functional.mse_loss(input, embeddings.detach())
        
        embs = input + (embeddings - input).detach()
        return embs.view(dims[0], dims[1], -1), l1 + l2
    
# autograd function for the embedding mapping which also implements the 
# skipping the gradient for this layer. 
class quantization_emb(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight):
        shape = weight.size()
        i_norm = (input**2).sum(1).view(-1, 1)
        w_norm = (weight**2).sum(1).view(1, -1)

        dist = i_norm + w_norm - 2.0 * torch.mm(input,
                                                torch.transpose(weight, 0, 1)) 
        idx = dist.argmin(1)
        #print(inds.float().mean())       
        one_hot = nn.functional.one_hot(idx, shape[0]).float()
        embs = torch.mm(one_hot, weight)
        #dist_err = 1 - torch.mm(input, output.t())
        return embs, one_hot

    @staticmethod
    def backward(ctx, emb_output, dist_output):
        return emb_output, None
    
# residual convolution block with 4 conv layers and 2 residual connections
class res_conv(nn.Module):
    def __init__(self, in_channels, out_channels, ks, stride):
        super(res_conv, self).__init__()   
        # check if the input is being downsampled and if the number of channels
        # increases. 
        self.req_downsample = stride != 1
        # if the number of channels is changed, the residual needs to upscale
        self.req_residual_dims = in_channels != out_channels
        # only works for uneven kernel sizes and even strides.
        pad = int((ks -1)/ 2)
        # the first layer can optionally downsample the time dimesion
        self.conv_1 = nn.Sequential(nn.Conv1d(in_channels, out_channels, 
                                              kernel_size = ks, 
                                              stride = stride, padding = pad),
                                    nn.BatchNorm1d(out_channels)
                                    )
        
        self.conv_2 = nn.Sequential(nn.Conv1d(out_channels, out_channels, 
                                              kernel_size = ks, padding = pad),
                                    nn.BatchNorm1d(out_channels)
                                    )
        
        self.conv_3 = nn.Sequential(nn.Conv1d(out_channels, out_channels, 
                                              kernel_size = ks, padding = pad),
                                    nn.BatchNorm1d(out_channels)
                                    )
        
        self.conv_4 = nn.Sequential(nn.Conv1d(out_channels, out_channels, 
                                              kernel_size = ks, padding = pad),
                                    nn.BatchNorm1d(out_channels)
                                    )
        
        self.relu = nn.functional.relu
        
        self.downsample = nn.MaxPool1d(stride, stride, padding = 0)
        self.residual_dims = nn.Conv1d(in_channels, out_channels, 
                                       kernel_size = 1, stride = 1)
           
    def forward(self, input):
        residual = input
        conv_out_1 = self.conv_2(self.relu(self.conv_1(input)))
        # if input is downsampled or n_channels increases, transform input to
        # match conv_out
        if self.req_downsample:
            # if the input size was uneven, a onesided 0 pad is required.
            if not input.size(-1)%2 == 0:
                residual = nn.functional.pad(residual, [0, 1])
            residual = self.downsample(residual)
        if self.req_residual_dims:
            residual = self.residual_dims(residual)
            
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
        mask = self.create_enc_mask(enc_input)
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
