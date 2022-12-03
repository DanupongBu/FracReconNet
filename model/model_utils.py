import importlib
import io
import logging
import os
import shutil
import sys
import uuid

#import h5py
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
from sklearn.decomposition import PCA

import torch
import torch.nn as nn
import torch.nn.functional as F


norm2d = nn.InstanceNorm2d
#norm2d = nn.BatchNorm2d
norm3d = nn.InstanceNorm3d
#norm3d = nn.BatchNorm3d

def single_dense_block(in_f, out_f, *args, **kwargs):
    """ Creat single encode block (norm+activation+conv)
    Args:   in_f  = input channel (int) for nn.Conv2d 
            out_f = output channel (int) for nn.Conv2d and nn.BatchNorm2d or nn.LayerNorm
            res = feature resolution for nn.LayerNorm
    """
    return nn.Sequential(norm2d(in_f),
                         nn.ReLU(),
                         #nn.LeakyReLU(0.01),
                         nn.Conv2d(in_f, out_f, *args, **kwargs))

def encode_block(in_f, out_f, res, *args, **kwargs):
    """ Creat single encode block (conv+norm+activation)
    Args:   in_f  = input channel (int) for nn.Conv2d 
            out_f = output channel (int) for nn.Conv2d and nn.BatchNorm2d or nn.LayerNorm
            res = feature resolution for nn.LayerNorm
    """
    '''
    print('in_f = {}'.format(in_f))
    print('out_f = {}'.format(out_f))
    print('res = {}'.format(res))
    '''
    return nn.Sequential(nn.Conv2d(in_f, out_f, *args, **kwargs), 
                         norm2d(out_f),
                         #norm2d([out_f,res,res]),
                         nn.ReLU(),
                         #nn.LeakyReLU(0.01)
                        )

def decode_block(in_f, out_f, res, *args, **kwargs):
    """ Creat single decode block (conv+norm+activation)
    Args:   in_f  = input channel (int) for nn.Conv2d 
            out_f = output channel (int) for nn.Conv2d and nn.BatchNorm2d
            res = feature resolution for nn.LayerNorm
    """
    return nn.Sequential(nn.Conv2d(in_f, out_f, *args, **kwargs), 
                         norm2d(out_f),
                         #nn.Dropout2d(p=0.2),
                         #norm2d([out_f,res,res]),
                         nn.ReLU(),
                         #nn.LeakyReLU(0.01)
                        )

def fully_conv2d(in_f, out_f, res, *args, **kwargs):
    """ Obsolete!!! Creat single nxn-convolution (conv + norm + activation)
    Args:   in_f  = input channel (int) for nn.Conv2d 
            out_f = output channel (int) for nn.Conv2d and nn.BatchNorm2d
            res = feature resolution for nn.LayerNorm
    """
    return nn.Sequential(nn.Conv2d(in_f, out_f, *args, **kwargs),
                         norm2d(out_f),
                         #norm2d([out_f,res,res]),
                         nn.ReLU(),
                         #nn.LeakyReLU(0.01)
                        )

def conv2d_block(in_f, out_f, res, *args, **kwargs):
    """ Creat single nxn-convolution (conv + norm + activation)
    Args:   in_f  = input channel (int) for nn.Conv2d 
            out_f = output channel (int) for nn.Conv2d and nn.BatchNorm2d
            res = feature resolution for nn.LayerNorm
    """
    return nn.Sequential(nn.Conv2d(in_f, out_f, *args, **kwargs),
                         norm2d(out_f),
                         #norm2d([out_f,res,res]),
                         nn.ReLU(),
                         #nn.LeakyReLU(0.01)
                        )

def conv3d_block(in_f, out_f, res, *args, **kwargs):
    """ Create single 3D-convolution block (conv3d + norm + activation)
    Args:   in_f = input channel (int) for nn.Conv3d
            out_f = output channel (int) for nn.Conv3D
            res = feature resolution for nn.LayerNorm
    """
    return nn.Sequential(nn.Conv3d(in_f, out_f, *args, **kwargs),
                         norm3d(out_f),
                         #norm3d([out_f,res,res,res]),
                         nn.ReLU(),
                         #nn.LeakyReLU(0.01)
                        )

def convtranspose2d_block(in_f, out_f, res, *args, **kwargs):
    """ Create single 3D-transpose-convolution block (convtranspose3d + norm + activation)
    Args:   in_f = input channel (int) for nn.ConvTranspose3d
            out_f = output channel (int) for nn.ConvTranspose3D
            res = feature resolution for nn.LayerNorm
    """
    return nn.Sequential(nn.ConvTranspose2d(in_f, out_f, *args, **kwargs),
                         norm2d(out_f),
                         nn.ReLU(),
                         #nn.LeakyReLU(0.01)
                        )

def convtranspose3d_block(in_f, out_f, res, *args, **kwargs):
    """ Create single 3D-transpose-convolution block (convtranspose3d + norm + activation)
    Args:   in_f = input channel (int) for nn.ConvTranspose3d
            out_f = output channel (int) for nn.ConvTranspose3D
            res = feature resolution for nn.LayerNorm
    """
    return nn.Sequential(nn.ConvTranspose3d(in_f, out_f, *args, **kwargs),
                         norm3d(out_f),
                         nn.ReLU(),
                         #nn.LeakyReLU(0.01)
                        )


 ##################  Weight Initialization ########################
def linear_initialize_sequence(sequential):
    """ Initialize 2D-convolution parameter
    Args:   nn.Sequential with include nn.Conv2d
    """
    for seq in sequential:
        #print('\nseq = {}'.format(seq))
        for module in seq:
            #print('module = {}'.format(module))
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                nn.init.constant_(module.bias, 0.1)
                #print('weight = {}'.format(module.weight))
                #print('bias = {}'.format(module.bias))

def conv2d_initialize_sequence(sequential):
    """ Initialize 2D-convolution parameter
    Args:   nn.Sequential with include nn.Conv2d
    """
    for seq in sequential:
        #print('\nseq = {}'.format(seq))
        for module in seq:
            #print('module = {}'.format(module))
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.1)
                #print('weight = {}'.format(module.weight))
                #print('bias = {}'.format(module.bias))
                
def conv3d_initialize_sequence(sequential):
    for seq in sequential:
        #print('\nseq = {}'.format(seq))
        for module in seq:
            #print('module = {}'.format(module))
            if isinstance(module, nn.Conv3d):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.1)
                #print('weight = {}'.format(module.weight))
                #print('bias = {}'.format(module.bias))
                
def convtranspose2d_initialize_sequence(sequential):
    for seq in sequential:
        #print('\nseq = {}'.format(seq))
        for module in seq:
            #print('module = {}'.format(module))
            if isinstance(module, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.1)
                #print('weight = {}'.format(module.weight))
                #print('bias = {}'.format(module.bias))

def convtranspose3d_initialize_sequence(sequential):
    for seq in sequential:
        #print('\nseq = {}'.format(seq))
        for module in seq:
            #print('module = {}'.format(module))
            if isinstance(module, nn.ConvTranspose3d):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.1)
                #print('weight = {}'.format(module.weight))
                #print('bias = {}'.format(module.bias))

                
 ################## Class layer ################## 
class LinearLayer(nn.Module):
    """ Create sigle block of linear layer with norm and activatetion 
    """
    def __init__(self, lin_sz, *args, **kwargs):
        super(LinearLayer, self).__init__()
        linear_block = [ single_linear_block(in_f, out_f) 
                        for in_f,out_f in zip(lin_sz,lin_sz[1:])  ]
        self.linear_block = nn.Sequential(*linear_block)
        linear_initialize_sequence(self.linear_block)
    def forward(self, x):
        return self.linear_block(x)
    
class LinearLayer_bn(nn.Module):
    """ Create sigle block of linear layer with norm and activatetion 
    """
    def __init__(self, lin_sz, bn_sz, *args, **kwargs):
        super(LinearLayer_bn, self).__init__()
        linear_bn_block = [ single_linear_bn_block(in_f, out_f, bn_sz) 
                        for in_f,out_f in zip(lin_sz,lin_sz[1:])  ]
        self.linear_bn_block = nn.Sequential( *linear_bn_block )
        linear_initialize_sequence(self.linear_block)
    def forward(self, x):
        return self.linear_bn_block(x)
    
class MyEncoder(nn.Module):
    """ Create a bundle of level for Encoder
    Args:  en_sz = 2D-list [[encode_block1],[encode_block2],[encode_block3],...,[encode_blockN]]
            *args, **kwargs = up on decode_block
    """
    def __init__(self, en_sz, *args, **kwargs):
        super(MyEncoder, self).__init__()
        encode_blocks = [ encode_block(in_f, out_f, *args, **kwargs) 
                         for in_f, out_f in zip(en_sz,en_sz[1:]) ]
        self.encode_blocks = nn.Sequential( *encode_blocks )
        #print(self.encode_blocks,'\n',type(self.encode_blocks))
        conv2d_initialize_sequence(self.encode_blocks)
    def forward(self, x):
        return self.encode_blocks(x)
    
class MyDenseLayer(nn.Module):
    """ Create a bundle of dense layer
    Args:   in_f = input channel 
            k = growth rate of channel
            layer = number of composite layer
            bn_size = bottom neck size
            *args, **kwargs = up on single_dense_block
    """
    def __init__(self, in_f, k, level, *args, **kwargs):
        super(MyDenseLayer, self).__init__()
        dense_blocks = [ single_dense_block(in_f + i*k, k, *args, **kwargs) 
                         for i in range(level) ]
        self.dense_blocks = nn.Sequential(*dense_blocks)
        conv2d_initialize_sequence(self.dense_blocks)
        
    def forward(self, x):
        x_cat = x
        for i , level in enumerate(self.dense_blocks):
            if i == 0:
                x = level(x_cat)
            else:
                x = level(x_cat)
            x_cat = torch.cat((x_cat,x),dim=1)
            #print('   x_cat = {}'.format(x_cat.size()))
            #print('   x = {}'.format(x.size()))
        #print('   # Dense layer output = {} \n'.format(x_cat.size()))
        return x_cat

class MyDecoder(nn.Module):
    """ Create a bundle of level
    Args:  de_sz = 2D-list [[decode_block1],[decode_block2],[decode_block3],...,[decode_blockN]]
            *args, **kwargs = up on decode_block
    """
    def __init__(self, de_sz, *args, **kwargs):
        super(MyDecoder, self).__init__()
        decode_blocks = [ decode_block(in_f, out_f, *args, **kwargs) 
                         for in_f, out_f in zip(de_sz,de_sz[1:]) ]
        self.decode_blocks = nn.Sequential( *decode_blocks )
        #print(self.decode_blocks,'\n',type(self.decode_blocks))
        conv2d_initialize_sequence(self.decode_blocks)
    def forward(self, x):
        return self.decode_blocks(x)
    
class AxialFusion(nn.Module):
    """ Axially fuse view1 and view2 2D_features by nn.Conv2d 
    """
    def __init__(self, de3d_sz, *args, **kwargs):
        super(AxialFusion, self).__init__()
        fusion_blocks = [ conv2d_block(in_f, out_f, *args, **kwargs) 
                          for in_f, out_f in zip(de3d_sz,de3d_sz[1:]) ]
        self.fusion_blocks = nn.Sequential( *fusion_blocks )
        conv2d_initialize_sequence(self.fusion_blocks)
    def forward(self, x):
        return self.fusion_blocks(x)
    
class Fusion2d(nn.Module):
    """ Fusion view1 and view2 features by nn.Conv2d 
    """
    def __init__(self, de_sz, *args, **kwargs):
        super(Fusion2d, self).__init__()
        fusion_blocks = [ decode_block(in_f, out_f, *args, **kwargs) 
                         for in_f, out_f in zip(de_sz,de_sz[1:]) ]
        self.fusion_blocks = nn.Sequential( *fusion_blocks )
        conv2d_initialize_sequence(self.fusion_blocks)
    def forward(self, x):
        return self.fusion_blocks(x)
    
class MyDecoder3d(nn.Module):
    """
    """
    def __init__(self, de3d_sz, *args, **kwargs):
        super(MyDecoder3d, self).__init__()
        #print('\nMyDecoder3d')
        decode3d_blocks = [ conv3d_block(in_f, out_f, *args, **kwargs)
                           for in_f, out_f in zip(de3d_sz,de3d_sz[1:]) ]
        #print(decode3d_blocks,'\n',type(decode3d_blocks))
        self.decode3d_blocks = nn.Sequential( *decode3d_blocks )
    def forward(self, x):
        return self.decode3d_blocks(x)
    
class FullyConv3d(nn.Module):  # same function as MyDeCoder3d
    """ Creater a bundle of level
    Args:   final_sz = 1D-list [final_sz1,final_sz2,...,final_szN]
            res = H and W dimension resolution 
            *args, **kwargs = up on fully_conv2d
    """
    def __init__(self, final_sz, *args, **kwargs):
        super(FullyConv3d, self).__init__()
        conv3d_blocks = [ conv3d_block( in_f, out_f, *args, **kwargs) 
                           for in_f , out_f in zip(final_sz,final_sz[1:])]
        self.conv3d_blocks = nn.Sequential( *conv3d_blocks )
        conv3d_initialize_sequence(self.conv3d_blocks)
    def forward(self, x):
        return self.conv3d_blocks(x)
    
class UpConv2d(nn.Module):
    """ 
    """
    def __init__(self, final_sz, *args, **kwargs):
        super(UpConv2d, self).__init__()
        convtranspose2d_blocks = [ convtranspose2d_block( in_f, out_f, *args, **kwargs) 
                                  for in_f , out_f in zip(final_sz,final_sz[1:])]
        self.convtranspose2d_blocks = nn.Sequential( *convtranspose2d_blocks )
        convtranspose2d_initialize_sequence(self.convtranspose2d_blocks)
    def forward(self, x):
        return self.convtranspose2d_blocks(x)
    
class UpConv3d(nn.Module):
    """ 
    """
    def __init__(self, final_sz, *args, **kwargs):
        super(UpConv3d, self).__init__()
        convtranspose3d_blocks = [ convtranspose3d_block( in_f, out_f, *args, **kwargs) 
                                  for in_f , out_f in zip(final_sz,final_sz[1:])]
        self.convtranspose3d_blocks = nn.Sequential( *convtranspose3d_blocks )
        convtranspose3d_initialize_sequence(self.convtranspose3d_blocks)
        
    def forward(self, x):
        return self.convtranspose3d_blocks(x)
    
class FinalClassify3d(nn.Module):    # New Version
    """ Creater a bundle of level
    Args:   final_sz = 1D-list [final_sz1,final_sz2,...,final_szN]
            res = H and W dimension resolution 
            *args, **kwargs = up on fully_conv2d
    """
    def __init__(self, final_sz, *args, **kwargs):
        super(FinalClassify3d, self).__init__()
        classify3d_blocks = [ conv3d_block( in_f, out_f, *args, **kwargs) 
                           for in_f , out_f in zip(final_sz,final_sz[1:])]
        self.classify3d_blocks = nn.Sequential( *classify3d_blocks )
        
        #conv3d_initialize_sequence(self.classify3d_blocks)
        
    def forward(self, x):
        return self.classify3d_blocks(x)

class UpsampleHW(nn.Module):
    """ Upsample 3D-volume or 5-dimension data only H & W dimensions 
        keeping N, C, D dimension as the same
    Args:   input = 5-dimensional yorch.tensor
            scale_factor = 2, 
            mode = 'Linear' or 'Bilinear'
            align_corners = True or False
    """
    def __init__(self, scale_factor, mode='bilinear', align_corners=True):
        super(UpsampleHW, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        self.upsample = nn.Upsample(scale_factor=self.scale_factor, 
                                    mode=self.mode, 
                                    align_corners=self.align_corners )
    def forward(self, x):
        assert x.dim() == 5 , 'Upsample3d: input should be 5D-torch.tensor'
        N, C, D, H, W = x.size()
        for i in range(C):
            if i==0:
                x_up = self.upsample(x[:,i,:,:,:]).unsqueeze(dim=1)
            else:
                x_up = torch.cat( (x_up,self.upsample(x[:,i,:,:,:]).unsqueeze(dim=1)) , dim=1)
            #print('x_up = {}   {}'.format(x_up.size(), x_up.dtype))
        return x_up
        