from __future__ import print_function, division
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import ipywidgets as widgets
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.autograd import Function, Variable
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import sampler
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as T
from torchvision import utils, transforms

from model.utils import show_sample, FemurDataset, NormalizeSample, ToTensor, ToTensor2
from model.model_utils import single_dense_block, encode_block, decode_block, fully_conv2d, conv2d_block, conv3d_block, convtranspose2d_block, convtranspose3d_block # basic module
from model.model_utils import linear_initialize_sequence, conv2d_initialize_sequence, conv3d_initialize_sequence, convtranspose2d_initialize_sequence, convtranspose3d_initialize_sequence  # weight initialized
from model.model_utils import LinearLayer, LinearLayer_bn, MyEncoder, MyDenseLayer, MyDecoder, Fusion2d, MyDecoder3d, FullyConv3d, UpConv2d, UpConv3d, FinalClassify3d, UpsampleHW

class fracReconNet(nn.Module):   # Signle GPU  Convolute without Dilation
    def __init__(self, in_f, en_sz, de_sz, de3d_sz, final_sz, *args, **kwargs):
        super(fracReconNet, self).__init__()
        assert len(en_sz)== len(de_sz) , 'These input {en_sz} and {de_sz} can not build FracReconNet'
        
        # Prepare feature resolution
        input_res = 256
        self.res1 = [ round(input_res*0.5**i) for i in range(len(en_sz)) ]
        self.res2 = [ round(input_res*0.5**i) for i in range(len(en_sz)) ]
        self.res3 = [ round(input_res*0.5**i) for i in range(len(en_sz)) ]
        self.res1.reverse()
        self.res2.pop(-1)
        self.res2.reverse()
        self.res3.pop(-1)
        self.res3.pop(-1)
        self.res3.reverse()
        self.res4 = self.res1.copy()
        self.res4.reverse()
        self.res4.pop(0)
        
        self.en_sz = en_sz
        #print('en_sz = {}\n'.format(self.en_sz))
        
        cat1 = [ x[0]+x[1]*x[2] for x in en_sz]
        cat1.reverse()
        cat1.pop(0)
        cat2 = [ x[-1] for x in de_sz ]
        cat = [ x1+x2 for x1,x2 in zip(cat1,cat2) ]
        cat = [en_sz[-1][0]+en_sz[-1][1]*en_sz[-1][2], *cat]
        
        self.de_sz = [ [x1,*x2] for x1,x2 in zip(cat,de_sz)]
        self.de3d_sz = de3d_sz
        self.final_sz = final_sz

        ############################################
        # Class Layer description : MyDenseLayer >> MyDecoder >> UpConv2d >> MyDecoder3d >> UpConv3d >> nn.Conv3d
        # Starting layer
        self.first_layer1 = nn.Conv2d(in_f, en_sz[0][0], kernel_size=3, stride=1, padding=1)
        self.first_layer2 = nn.Conv2d(in_f, en_sz[0][0], kernel_size=3, stride=1, padding=1)
        
        # Dense Connection Encode layer
        self.dense_layer1 = nn.ModuleList([ MyDenseLayer(en_sz[i][0],en_sz[i][1],en_sz[i][2], 
                                                         kernel_size=3, stride=1, padding=1) 
                                           for i in range(len(en_sz))] )
        self.dense_layer2 = nn.ModuleList([ MyDenseLayer(en_sz[i][0],en_sz[i][1],en_sz[i][2], 
                                                         kernel_size=3, stride=1, padding=1) 
                                           for i in range(len(en_sz))] )
        # Pooling2D
        self.pool2d = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, return_indices=True)
        #self.adaptpool2d = nn.ModuleList([ nn.AdaptiveMaxPool2d((x,x)) for x in self.res4])
        
        # Decode layer for 2D
        self.decode_layer1 = nn.ModuleList([ MyDecoder(self.de_sz[i], self.res1[i], kernel_size=3, stride=1, padding=1) for i in range(len(self.de_sz))] )
        self.decode_layer2 = nn.ModuleList([ MyDecoder(self.de_sz[i], self.res1[i], kernel_size=3, stride=1, padding=1) for i in range(len(self.de_sz))] )

        # UpConv2d for Decoder
        self.upconv2d1 = nn.ModuleList( [ UpConv2d([self.de_sz[i][-1],self.de_sz[i][-1]], self.res1[i], kernel_size=2, stride=2, padding=0) for i in range(len(self.de_sz)) ] )
        self.upconv2d2 = nn.ModuleList( [ UpConv2d([self.de_sz[i][-1],self.de_sz[i][-1]], self.res1[i], kernel_size=2, stride=2, padding=0) for i in range(len(self.de_sz)) ] )
        
        # for 3D connection pyramid for Fusion
        self.final_layer = nn.ModuleList([ MyDecoder3d(self.final_sz[i], self.res1[i], kernel_size=3, stride=1, padding=1) for i in range(len(self.final_sz))])
        
        # UpConv3d for Fusion
        self.upconv3d = nn.ModuleList( [ UpConv3d([self.final_sz[i][-1], 3], self.res2[i], kernel_size=2, stride=2, padding=0) for i in range(len(self.de_sz)-1) ] )

        # Final classification by Conv3d(1x1x1)
        self.final_layer2 = nn.Sequential(nn.Conv3d(self.final_sz[-1][-1], 3, kernel_size=1, stride=1, padding=0), nn.Softmax(dim=1) )  # Change output channel to {2,3} upto ToTensor version
        
    def forward(self, x1, x2):
        #print('\n--- Encode loop ---')
        x1 = self.first_layer1(x1)
        x2 = self.first_layer2(x2)
        encode_trace1 = []
        encode_trace2 = []
        i = 0
        for layer1 , layer2 in zip(self.dense_layer1 , self.dense_layer2):   # for loop on each nn.Sequential layer
            x1 = layer1(x1)         # layer = each dense blocks
            x2 = layer2(x2)         # layer = each dense blocks
            encode_trace1.append(x1)    # trace x for concatenation with decode layer
            encode_trace2.append(x2)    # trace x for concatenation with decode layer
            if i != len(self.dense_layer1)-1:
                x1 , _ = self.pool2d(x1)        # for MaxPool2d or AvgPool2d
                x2 , _ = self.pool2d(x2)        # for MaxPool2d or AvgPool2d
            i += 1
        
        #print('\n--- Decode loop and Fusion loop ---')
        res = [ round(x1[-1]/x2) for x1,x2 in zip(self.de_sz,self.res1) ]
        #print('inside loop res = {}'.format(res))
        i = 0
        for layer1,layer2,fusionlayer in zip(self.decode_layer1, self.decode_layer2, self.final_layer):
            #print('Level: {}'.format(i))
            if i==0:
                x1 = encode_trace1[len(self.dense_layer1)-1-i]
                x2 = encode_trace2[len(self.dense_layer2)-1-i]
            else:   # encoder and decoder fusion
                x1 = torch.cat( (encode_trace1[len(self.dense_layer1)-1-i],x1), dim=1)
                x2 = torch.cat( (encode_trace2[len(self.dense_layer1)-1-i],x2), dim=1)
            N,C,H,W = x1.size()
            x1 = layer1(x1)
            x2 = layer2(x2)
            
            # Covert to 3D volumes or 5D-Tensor
            X1 = x1.view(-1,res[i],H,H,W)
            X2 = x2.view(-1,res[i],H,H,W).transpose(4,2).flip(4)
                
            # 3D volume fusion (concatenation)
            if i==0:
                X = torch.cat((X1,X2), dim=1)
            else:
                X = torch.cat((X1,X2,X), dim=1)
            
            X = fusionlayer(X)
            if i!=len(self.decode_layer1)-1:
                x1 = self.upconv2d1[i](x1)
                x2 = self.upconv2d1[i](x2)
                X = self.upconv3d[i](X)
            i+=1
        
        #print('\n--- Final classify ---')
        X = self.final_layer2(X)

        return X

    
    