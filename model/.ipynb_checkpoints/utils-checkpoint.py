# Simple Toolbox
#%matplotlib inline
from __future__ import print_function, division
import matplotlib
import matplotlib.pyplot as plt
import os
from skimage import io, transform
import numpy as np
import cv2 as cv
import sys
import ipyvolume as ipv
import ipywidgets
import pandas as pd
import datetime
import time

# Advance Toolbox
import torch
import torch.nn as nn
import torch.nn.functional as F  # useful stateless functions
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

from torch.utils.data import Dataset, DataLoader
from torch.utils.data import sampler

import torchvision
import torchvision.datasets as dset
import torchvision.transforms as T
from torchvision import utils, transforms



class FemurDataset(Dataset):            # build torch.utils.data.Dataset class
    """ 3D intensity volume and AP & Lateral datasets 
    Args:
            csv_file (string): Path to the excel file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
    """
    
    def __init__(self, csv_file, root_dir, transform=None):       # load RAW data such as image and attribute
        self.files_frame = pd.read_excel(csv_file)
        self.root_dir = root_dir
        self.transform = transform               # use with torchvision.transform
        
    def __len__(self):                           # for "len(variable) to return shape of RAW data
        return len(self.files_frame)
    
    def __getitem__(self, idx):                  # for enable slicing data like "variable[idx]"
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        target = None                                    #np.load(os.path.join(self.root_dir, self.files_frame.iloc[idx,1]))  # Intensity[n,0] or Mask[n,1]
        view1 = np.load(os.path.join(self.root_dir, self.files_frame.iloc[idx,2]))
        view2 = np.load(os.path.join(self.root_dir, self.files_frame.iloc[idx,3]))
        drr1_name = self.files_frame.iloc[idx,2]
        drr2_name = self.files_frame.iloc[idx,3]
        sample = {'Target':target , 'view1':view1 , 'view2':view2 , 'Output':None ,
                  'drr1_name':drr1_name , 'drr2_name':drr2_name }
        
        if self.transform:
            sample = self.transform(sample)

        return sample
    
class NormalizeSample(object):
    """
    Convert N-dimension input integer to floating number
    INPUT: numpy array with orbitary date type
    """
    def __call__(self, sample):
        target, view1, view2 = sample['Target'], sample['view1'], sample['view2']
        target = np.zeros((256,256,256))
        view1 = np.float32(view1)
        view2 = np.float32(view2)
        return { 'Target':target , 'view1':view1 , 'view2':view2 }

class ToTensor(object):
    """ Decoding an auxiliary claas for 3DReconNet-AC and FracReconNet
        Convert ndarrays in sample to Tensors = [N,D,H,W] (Correct 4D-Tensor) which have values={0,1,2}
        Input:  [0] = background
                [1,2,3,...,10] = fragment of femur bone
                [20] = pelvic bone
                [30] = soft-tissue
                [40] = fracture
                [50] = comminute
        Output: Positive non-zeros class in same channel which have label = 0,1,2  except pelvic volume
        Output classes = {0==background  1==bone class  2==auxiliary class}
        ToTensor is deployed with classes: FemurDataSet and NormalizeSample
    """
    def __call__(self, sample):        # callable classes
        #print('### ToTensor ###')
        target, view1, view2 = sample['Target'], sample['view1'], sample['view2']   # target = [D,W,-H]
        
        classes = np.unique(target)
        #print('    target = {}   {}   {}'.format(target.shape,type(target), target.dtype))
        #print('    classes = {}'.format(classes))
        if view1.ndim == 2:               # set image to get proper tensor's image format
            view1 = view1[np.newaxis,...]    # 1-view
            view2 = view2[np.newaxis,...]  # 2-view
        
        # create 3D np.array:  first-dim<0.5 for background,  first-dim>0.5 for foreground
        t1 = time.time()
        target2 = np.zeros(target.shape)    # [D,W,-H]
        target2[ target==40 ] = 2
        target2[ (target>0)*(target<10) ] = 1
        
        # transpose: [D,W,-H] to [D,H,W]
        target2 = np.flip(target2.transpose(0,2,1), axis=1).copy()
        
        return {'Target': torch.from_numpy(target2).int().squeeze(),
                'view1': torch.from_numpy(view1).float(),
                'view2':torch.from_numpy(view2).float()}
                # 'drr1_name':sample['drr1_name'], 'drr2_name':sample['drr2_name']}


class ToTensor2(object):
    """ A conventional ToTensor for 3DReconNet
        Convert ndarrays in sample to Tensors = [N,D,H,W] (Correct 4D-Tensor) which have values={0,1}
        Input:  [0] = background
                [1,2,3,...,10] = fragment of femur bone
                [20] = pelvic bone
                [30] = soft-tissue
                [40] = fracture
                [50] = comminute
        Output: Positive non-zeros class in same channel which have label = 0,1  except pelvic volume
        Output classes = {0==background+fracture  1==femurMask}
        ToTensor is deployed with classes: FemurDataSet and NormalizeSample
    """
    def __call__(self, sample):        # callable classes
        #print('### ToTensor2 ###')
        target, view1, view2 = sample['Target'], sample['view1'], sample['view2']   # target = [D,W,-H]
        classes = np.unique(target)
        #print('    target = {}   {}   {}'.format(target.shape,type(target), target.dtype))
        #print('    classes = {}'.format(classes))
        if view1.ndim == 2:               # set image to get proper tensor's image format
            view1 = view1[np.newaxis,...]    # 1-view
            view2 = view2[np.newaxis,...]  # 2-view
        
        # create 3D np.array:  first-dim<0.5 for background,  first-dim>0.5 for foreground
        t1 = time.time()
        target2 = np.zeros(target.shape)    # [D,W,-H]
        target2[ (target>0)*(target<10) ] = 1
        
        # transpose: [D,W,-H] to [D,H,W]
        target2 = np.flip(target2.transpose(0,2,1), axis=1).copy()
        t2 = time.time()
        #print('   target2 = {}   {}   {}'.format(target2.shape, type(target2), target2.dtype))
        #print('   view1 = {}   {}   {}'.format(view1.shape, type(view1), view1.dtype))
        #print('   view2 = {}   {}   {}'.format(view2.shape, type(view2), view2.dtype))
        #print('   Time = {} sec.'.format(t2-t1))
        
        return {'Target': torch.from_numpy(target2).int().squeeze(),
                'view1': torch.from_numpy(view1).float(),
                'view2':torch.from_numpy(view2).float() }

    
# Helper function to visualize sample i-th
def show_sample(sampleN, view='all', showOutput=True, detail=False):
    """
    ###########################################################
    # sampleN: {'AP':np.array(2D AP-view Radiograph) ,        #
    #           'LAT':np.array(2D LAT-view radiograph) ,      #
    #           'Target':np.array(3D intensity voxel) ,       #
    #           'Output':np.array(3D output from Network) }   #
    #                                                         #
    # view:   'all' = view both Volume and AP & LAT view      #
    #         'volume' = only view Volume                     #
    #         'plain' = only view AP & LAT plain films        #
    #                                                         #
    # showOutput: True >> show 3D output from Network         #
    #             False >> Don't show 3D output from Network  #
    #                                                         #
    # detail: True = print data.shape                         #
    #         False = don't print                             #
    #                                                         #
    ###########################################################
    """
    
    target = sampleN['Target']
    ap = sampleN['AP']
    lat = sampleN['LAT']
    if sampleN['Output'] is not None:
        output = sampleN['Output']
    
    if detail==True:
        print('   AP = ' + str(type(ap)) + '   Shape = ' + str(ap.shape) + '   dtype = ' + str(ap.dtype))
        print('   LAT = ' + str(type(lat)) + '   Shape = ' + str(lat.shape) + '   dtype = ' + str(lat.dtype))
        print('   Target = ' + str(type(target)) + '   Shape = ' + str(target.shape) + '   dtype = ' + str(target.dtype))
        if showOutput==True and 'output' in locals():
            print('   Output = ' + str(type(output)) + '   Shape = ' + str(output.shape) + '   dtype = ' + str(output.dtype))
        
    if view=='all' or view=='plain':          # show 2D-images
        fig, ax = plt.subplots(1,2)
        fig.set_size_inches(12,12)
        fig.tight_layout()   # fig.tight_layout(pad=10)
        ax[0].imshow(ap, cmap='gray')
        ax[0].set_title('DRR AP Projection')
        ax[1].imshow(lat, cmap='gray')
        ax[1].set_title('DRR LAT Projection')
        
    if view=='all' or view=='volume':         # show 3D-instensity
        ipv.figure()
        ipv.volshow(target, level=[0.1, 0.5, 0.67], opacity=[0.01, 0.1, 0.1],)
        ipv.view(270, 90)      # ipv.view(around_currnt_Y-,around_current_X)
        ipv.show()
        time.sleep(2)
        if showOutput==True:# and 'output' in locals():
            ipv.figure()
            ipv.volshow(output, level=[0.1, 0.5, 0.67], opacity=[0.01, 0.1, 0.1],)
            ipv.view(270, 90)
            ipv.show()
            
            
def show_sample_batch(sample_batched):
#   Show image with landmarks for a batch of samples.
#   Input: sample_batched = a single batched sample
#          (in a single batched have 'batch_size' samples)
    
    voxel_batch = sample_batched['Volume']
    drr1_batch = sample_batched['AP']  
    drr2_batch = sample_batched['LAT']
    
    batch_size = len(voxel_batch)
    im_size = drr1_batch.size(2)
    grid_border_size = 2
    grid = utils.make_grid(drr1_batch)            # torchvision.utils class
    grid_numpy = grid.numpy().transpose((1,2,0))
    print('grid.shape = {}'.format(grid.size()))
    print('grid_numpy.shape = {}'.format(grid_numpy.shape))
    plt.imshow(grid_numpy, cmap='gray')
    
    #return grid_numpy