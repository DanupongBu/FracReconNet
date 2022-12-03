import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
import datetime
#%matplotlib inline

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

from model.losses import FocalLossMulticlass
from model.utils import show_sample, FemurDataset, NormalizeSample, ToTensor, ToTensor2
from model.matricesOperator import iou

dtype = torch.float32
model = None
optimizer = None
scheduler = None
criterion = None
device = None

def train_mixed(model, trainLoader, valLoader, optimizer, scheduler, criterion, batch_sz, epochs, saved_name, saved_dict, device):
    ''' Mixed precision training Version2 
        Build-in model saver, saved_dict
    '''
    print('Train on: {}'.format(device))
    print('Optimizer = {} \n'.format(optimizer))
    scaler = torch.cuda.amp.GradScaler()
    for e in range(epochs):
        time1 = time.time()
        print('----- Epoch = {} ----- # Learning rate = {:.4e}'.format(e+1, optimizer.param_groups[0]["lr"]))
        train_loss, train_acc, val_loss, val_acc = 0, 0, 0, 0   # reset every epoch
        
        for t, train_sample in enumerate(trainLoader):
            #print('trainLoader = {}'.format(len(trainLoader)))
            #print('batch size = {} \n'.format(train_sample['Target'].size()))
            if train_sample['Target'].size(0)%batch_sz != 0 or t==len(trainLoader)-1:  # exclude inequal batch
                print('Final training accuracy = {:.4f}'.format(acc))
                break
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=True):
                model.train(mode=True)    # put model to training mode
                target = train_sample['Target'].to(device=device, dtype=dtype)
                view1 = train_sample['view1'].to(device=device, dtype=dtype)
                view2 = train_sample['view2'].to(device=device, dtype=dtype)
                
                # calculation of output
                output = model(view1, view2)
                
                # Calculate loss of this sample batch
                loss = criterion(output, target.long())
                train_loss += loss.item()
                
                # Check accuracy
                acc = iou((output[:,1]>=0.5).float(), (target==1).float()) 
                train_acc += acc.detach().item()
                
                if t%(round(len(trainLoader)/11)) == 0:
                    print('Iteration: {}   |   Loss = {:.4f}   |   Accuracy = {:.4f}'.format(t, loss.item(), acc))
                if t==len(trainLoader)-1:
                    print('Final training accuracy = {:.4f}'.format(acc))
                
            scaler.scale(loss).backward()
            torch.cuda.synchronize()
            scaler.step(optimizer)
            scaler.update()
            
        # Append Loss and Accuracy history
        train_loss /= round(len(trainLoader))    # mean
        train_acc /= round(len(trainLoader))     # mean
        saved_dict['train_loss_history'].append(train_loss)
        saved_dict['train_acc_history'].append(train_acc)
        print('Training loss = {:.4f}'.format(train_loss))
        print('Training accuracy = {:.4f}'.format(train_acc))
        print('Max. of [Mean Training accuracy] = {:.4f}'.format(max(saved_dict['train_acc_history'])))
        time2 = time.time()
        print('Duration training time = {} Min.\n'.format((time2-time1)/60))
        
###########################################################################################################################
        
        print('### Validation loop ###')
        model.eval()
        time3 = time.time()
        for t, val_sample in enumerate(valLoader):
            if val_sample['Target'].size(0)%batch_sz%batch_sz != 0 or t==len(valLoader)-1: # exclude inequal batch
                print('Final validation accuracy = {:,.4f}'.format(acc))
                break
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    target = val_sample['Target'].to(device=device, dtype=dtype)
                    view1 = val_sample['view1'].to(device=device, dtype=dtype)
                    view2 = val_sample['view2'].to(device=device, dtype=dtype)
                    output = model(view1,view2)
                    
                    loss = criterion(output ,target.long())         # for ToTensor6-9
                    val_loss += loss.item()
                    
                    acc = iou((output[:,1]>=0.5).float(), (target==1).float())
                    val_acc += acc.detach().item()
                    
                    if t%(round(len(valLoader)/11)) == 0:
                        print('Iteration: {}   |   Loss = {:.4f}   |   Accuracy = {:.4f}'.format(t, loss.item(), acc))
                    if t==len(valLoader)-1:
                        print('Final validation accuracy = {:,.4f}'.format(acc))
        
        val_loss /= round(len(valLoader))    # mean
        val_acc /= round(len(valLoader))     # mean
        saved_dict['val_loss_history'].append(val_loss)
        saved_dict['val_acc_history'].append(val_acc)
        saved_dict['timestamp'] = str(datetime.datetime.now())     # update timestamp
        torch.cuda.synchronize()
        scheduler.step(val_acc)
        print('# Validation loss = {:.4f}'.format(val_loss))
        print('Validation accuracy = {:.4f}'.format(val_acc))
        print('Max. of [Mean Validation accuracy] = {:.4f}'.format(max(saved_dict['val_acc_history'])))
        time4 = time.time()
        print('Duration validation time = {} Min. \n'.format((time4-time3)/60))
        
        ### Save the best trained model and training history ###
        if val_acc >= max(saved_dict['val_acc_history']):   # save the best trained
            print(' *** Update the best model state dict at epoch: {}  at time: {} ***'.format(e+1, str(datetime.datetime.now())))
            saved_dict['model_state_dict'] = model.module.state_dict()    # update the best model
        else:   # just save training history and keep the best trained model
            print('\n *** Update training history and the last model state dict at epoch = {}  at time: {} ***\n'.format(e+1, str(datetime.datetime.now())))
            # don't update any params in the trained model
            
        ### End of training ###
        ### Save trained parameters at the last epoch and training history, keeping the best trained model   
        saved_dict['optimizer_state_dict'] = optimizer.state_dict()
        saved_dict['scheduler_state_dict'] = scheduler.state_dict()
        torch.save(saved_dict, saved_name)
        print(' *** Saved End ***\n')
        print('-'*120,'\n\n')
    
    return saved_dict