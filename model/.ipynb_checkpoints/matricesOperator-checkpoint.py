import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import skimage
from skimage import io, transform, measure
from sklearn.neighbors import NearestNeighbors
import scipy
from scipy.spatial.distance import directed_hausdorff
import time

#%matplotlib inline

# Cautions: these function expect binary input (0 or 1) as any dtype on both CPU and CUDA 

############### overlap_based_metrices ###############
def overlap_based_metrices(a,b, sum_of_matrix=False):
    """ Combine IoU, TP, FP, NP, NF
        Args:   a = output {torch.Tensors}
                b = target {torch.Tensors}
        Return  dict['IoU','TP','TN','FP','FN']
    """
    return { 'IoU':iou(a, b), 
            'TP':TP(a, b, sum_of_matrix), 
            'FP':FP(a, b, sum_of_matrix), 
            'FN':FN(a, b, sum_of_matrix), 
            'TN':TN(a, b, sum_of_matrix) }

def TP(a,b, sum_of_matrix=False):  # True-Positive    a = output   b = target
    if a.size() != b.size():
        print('a.size not equat to b.size')
        print('a.size = {}'.format(a.size()))
        print('b.size = {}'.format(b.size()))
        return None
    inter = a*(a==b)
    if sum_of_matrix:
        return inter.sum()
    else:
        return inter

def FP(a,b, sum_of_matrix=False):    # a = output   b = target
    if a.size() != b.size():
        print('a.size not equat to b.size')
        print('a.size = {}'.format(a.size()))
        print('b.size = {}'.format(b.size()))
        return None
    #fp = a~b
    fp = a-b
    fp = a*(fp>0)
    if sum_of_matrix:
        return fp.sum()
    else:
        return fp

def TN(a,b, sum_of_matrix=False):
    if a.size() != b.size():
        print('a.size not equat to b.size')
        print('a.size = {}'.format(a.size()))
        print('b.size = {}'.format(b.size()))
        return None
    ones = torch.ones_like(a)
    tn = ones-union(a,b)
    tn = ones*(tn>0)
    if sum_of_matrix:
        return tn.sum()
    else:
        return tn

def FN(a,b, sum_of_matrix=False):
    if a.size() != b.size():
        print('a.size not equat to b.size')
        print('a.size = {}'.format(a.size()))
        print('b.size = {}'.format(b.size()))
        return None
    #fn = b~a
    fn = b-a
    fn = b*(fn>0)
    if sum_of_matrix:
        return fn.sum()
    else:
        return fn

def union(a,b, sum_of_matrix=False):      # Union (FP+TP+FN)
    if a.size() != b.size():
        print('a.size not equat to b.size')
        print('a.size = {}'.format(a.size()))
        print('b.size = {}'.format(b.size()))
        return None
    unite = a+b-TP(a,b)
    if sum_of_matrix:
        return unite.sum()
    else:
        return unite

def iou(a,b):       # (TP)/(FP+TP+FN)
    if a.size() != b.size():
        print('a.size not equat to b.size')
        print('a.size = {}'.format(a.size()))
        print('b.size = {}'.format(b.size()))
        return None
    interSum = TP(a,b).sum().float()
    uniteSum = union(a,b).sum().float()
    iou = (interSum)/(uniteSum+1)
    return iou

def diceCoefficient(a,b):    # dice coefficient = (2TP)/(2TP+FP+FN)
    if a.size() != b.size():
        print('a.size not equat to b.size')
        print('a.size = {}'.format(a.size()))
        print('b.size = {}'.format(b.size()))
        return None
    tp = TP(a,b).sum().float()
    fp = FP(a,b).sum().float()
    fn = FN(a,b).sum().float()
    dice = (2*tp)/(2*tp + fp + fn + 1)
    return dice

def diceLoss(a,b):
    if a.size() != b.size():
        print('a.size not equat to b.size')
        print('a.size = {}'.format(a.size()))
        print('b.size = {}'.format(b.size()))
        return None
    return 1-diceCoefficient(a,b)

def hausdorff_voxel(output, target):
    """ Calculate L1 hausdorff distance from voxel of output and target
        which are extracted boundary by marching cubes algorithm.
        Args:   output and target = numpy array or torch tensor
    """
    occupy_thresh = 50
    if target.sum()>occupy_thresh and output.sum()>=occupy_thresh :
        if isinstance(output,torch.Tensor):
            output = output.detach().cpu().numpy().squeeze()
        else:
            output = output.squeeze()
        if isinstance(target,torch.Tensor):
            target = target.detach().cpu().numpy().squeeze()
        else:
            target = target.squeeze()

        #t1 = time.time()
        vert1, _, _, _ = measure.marching_cubes(output)
        vert2, _, _, _ = measure.marching_cubes(target)
        #print('vert1 = {}   vert2 = {}'.format(vert1.shape, vert2.shape))
        '''
        fig = plt.figure(figsize=(18,10))
        ax = fig.add_subplot(121, projection='3d')
        ax.scatter(vert1[:,0],vert1[:,1],vert1[:,2], c='r')
        ax = fig.add_subplot(122, projection='3d')
        ax.scatter(vert2[:,0],vert2[:,1],vert2[:,2], c='g')
        plt.show()
        '''
        #t2 = time.time()
        h1 = directed_hausdorff(vert1,vert2)[0]
        h2 = directed_hausdorff(vert2,vert1)[0]
        #print('h1 = {}\nh2 = {}'.format(h1,h2))
        #t3 = time.time()
        #print('Marching cubes times = {} sec.'.format(t2-t1))
        #print('Directed Hausdorff times = {} sec.'.format(t3-t2))
        h = np.max((h1,h2))
    elif target.sum()>=occupy_thresh and output.sum()==0 :
        h = -30     # False Negative of fracture
    elif  target.sum()==0 and output.sum()>=occupy_thresh :
        h = -20     # False Positive of fracture
    else:     # target.sum()==0 and output.sum()==occupy_thresh :
        h = 0       # No fracture
    return h


############### surface_distance_measurement ###############
def mesh_surface_nearest_distance(target, output, res, return_vert_dist=False, verbose=True):
    """ Finding mesh surface from voxel and KNN distances between target and output vertices
        Args:       target = ground truth voxel [D,H,W] (np.ndarray)
                    output = mask voxel [D,H,W] (np.ndarray)
                    res = voxel resolution in [mm] (float)
                    
        Return:     dict['distances'] = nearest neighbor euclidean distance
                    dict['indices'] = nearest indices in output corresponding to target
                Note: the surface distance is based on ground truth's vertices
    """
    
    assert target.ndim == output.ndim or target.shape == output.shape, 'Dimensional Error: target({}) and ground truth({}) are not match'.format(target.shape,output.shape)
    if verbose==True:
        print('Finding MeshSurface and K-Nearest Distance')
        print('target = {}   output = {}'.format(target.shape, output.shape))
    
    # Voxel to 3D-point
    target_vert, target_face, _, _ = measure.marching_cubes(target)
    output_vert, output_face, _, _ = measure.marching_cubes(output)
    target_vert[:,1] = -target_vert[:,1]   # inverse y-axis position
    output_vert[:,1] = -output_vert[:,1]   # inverse y-axis position
    if verbose==True:
        print('target_verbose = {}   output_verbose = {}'.format(target_vert.shape, output_vert.shape))
    
    # Finding nesrestneighbors
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(output_vert)
    distances, indices = nbrs.kneighbors(target_vert)
    distances = res*distances   # convert voxel-distance into mm. unit
    if verbose==True:
        print('Result:  distances = {}'.format(distances[:,0].shape))
        print('         indices = {}\n'.format(indices.shape))
    
    if return_vert_dist==True:
        return {'target_vert':target_vert, 'distances':distances, 'target_face':target_face}
    else:
        return {'distances':distances}    

def surface_distance_measurement(target, output, res, return_vert_dist=False, verbose=True):
    ''' Recall mesh_surface_nearest_distance function then calculate surface distance metrices: ASD,ASSD,RMSD,HD
        Args:       target = ground truth voxel [D,H,W] (np.ndarray)
                    output = mask voxel [D,H,W] (np.ndarray)
                    res = voxel resolution in [mm] (float)
                    
        Return:     dict['distances'] = nearest neighbor euclidean distance
                    dict['indices'] = nearest indices in output corresponding to target
                Note: the surface distance is based on ground truth's vertices
    '''
    surface_dist_gt = mesh_surface_nearest_distance(target, output, res, return_vert_dist=return_vert_dist, verbose=verbose)
    surface_dist_ot = mesh_surface_nearest_distance(output, target, res, return_vert_dist=return_vert_dist, verbose=verbose)
    ASD = [ surface_dist_gt['distances'].mean(), surface_dist_ot['distances'].mean() ]
    ASSD = (surface_dist_gt['distances'].sum()+surface_dist_ot['distances'].sum())/(len(surface_dist_gt['distances'])+len(surface_dist_ot['distances']))
    RMSD = [ np.sqrt((surface_dist_gt['distances']**2).mean()) , np.sqrt((surface_dist_ot['distances']**2).mean()) ]
    HD = [ surface_dist_gt['distances'].max(), surface_dist_ot['distances'].max() ]
    BHD = max(HD)
    if verbose == True:
        print('# Surface Distances Measurement #')
        print('Average Surface Distances:            gt-based = {:,.2f} mm    ot-based = {:,.2f} mm'.format(ASD[0],ASD[1]))
        print('Average Symmetric Surface Distances            = {:,.2f} mm'.format(ASSD))
        print('RMS Surface Distances:                gt-based = {:,.2f} mm    ot-based = {:,.2f} mm'.format(RMSD[0],RMSD[1]))
        print('Houndorff  Distances:                 gt-based = {:,.2f} mm   ot-based = {:,.2f} mm'.format(HD[0],HD[1]))
        print('Bidirectional Houndorff Distances:             = {:,.2f} mm'.format(BHD))
    
    if return_vert_dist == True:
        return {'surface_dist_gt':surface_dist_gt, 'surface_dist_ot':surface_dist_ot, 
                'ASD_gt':ASD[0],'ASD_ot':ASD[1], 'ASSD':ASSD, 
                'RMSD_gt':RMSD[0], 'RMSD_ot':RMSD[1], 
                'HD_gt':HD[0], 'HD_ot':HD[1], 'BHD':BHD}
    else:
        return {'ASD_gt':ASD[0],'ASD_ot':ASD[1], 'ASSD':ASSD, 
                'RMSD_gt':RMSD[0], 'RMSD_ot':RMSD[1], 
                'HD_gt':HD[0], 'HD_ot':HD[1], 'BHD':BHD}
