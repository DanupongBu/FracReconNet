import torch
import torch.nn.functional as F
from torch import nn as nn
from torch.autograd import Variable
from torch.nn import MSELoss, SmoothL1Loss, L1Loss

#from pytorch3dunet.embeddings.contrastive_loss import ContrastiveLoss
#from pytorch3dunet.unet3d.utils import expand_as_one_hot


def compute_per_channel_dice(input, target, epsilon=1e-6, weight=None):
    """
    Computes DiceCoefficient as defined in https://arxiv.org/abs/1606.04797 given  a multi channel input and target.
    Assumes the input is a normalized probability, e.g. a result of Sigmoid or Softmax function.

    Args:
         input (torch.Tensor): NxCxSpatial input tensor
         target (torch.Tensor): NxCxSpatial target tensor
         epsilon (float): prevents division by zero
         weight (torch.Tensor): Cx1 tensor of weight per channel/class
    """

    # input and target shapes must match
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"

    input = flatten(input)
    target = flatten(target)
    target = target.float()

    # compute per channel Dice Coefficient
    intersect = (input * target).sum(-1)
    if weight is not None:
        intersect = weight * intersect

    # here we can use standard dice (input + target).sum(-1) or extension (see V-Net) (input^2 + target^2).sum(-1)
    denominator = (input * input).sum(-1) + (target * target).sum(-1)
    return 2 * (intersect / denominator.clamp(min=epsilon))


class _MaskingLossWrapper(nn.Module):
    """
    Loss wrapper which prevents the gradient of the loss to be computed where target is equal to `ignore_index`.
    """

    def __init__(self, loss, ignore_index):
        super(_MaskingLossWrapper, self).__init__()
        assert ignore_index is not None, 'ignore_index cannot be None'
        self.loss = loss
        self.ignore_index = ignore_index

    def forward(self, input, target):
        mask = target.clone().ne_(self.ignore_index)
        mask.requires_grad = False

        # mask out input/target so that the gradient is zero where on the mask
        input = input * mask
        target = target * mask

        # forward masked input and target to the loss
        return self.loss(input, target)


class SkipLastTargetChannelWrapper(nn.Module):
    """
    Loss wrapper which removes additional target channel
    """

    def __init__(self, loss, squeeze_channel=False):
        super(SkipLastTargetChannelWrapper, self).__init__()
        self.loss = loss
        self.squeeze_channel = squeeze_channel

    def forward(self, input, target):
        assert target.size(1) > 1, 'Target tensor has a singleton channel dimension, cannot remove channel'

        # skips last target channel if needed
        target = target[:, :-1, ...]

        if self.squeeze_channel:
            # squeeze channel dimension if singleton
            target = torch.squeeze(target, dim=1)
        return self.loss(input, target)


class _AbstractDiceLoss(nn.Module):
    """
    Base class for different implementations of Dice loss.
    """

    def __init__(self, weight=None, sigmoid_normalization=True):
        super(_AbstractDiceLoss, self).__init__()
        self.register_buffer('weight', weight)
        # The output from the network during training is assumed to be un-normalized probabilities and we would
        # like to normalize the logits. Since Dice (or soft Dice in this case) is usually used for binary data,
        # normalizing the channels with Sigmoid is the default choice even for multi-class segmentation problems.
        # However if one would like to apply Softmax in order to get the proper probability distribution from the
        # output, just specify sigmoid_normalization=False.
        if sigmoid_normalization:
            self.normalization = nn.Sigmoid()
        else:
            self.normalization = nn.Softmax(dim=1)

    def dice(self, input, target, weight):
        # actual Dice score computation; to be implemented by the subclass
        raise NotImplementedError

    def forward(self, input, target):
        # get probabilities from logits
        input = self.normalization(input)

        # compute per channel Dice coefficient
        per_channel_dice = self.dice(input, target, weight=self.weight)

        # average Dice score across all channels/classes
        return 1. - torch.mean(per_channel_dice)

class DiceLoss(_AbstractDiceLoss):
    """Computes Dice Loss according to https://arxiv.org/abs/1606.04797.
    For multi-class segmentation `weight` parameter can be used to assign different weights per class.
    The input to the loss function is assumed to be a logit and will be normalized by the Sigmoid function.
    """
    def __init__(self, weight=None, sigmoid_normalization=True):
        super().__init__(weight, sigmoid_normalization)

    def dice(self, input, target, weight):
        return 1.0 - compute_per_channel_dice(input, target, weight=self.weight)

class GeneralizedDiceLoss(_AbstractDiceLoss):
    """Computes Generalized Dice Loss (GDL) as described in https://arxiv.org/pdf/1707.03237.pdf.
    """

    def __init__(self, sigmoid_normalization=True, epsilon=1e-6):
        super().__init__(weight=None, sigmoid_normalization=sigmoid_normalization)
        self.epsilon = epsilon

    def dice(self, input, target, weight):
        assert input.size() == target.size(), "'input' and 'target' must have the same shape"

        input = flatten(input)
        target = flatten(target)
        target = target.float()

        if input.size(0) == 1:
            # for GDL to make sense we need at least 2 channels (see https://arxiv.org/pdf/1707.03237.pdf)
            # put foreground and background voxels in separate channels
            input = torch.cat((input, 1 - input), dim=0)
            target = torch.cat((target, 1 - target), dim=0)

        # GDL weighting: the contribution of each label is corrected by the inverse of its volume
        w_l = target.sum(-1)
        w_l = 1 / (w_l * w_l).clamp(min=self.epsilon)
        w_l.requires_grad = False

        intersect = (input * target).sum(-1)
        intersect = intersect * w_l

        denominator = (input + target).sum(-1)
        denominator = (denominator * w_l).clamp(min=self.epsilon)

        return 2 * (intersect.sum() / denominator.sum())
    
    
class FocalLossMulticlass(nn.Module):
    """
        FocalLossMulticlass
        Args:   gamma = focusing parameter [1,5]
                weight = torch.tensor[weight for each class]
    """
    #def __init__(self, weight=None, gamma=2, reduction='mean'):
    def __init__(self, weight, gamma, reduction='none'):
        super(FocalLossMulticlass, self).__init__()
        self.gamma = gamma
        self.weight = weight       # weight parameter will act as the alpha parameter to balance class weights with define device
        self.reduction = reduction
        
    def forward(self, input, target):
        ''' For target of 4D-Tensor input should be 3D-Tensor as [N,H,W], each voxel is 0,1,2,...,classes-1 
            For target of 5D-Tensor input should be 4D-Tensor as [N,D,H,W], each voxel is 0,1,2,...,classes-1 
            self.weight.size() should be matched number of classes
        '''
        #print('\nFocalLossMulticlass')
        #print('input = {}   {}'.format(input.size(), target.size()))
        ce_loss = F.cross_entropy(input, target, reduction='none', weight=self.weight)
        #print('ce_loss = {}   {}'.format(ce_loss, ce_loss.size()))
        pt = torch.exp(-ce_loss)
        #print('pt = {}   {}'.format(pt, pt.size()))
        focal_loss = ((1 - pt) ** self.gamma * ce_loss)
        #print('focal_loss = {}   {}'.format(focal_loss, focal_loss.size()))
        
        if self.reduction=='mean':
            focal_loss = focal_loss.mean()
        elif self.reduction=='sum':
            focal_loss = focal_loss.sum()
        elif self.reduction=='none':
            pass
        else:
            assert self.reduction in ['mean','none','sum'], 'Input reduction error'
        return focal_loss
    
        '''weight = torch.tensor([0.1,0.1,0.1,0.7,0.1])
        print(weight)
        output = torch.tensor([[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0]]).float()
        target = torch.tensor([0,1,2,3])
        print('output = ', output, output.size())
        print('target = ', target, target.size(), target.dtype)
        criterion = FocalLossMulticlass(weight=weight, gamma=2, reduction='sum')
        loss = criterion(output, target)
        print('\nloss = {}   {}   {}'.format(loss, type(loss), loss.dtype))'''


class BCEDiceLoss(nn.Module):
    """Linear combination of BCE and Dice losses"""

    def __init__(self, alpha, beta):
        super(BCEDiceLoss, self).__init__()
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')   # defualt reduction='mean'  change to 'none' or 'sum'
        self.beta = beta
        self.dice = DiceLoss()

    def forward(self, input, target):
        return self.alpha * self.bce(input, target) + self.beta * self.dice(input, target)
    
class MulticlassBCEDiceLoss(nn.Module):
    """ Linear combination of MulticlassBCE and Dice losses
        pos_weight = tensor size equal to target
        Args:   alpha = weight of BCE loss
                beta = weight of Dice loss
                posWeight = [backMask,boneMask,fracMask]  weight balance
                reduction = 'none' or 'sum' or 'mean'
    """
    def __init__(self, alpha, beta, posWeight, reduction):
        super(MulticlassBCEDiceLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.dice = DiceLoss()
        self.posWeight = posWeight   # tuple(int)
        self.reduction = reduction
    def forward(self, input, target):
        pos_weight = target.clone()
        #print('posWeight = {}'.format(self.posWeight))
        
        pos_weight[:,0] = pos_weight[:,0]*self.posWeight[0]
        pos_weight[:,1] = pos_weight[:,1]*self.posWeight[1]
        pos_weight[:,2] = pos_weight[:,2]*self.posWeight[2]
        #print('posWeight = {}'.format(pos_weight.size()))
        bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction=self.reduction) 
        return self.alpha*bce(input, target) + self.beta*self.dice(input, target)
    
class MulticlassBCEDiceLoss2(nn.Module):
    """ Linear combination of MulticlassBCE and Dice losses
        pos_weight = tensor size equal to target
        Args:   alpha = weight of BCE loss
                beta = weight of Dice loss
                posWeight = tuple(backMask,boneMask,fracMask)  weight balance
                reduction = 'none' or 'sum' or 'mean'
    """
    def __init__(self, alpha, beta, posWeight, reduction):
        super(MulticlassBCEDiceLoss2, self).__init__()
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.beta = beta
        self.dice = DiceLoss()
        self.posWeight = posWeight
        self.reduction = reduction
    def forward(self, input, target):
        posWeight = target.clone()
        loss1 = (self.bce(input[:,0],target[:,0])*self.posWeight[0]).unsqueeze(1)
        loss2 = (self.bce(input[:,1],target[:,1])*self.posWeight[1]).unsqueeze(1)
        loss3 = (self.bce(input[:,2],target[:,2])*self.posWeight[2]).unsqueeze(1)
        #print('loss1 = {}\nloss2 = {}\nloss3 = {}\n'.format(loss1.size(), loss2.size(), loss3.size()))
        bce = torch.cat( (loss1,loss2,loss3) , dim=1)
        #print('bce = {}'.format(bce.size()))
        if self.reduction=='sum':
            bce = bce.sum()
        elif self.reduction=='mean':
            bce = bce.mean()
        else:
            pass
        #print('bce_post = {}'.format(bce.size()))
        return self.alpha*bce + self.beta*self.dice(input, target)

class BCEHNMDiceLoss(nn.Module):
    """ Linear combination of BCE(Hard negative mining) + DiceLoss 
    Args:   alpha = weight of BCE(Hard negative mining)
            beta = weight of DiceLoss
            k = hard negative voxel to calculate
            ratio = ratio of each class to calculate hard negative
    """
    def __init__(self, alpha, beta, k=60, ratio=(0.5,0.5)):
        super(BCEHNMDiceLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.ratio = ratio
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.dice = DiceLoss()
        
    def forward(self, input, target):
        """ Forward pass
        Args:   input = calculated output from network  [N,C,D,H,W]   dtype=torch.float32
                target = target output from dataset     [N,C,D,H,W]   dtype=torch.float32
        """
        #print('BCEHNMDiceLoss:\ntarget = {}'.format(target.size()))
        N, C, _, _, _ = target.size()
        assert C==len(self.ratio) , 'ratio should have same dimension as C'
        
        bce = self.bce(input, target)
        loss_matrix = (target*bce)
        loss_flatten = loss_matrix.view(N,C,-1)
        loss_sort , _ = torch.sort(loss_flatten, dim=2, descending=True)
        #print('loss_sort = {}\n'.format(loss_sort))
        
        bce_hnm_loss = 0
        for i in range(C):
            #print('i:{}  K={}  loss_sort:{}'.format(i,round(self.ratio[i]*self.k),loss_sort[:,i,:].size()))
            bce_hnm_loss += loss_sort[:,i,0:round(self.ratio[i]*self.k)].sum()
            #print('bce_hnm_loss_i = {}'.format(bce_hnm_loss))
        
        dice_loss = self.dice(input,target)
        #print('\ndice_loss = {}'.format(dice_loss))
        return self.alpha*bce_hnm_loss + self.beta*dice_loss
        
class WeightedCrossEntropyLoss(nn.Module):
    """WeightedCrossEntropyLoss (WCE) as described in https://arxiv.org/pdf/1707.03237.pdf
    """

    def __init__(self, ignore_index=-1):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, input, target):
        weight = self._class_weights(input)
        return F.cross_entropy(input, target, weight=weight, ignore_index=self.ignore_index)

    @staticmethod
    def _class_weights(input):
        # normalize the input first
        input = F.softmax(input, dim=1)
        flattened = flatten(input)
        nominator = (1. - flattened).sum(-1)
        denominator = flattened.sum(-1)
        class_weights = Variable(nominator / denominator, requires_grad=False)
        return class_weights

class PixelWiseCrossEntropyLoss(nn.Module):
    def __init__(self, class_weights=None, ignore_index=None):
        super(PixelWiseCrossEntropyLoss, self).__init__()
        self.register_buffer('class_weights', class_weights)
        self.ignore_index = ignore_index
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, target, weights):
        assert target.size() == weights.size()
        # normalize the input
        log_probabilities = self.log_softmax(input)
        # standard CrossEntropyLoss requires the target to be (NxDxHxW), so we need to expand it to (NxCxDxHxW)
        target = expand_as_one_hot(target, C=input.size()[1], ignore_index=self.ignore_index)
        # expand weights
        weights = weights.unsqueeze(0)
        weights = weights.expand_as(input)

        # create default class_weights if None
        if self.class_weights is None:
            class_weights = torch.ones(input.size()[1]).float().to(input.device)
        else:
            class_weights = self.class_weights

        # resize class_weights to be broadcastable into the weights
        class_weights = class_weights.view(1, -1, 1, 1, 1)

        # multiply weights tensor by class weights
        weights = class_weights * weights

        # compute the losses
        result = -weights * target * log_probabilities
        # average the losses
        return result.mean()


class TagsAngularLoss(nn.Module):
    def __init__(self, tags_coefficients):
        super(TagsAngularLoss, self).__init__()
        self.tags_coefficients = tags_coefficients

    def forward(self, inputs, targets, weight):
        assert isinstance(inputs, list)
        # if there is just one output head the 'inputs' is going to be a singleton list [tensor]
        # and 'targets' is just going to be a tensor (that's how the HDF5Dataloader works)
        # so wrap targets in a list in this case
        if len(inputs) == 1:
            targets = [targets]
        assert len(inputs) == len(targets) == len(self.tags_coefficients)
        loss = 0
        for input, target, alpha in zip(inputs, targets, self.tags_coefficients):
            loss += alpha * square_angular_loss(input, target, weight)

        return loss


class WeightedSmoothL1Loss(nn.SmoothL1Loss):
    def __init__(self, threshold, initial_weight, apply_below_threshold=True):
        super().__init__(reduction="none")
        self.threshold = threshold
        self.apply_below_threshold = apply_below_threshold
        self.weight = initial_weight

    def forward(self, input, target):
        l1 = super().forward(input, target)

        if self.apply_below_threshold:
            mask = target < self.threshold
        else:
            mask = target >= self.threshold

        l1[mask] = l1[mask] * self.weight

        return l1.mean()


def square_angular_loss(input, target, weights=None):
    """
    Computes square angular loss between input and target directions.
    Makes sure that the input and target directions are normalized so that torch.acos would not produce NaNs.

    :param input: 5D input tensor (NCDHW)
    :param target: 5D target tensor (NCDHW)
    :param weights: 3D weight tensor in order to balance different instance sizes
    :return: per pixel weighted sum of squared angular losses
    """
    assert input.size() == target.size()
    # normalize and multiply by the stability_coeff in order to prevent NaN results from torch.acos
    stability_coeff = 0.999999
    input = input / torch.norm(input, p=2, dim=1).detach().clamp(min=1e-8) * stability_coeff
    target = target / torch.norm(target, p=2, dim=1).detach().clamp(min=1e-8) * stability_coeff
    # compute cosine map
    cosines = (input * target).sum(dim=1)
    error_radians = torch.acos(cosines)
    if weights is not None:
        return (error_radians * error_radians * weights).sum()
    else:
        return (error_radians * error_radians).sum()


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    # number of channels
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)


def get_loss_criterion(config):
    """
    Returns the loss function based on provided configuration
    :param config: (dict) a top level configuration object containing the 'loss' key
    :return: an instance of the loss function
    """
    assert 'loss' in config, 'Could not find loss function configuration'
    loss_config = config['loss']
    name = loss_config.pop('name')

    ignore_index = loss_config.pop('ignore_index', None)
    skip_last_target = loss_config.pop('skip_last_target', False)
    weight = loss_config.pop('weight', None)

    if weight is not None:
        # convert to cuda tensor if necessary
        weight = torch.tensor(weight).to(config['device'])

    pos_weight = loss_config.pop('pos_weight', None)
    if pos_weight is not None:
        # convert to cuda tensor if necessary
        pos_weight = torch.tensor(pos_weight).to(config['device'])

    loss = _create_loss(name, loss_config, weight, ignore_index, pos_weight)

    if not (ignore_index is None or name in ['CrossEntropyLoss', 'WeightedCrossEntropyLoss']):
        # use MaskingLossWrapper only for non-cross-entropy losses, since CE losses allow specifying 'ignore_index' directly
        loss = _MaskingLossWrapper(loss, ignore_index)

    if skip_last_target:
        loss = SkipLastTargetChannelWrapper(loss, loss_config.get('squeeze_channel', False))

    return loss


SUPPORTED_LOSSES = ['BCEWithLogitsLoss', 'BCEDiceLoss', 'CrossEntropyLoss', 'WeightedCrossEntropyLoss',
                    'PixelWiseCrossEntropyLoss', 'GeneralizedDiceLoss', 'DiceLoss', 'TagsAngularLoss', 'MSELoss',
                    'SmoothL1Loss', 'L1Loss', 'WeightedSmoothL1Loss']


def _create_loss(name, loss_config, weight, ignore_index, pos_weight):
    if name == 'BCEWithLogitsLoss':
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif name == 'BCEDiceLoss':
        alpha = loss_config.get('alphs', 1.)
        beta = loss_config.get('beta', 1.)
        return BCEDiceLoss(alpha, beta)
    elif name == 'CrossEntropyLoss':
        if ignore_index is None:
            ignore_index = -100  # use the default 'ignore_index' as defined in the CrossEntropyLoss
        return nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
    elif name == 'WeightedCrossEntropyLoss':
        if ignore_index is None:
            ignore_index = -100  # use the default 'ignore_index' as defined in the CrossEntropyLoss
        return WeightedCrossEntropyLoss(ignore_index=ignore_index)
    elif name == 'PixelWiseCrossEntropyLoss':
        return PixelWiseCrossEntropyLoss(class_weights=weight, ignore_index=ignore_index)
    elif name == 'GeneralizedDiceLoss':
        sigmoid_normalization = loss_config.get('sigmoid_normalization', True)
        return GeneralizedDiceLoss(sigmoid_normalization=sigmoid_normalization)
    elif name == 'DiceLoss':
        sigmoid_normalization = loss_config.get('sigmoid_normalization', True)
        return DiceLoss(weight=weight, sigmoid_normalization=sigmoid_normalization)
    elif name == 'TagsAngularLoss':
        tags_coefficients = loss_config['tags_coefficients']
        return TagsAngularLoss(tags_coefficients)
    elif name == 'MSELoss':
        return MSELoss()
    elif name == 'SmoothL1Loss':
        return SmoothL1Loss()
    elif name == 'L1Loss':
        return L1Loss()
    elif name == 'ContrastiveLoss':
        return ContrastiveLoss(loss_config['delta_var'], loss_config['delta_dist'], loss_config['norm'],
                               loss_config['alpha'], loss_config['beta'], loss_config['gamma'])
    elif name == 'WeightedSmoothL1Loss':
        return WeightedSmoothL1Loss(threshold=loss_config['threshold'], initial_weight=loss_config['initial_weight'],
                                    apply_below_threshold=loss_config.get('apply_below_threshold', True))
    else:
        raise RuntimeError(f"Unsupported loss function: '{name}'. Supported losses: {SUPPORTED_LOSSES}")
