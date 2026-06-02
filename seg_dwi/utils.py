
import numpy as np
import torch
import torch.nn.functional as F
from skimage import measure
import math

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
import SimpleITK as sitk 


def mkdir(FolderPath, rm = False):
    if not os.path.isdir(FolderPath):
        os.makedirs(FolderPath)
    elif rm and os.path.exists(FolderPath):
        shutil.rmtree(path=FolderPath)
        os.makedirs(FolderPath)
    # elif os.path.exists(FolderPath):
    # FolderPath = FolderPath + ' ' + timenow
    # shutil.rmtree(path=FolderPath)
    # os.makedirs(FolderPath)
    return FolderPath

class DSC():
    def __init__(self, epsilon=1e-5, ignore_index=None, **kwargs):
        self.epsilon = epsilon
        self.ignore_index = ignore_index
    def __call__(self, input, target, cal_mean = True):
#         print(input.shape)
        input = (input > 0.5).float()
        target = (target > 0.5).float()
        dice = compute_dice(input, target, epsilon=self.epsilon, ignore_index=self.ignore_index)
        if cal_mean:
            return dice.mean().item()
        return dice.item()
    
def compute_dice(input, target, epsilon=1e-5, ignore_index=None, weight=None):
    # assumes that input is a normalized probability

    # input and target shapes must match
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"

    # mask ignore_index if present
    if ignore_index is not None:
        mask = target.clone().ne_(ignore_index)
        mask.requires_grad = False

        input = input * mask
        target = target * mask

    input = flatten(input)
    target = flatten(target)

    target = target.float()
    # Compute per channel Dice Coefficient
    intersect = (input * target).sum(-1)
    if weight is not None:
        intersect = weight * intersect

    denominator = (input + target).sum(-1)
    return 2. * intersect / denominator.clamp(min=epsilon)    

def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)

def resize_image_itk(itkimage, newSize, resamplemethod=sitk.sitkNearestNeighbor):

    resampler = sitk.ResampleImageFilter()
    originSize = itkimage.GetSize()  
    originSpacing = itkimage.GetSpacing()  
    newSize = np.array(newSize,float)
    factor = originSize / newSize
    newSpacing = originSpacing * factor
    newSize = newSize.astype(np.int)   
    resampler.SetReferenceImage(itkimage)   
    resampler.SetSize(newSize.tolist())
    resampler.SetOutputSpacing(newSpacing.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itkimgResampled = resampler.Execute(itkimage)  
    return itkimgResampled

"""resize原始图像和标签: mask用最近邻插值，MRI图像用线性插值"""
def resize_raw_data(data_path):
    #resize image
    itk_img= sitk.ReadImage(data_path) 
    data_dcminfo = [itk_img.GetSpacing(),itk_img.GetDirection(),itk_img.GetOrigin()]
    tmp_img = sitk.GetArrayFromImage(itk_img)
        
    itk_img = sitk.GetImageFromArray(tmp_img)
        
    return sitk.GetArrayFromImage(itk_img), data_dcminfo

def resize_raw_label(label_path):
    #resize label
    itk_label= sitk.ReadImage(label_path) 

    tmp_label = sitk.GetArrayFromImage(itk_label)


    itk_label = sitk.GetImageFromArray(tmp_label)

    return sitk.GetArrayFromImage(itklabelResampled)
