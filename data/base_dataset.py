### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

 # This function resizes or crops the appropriate input size according to the method specified by the user.
 # size: Enter the size of the picture

import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import random

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass

def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    if opt.resize_or_crop == 'resize_and_crop': # opt.loadSize is the size you enter, scale the image to this size
        new_h = new_w = opt.loadSize  # Set the width and height to the same size          
    elif opt.resize_or_crop == 'scale_width_and_crop': # I have set ‘scale_width_and_crop’ at opt
        new_w = opt.loadSize
        new_h = opt.loadSize * h // w # The height is calculated according to the original image aspect ratio


    x = random.randint(0, np.maximum(0, new_w - opt.fineSize)) #? ? ? I don't understand what the random number means here

    y = random.randint(0, np.maximum(0, new_h - opt.fineSize))
    
    flip = random.random() > 0.5  # Whether the random number is greater than 0.5, flip is a bool type variable, this line of code means randomly generated True or Fals
    return {'crop_pos': (x, y), 'flip': flip} # The final return value, on line 45 of data.aligned_dataset, is passed as a params to the get_transform() function below


 # Image transformation
def get_transform(opt, params, method=Image.BICUBIC, normalize=True):
    transform_list = []
    if 'resize' in opt.resize_or_crop: # If opt.resize_or_crop has'resize
        osize = [opt.loadSize, opt.loadSize]
        transform_list.append(transforms.Scale(osize, method))   
    elif 'scale_width' in opt.resize_or_crop:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.loadSize, method)))
        ### The lambda function is also called an anonymous function, that is, the function has no specific name. Let's take a look at the simplest example:
    # def f(x):
    #   return x**2
    # print f(4)
    #
         # If you use lambda in Python, write it like this:
    # g = lambda x : x**2
    # print g(4)

        
        
    if 'crop' in opt.resize_or_crop:  # Use transforms.Lambda to encapsulate it as a transforms strategy
        transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.fineSize)))

    if opt.resize_or_crop == 'none':
        base = float(2 ** opt.n_downsample_global)
        if opt.netG == 'local':
            base *= (2 ** opt.n_local_enhancers)
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base, method)))

    if opt.isTrain and not opt.no_flip:
        transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    transform_list += [transforms.ToTensor()]

    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5), # mean and std are both 0.5
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def normalize():    
    return transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size        
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img
    return img.resize((w, h), method)

def __scale_width(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    if (ow == target_width):
        return img    
    w = target_width
    h = int(target_width * oh / ow)    
    return img.resize((w, h), method)

 # Random panning sliding crop
def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size # input size opt.fineSize
    if (ow > tw or oh > th):        
        return img.crop((x1, y1, x1 + tw, y1 + th)) # Random crop, because although the size of each crop is the same, the starting point is different

    return img

def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img
