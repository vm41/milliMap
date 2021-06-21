### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import torch
# Create a model and return to the model

def create_model(opt):
    if opt.model == 'pix2pixHD': # select pix2pixHD model
        from .pix2pixHD_model import Pix2PixHDModel, InferenceModel
        if opt.isTrain: # True for training
            model = Pix2PixHDModel()
        else: # Otherwise, if only forward propagation is used for demonstration, it is False

            model = InferenceModel()
    else: # select UIModel model
    	from .ui_model import UIModel
    	model = UIModel()
    model.initialize(opt) # Model initialization parameters
    if opt.verbose: # The default is false, indicating that no model has been saved before
        print("model [%s] was created" % (model.name())) # print label2city model was created


    if opt.isTrain and len(opt.gpu_ids):
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids) # Multi-GPU training


    return model
