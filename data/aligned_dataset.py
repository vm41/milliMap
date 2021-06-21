### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
# Data reading method
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image

 # Return to a dictionary with organized data sets: pictures + categories
class AlignedDataset(BaseDataset): # init are all path settings
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot    
 ### input A (label maps) # The path of the label map
        ### input A (label maps)
        dir_A = '_A' if self.opt.label_nc == 0 else '_label'
        self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A) # './geometry' + 'train' + '_label
        ### sort is a method applied to list, sorted can sort all iterable objects.
                 # The sort method of list returns the operation on the existing list;
                 # And the built-in function sorted method returns a new list, rather than the operation based on the original
                 # (It turns out that sorting the string directly is not the same as the actual int value sorting result, the picture names are not in order from small to large)
        self.A_paths = sorted(make_dataset(self.dir_A)) # return the list of picture paths under self.dir_A


        ### input B (real images) # path of real image
        if opt.isTrain or opt.use_encoded_image:
            dir_B = '_B' if self.opt.label_nc == 0 else '_img'
            self.dir_B = os.path.join(opt.dataroot, opt.phase + dir_B)  
            self.B_paths = sorted(make_dataset(self.dir_B))
            # self.B_paths = self.A_paths


        ### instance maps # path of instance map
        if not opt.no_instance: # If no_instance is true, no instance graph is added
            self.dir_inst = os.path.join(opt.dataroot, opt.phase + '_inst')
            self.inst_paths = sorted(make_dataset(self.dir_inst))
            # self.inst_paths = self.A_paths

        ### load precomputed instance-wise encoded features
        if opt.load_features:                              
            self.dir_feat = os.path.join(opt.dataroot, opt.phase + '_feat')
            print('----------- loading features from %s ----------' % self.dir_feat)
            self.feat_paths = sorted(make_dataset(self.dir_feat))  # There is no train_feat picture in this article

        self.dataset_size = len(self.A_paths) 
      
    def __getitem__(self, index):   # The specific operations in getitem are the key operations of this class
 ### input A (label maps) # Read label map      
        ### input A (label maps)
        A_path = self.A_paths[index]   # Get image path
         # A = Image.open(self.dir_A +'/' + A_path) # First read an image

        A = Image.open(A_path)        
        params = get_params(self.opt, A.size) # According to the input opt ​​and size, return random parameters
        if self.opt.label_nc == 0:
            transform_A = get_transform(self.opt, params)
            tmp = A.convert('RGB')
            A_tensor = transform_A(tmp)
        else:
            transform_A = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
            A_tensor = transform_A(A) * 255.0 # To preprocess the data, after to_tensor operation, multiply by 255


        B_tensor = inst_tensor = feat_tensor = 0
        ### input B (real images) ### input B (real images) # Then read in the real image B
        if self.opt.isTrain or self.opt.use_encoded_image:
            B_path = self.B_paths[index]  # B = Image.open(self.dir_B + '/' + B_path).convert('RGB') 
            B = Image.open(B_path).convert('RGB')
            transform_B = get_transform(self.opt, params)      
            B_tensor = transform_B(B)

        ### if using instance maps    ### if using instance maps # Then read in the instance, and then it will be processed into an edge map, which is consistent with the description in the paper.
      
        if not self.opt.no_instance: # no_instance default value is true
            inst_path = self.inst_paths[index] # inst = Image.open(self.dir_inst + '/' + inst_path)
            inst = Image.open(inst_path)
            inst_tensor = transform_A(inst) # Same as semantic processing 0-1

            if self.opt.load_features: # Note that the role of self.opt.load_features is to read the pre-calculated features of each category. There are 10 categories in the paper, which are formed by clustering. But the default is not implemented. I personally read the thesis and knew nothing about this part. I will study it later if there is demand.

                feat_path = self.feat_paths[index]            
                feat = Image.open(feat_path).convert('RGB')
                norm = normalize()
                feat_tensor = norm(transform_A(feat))                            

        input_dict = {'label': A_tensor, 'inst': inst_tensor, 'image': B_tensor, 
                      'feat': feat_tensor, 'path': A_path}

        return input_dict # Return a dictionary recording the above read and processed data set.


    def __len__(self):
        return len(self.A_paths) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'AlignedDataset'
