###############################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
###############################################################################
import torch.utils.data as data
from PIL import Image
import os

 # Picture extensions supported by this program

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
]


def is_image_file(filename):
    ### The any() function is used to determine whether the given iterable parameter iterable is all False, then return False, and if one is True, return True.
         # Elements are TRUE except 0, empty, and FALSE.
         # The function is equivalent to:
    # def any(iterable):
    #     for element in iterable:
    #         if element:
    #             return True
    #     return False
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

# Make a data set: get a picture path list of the data set
def make_dataset(dir): # dir is the path of the data set folder

    images = []  # create empty list
    assert os.path.isdir(dir), '%s is not a valid directory' % dir # confirm path exists
### The os.walk() method is a simple and easy-to-use file and directory traversal tool that can help us efficiently deal with files and directories
         # top - is the address of the directory you want to traverse, and returns a triple (root, dirs, files).
         # root refers to the address of the folder currently being traversed, which is the same as the entered dir of os.walk(dir)
         # dirs is a list, the content is the names of all the directories in the folder (excluding subdirectories), if not, it is []
         # files is also a list, the content is the name of all files in the folder (excluding subdirectories), if not, it is []

    for root, _, fnames in sorted(os.walk(dir)): # fnames is the photo file read in the file

        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname) # combine the folder path dir with the picture name fname
                images.append(path) # Store the image path in the image list
 # temp = fname
                # images.append(temp)

    return images # Return to the image path list



def default_loader(path):
    return Image.open(path).convert('RGB')


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = make_dataset(root) # imgs is a list of image paths in the root directory
        if len(imgs) == 0: # Number of pictures = 0 Error
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index] # Get the specified image path
        img = self.loader(path) # load image
        if self.transform is not None:
            img = self.transform(img) # image to transform
        if self.return_paths:
            return img, path # return image and path
        else:
            return img # return image only

    def __len__(self):
        return len(self.imgs) # return the number of pictures in the specified directory

