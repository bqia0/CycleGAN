import numpy as np
import copy
import os
from torch.nn import init
from torch.utils.data import Dataset
from PIL import Image


class ImagePool(object):
    """ Discriminators are trained using a history of generated images
        to prevent model oscillation.

        Class serves as a buffer for previously generated images.

    """
    def __init__(self, max_elements=50):
        self.max_elements = max_elements
        self.curr_elements = 0
        self.images = []
    
    def __call__(self, in_images):
        """ Return a list of images (same size as in_images)
            If ImagePool is not full, insert new images into buffer.
            Otherwise:
                Half of the time, insert new image into buffer and return 
                previous image.
                Other half of the time, return new image without insertion.
        """
        ret_images = []
        for image in in_images:
            if self.curr_elements < self.max_elements:
                self.images.append(image)
                ret_images.append(image)
                self.curr_elements+=1
            else:
                if np.random.uniform() > 0.5:
                    idx = np.random.randint(0, self.curr_elements)
                    temp = copy.deepcopy(self.images[idx])
                    self.images[idx] = image
                    ret_images.append(temp)
                else:
                    ret_images.append(image)
        return ret_images

# Taken from:
# https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/8cda06f7c36b012769efac63adc1a68586b8fb85/models/networks.py#L67
def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight, 1.0, init_gain)
            init.constant_(m.bias, 0.0)
        # elif classname.find('InstanceNorm2d') != -1: 
        #     init.normal_(m.weight, 1.0, init_gain)
        #     init.constant_(m.bias, 0.0)


    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

class LambdaLR(object):
    """Learning rate schedule lambda function
    
    Constant learning rate until decay_epoch, then linearly decay
    to zero for rest of epochs

    Assume function takes in 1-indexed epochs
    """

    def __init__(self, num_epochs, decay_epoch):
        self.num_epochs = num_epochs
        self.decay_epoch = decay_epoch
        print("LambdaLR: %d epochs | %d decay epoch" % (num_epochs, decay_epoch))
    
    def step(self, epoch):
        ret = 1.0 - max(0, epoch - self.decay_epoch)/(self.num_epochs - self.decay_epoch)
        # print("Learning rate factor: {}".format(ret))
        return ret

class CycleGANDataset(Dataset):
    def __init__(self, root, transform=None, mode='train'):
        """
        Args:
            root (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            mode (string): train or test?
        """
        self.root = root
        self.transform = transform
        self.mode = mode

        self.A_dir = root+ '/%s/%sA' % (mode, mode)
        self.B_dir = root+ '/%s/%sB' % (mode, mode)

        self.files_A = os.listdir(self.A_dir)
        self.files_B = os.listdir(self.B_dir)

    def __len__(self):
        return min(len(self.files_A), len(self.files_B))

    def __getitem__(self, idx):
        A_sample = Image.open(os.path.join(self.A_dir, self.files_A[idx]).replace("\\","/"))
        B_sample = Image.open(os.path.join(self.B_dir, self.files_B[idx]).replace("\\","/"))

        if self.transform:
            A_sample = self.transform(A_sample)
            B_sample = self.transform(B_sample)

        return A_sample, B_sample