import numpy as np
import copy
from torch.nn import init


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
        return image

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
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
        elif classname.find('InstanceNorm2d') != -1: 
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)


    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>