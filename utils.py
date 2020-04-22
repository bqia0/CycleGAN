import numpy as np
import copy

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
