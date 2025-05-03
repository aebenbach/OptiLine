import os 
import numpy as np
from pathlib import Path
from PIL import Image
import cv2

def squish_matrix(matrix):
    if not isinstance(matrix, np.ndarray):
        raise TypeError("Input must be a numpy array")
    if matrix.ndim < 2:
        raise ValueError("Input must be at least a 2D matrix")

    # Flatten across color channels if they exist
    if matrix.ndim == 3:
        nonzero_mask = np.any(matrix != 0, axis=2)
    else:
        nonzero_mask = matrix != 0

    # Find non-zero element indices
    nonzero_indices = np.argwhere(nonzero_mask)

    if nonzero_indices.size == 0:
        # No non-zero elements, return an empty array
        return np.array([[]], dtype=matrix.dtype)

    # Get the bounding box of non-zero elements
    row_min, col_min = nonzero_indices.min(axis=0)
    row_max, col_max = nonzero_indices.max(axis=0)

    # Slice the matrix to the bounding box
    if matrix.ndim == 3:
        squished = matrix[row_min:row_max+1, col_min:col_max+1, :]
    else:
        squished = matrix[row_min:row_max+1, col_min:col_max+1]
    return squished

class ImageIterator:
    def __init__(self, directory_path, index = 0):
        self.image_paths = sorted(Path(directory_path).glob('*.jpg'), key=lambda x: x.name)
        self.index = index

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.image_paths):
            raise StopIteration
        image_path = self.image_paths[self.index]
        self.index += 1

        # Load the image as numpy array (YOLOv8 can accept file path or numpy array)
        img = cv2.imread(str(image_path))  # BGR by default

        return img

class CameraFeed:
    def __init__(self, frames_path, 
                 mask = None,  # Implies an expected image shape
                 skip_to = 0,
                 expected = None,):
        
        self.mask = mask 

        self.feed = ImageIterator(frames_path, index = skip_to)

    def __iter__(self):
        return self

    def __next__(self):
        im = next(self.feed)


        if type(self.mask) != type(None):
            im = im * self.mask
            # print('1',im.shape)
            im = squish_matrix(im)
            # print('2',im.shape)

        return im
    

if __name__ == '__main__':
    feed = CameraFeed(r'OptiLine/data/frames', skip_to= 1929)

    print(next(feed).shape, next(feed))