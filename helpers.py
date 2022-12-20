"""Some helper functions for project 2."""
import csv
import numpy as np
import cv2 as cv
import random
from PIL import Image, ImageOps
from tensorflow.keras.utils import Sequence

class DataGenerator(Sequence):
    """
    Generator for the input data, this allows us to use a lot of inputs with the gpu even if the gpu cannot store all of it at once
    taken from https://stackoverflow.com/questions/62916904/failed-copying-input-tensor-from-cpu-to-gpu-in-order-to-run-gatherve-dst-tensor
    """
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y

def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y == "b")] = -1

    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in .csv format for submission to Kaggle or AIcrowd
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, "w") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})

def split_into_patches(img, patchsize):
    """
    Splits an image into patches of size patchsize. Splits "on a grid" and pads with RGB value (0,0,0)
    if the dimension of the image is not integer divisible by patchsize.
    
    Arguments: img - numpy array representing the image
               patchsize - size of the patches that the image will be cut into
               
    Returns: numpy array of patches of the original image
    """
    
    height = img.shape[0]
    width = img.shape[1]
    
    height_extra = height % patchsize
    width_extra = width % patchsize
    if (height_extra != 0 or width_extra != 0): # In case image does not split into grid, pad it with RGB = (0,0,0)
        pad_bottom = 0
        pad_right = 0
        if(height_extra != 0): # In case the image is not a square and one dimension is divisible by patchsize
            pad_bottom = patchsize - height_extra
        if(width_extra != 0):
            pad_right = patchsize - width_extra
        img = cv.copyMakeBorder(img, 0, pad_bottom, 0, pad_right, cv.BORDER_CONSTANT, value=(0, 0, 0))
        height = img.shape[0]
        width = img.shape[1]
        
    sub_images = []
    for x in range(height // patchsize):
        for y in range(width // patchsize):
            tlc = (x*patchsize, y*patchsize) # Top left corner
            sub_images.append(img[tlc[0]:tlc[0]+patchsize,tlc[1]:tlc[1]+patchsize])
    return np.array(sub_images)

def split_into_random_n_patches(img, n, patchsize, tlcs=[]):
    """
    Splits an image into random patches of size patchsize. Patches will always be inside the image. (no padding)
    
    Arguments: img - numpy array representing the image
               n - number of patches to generate
               patchsize - size of the patches that the image will be cut into
               tlcs - [OPTIONAL] list of coordinates of the top-left corners of the patches. Makes this function
               deterministic; used to generate the same patches for ground truth images.
               
    Returns: numpy array of patches of the image
             python list of tuples representing the coordinates of the top left corners of the patches
    """
    
    tlcs_empty_flag = len(tlcs) == 0
    assert tlcs_empty_flag or len(tlcs) == n, "Invalid top-left corners list. Must be either empty or have length n"
    
    sub_images = []
    width_valid_tlc = img.shape[0] - patchsize
    height_valid_tlc = img.shape[1] - patchsize
    
    for i in range(n):
        if tlcs_empty_flag: # This allows us to use this function for both images and ground truth
            # In case of image: we generate random coordinates
            tlc = np.random.randint(0, width_valid_tlc), np.random.randint(0, height_valid_tlc)
            tlcs.append(tlc)
        # Fetch coordinates either from above or already given tlcs list
        sub_images.append(img[tlcs[i][0]:tlcs[i][0]+patchsize,tlcs[i][1]:tlcs[i][1]+patchsize])
            
    return np.array(sub_images), tlcs

def get_rotations_0_90_180_270(img):
    """
    Rotates an image by 0, 90, 180 and 270 degrees
    
    Arguments: img - numpy array representing the image
               
    Returns: numpy array with the original image and rotations
    """
    
    img_rotations = [img]
    img_rotations.append(cv.rotate(img, cv.ROTATE_90_CLOCKWISE))
    img_rotations.append(cv.rotate(img, cv.ROTATE_180))
    img_rotations.append(cv.rotate(img, cv.ROTATE_90_COUNTERCLOCKWISE))
    
    return np.array(img_rotations)

def get_rotation_deg_n(img, degrees, center=(50, 50)):
    """
    Rotates an image by degrees degrees
    
    Arguments: img - numpy array representing the image
               degrees - degree of rotation
               
    Returns: numpy array of the rotated image
    """
    
    h, w = img.shape[:2]
    cX, cY = 0.01 * center[0] * w, 0.01 * center[0] * h
    M = cv.getRotationMatrix2D((cX, cY), degrees, 1.0)
    rotated = cv.warpAffine(img, M, (w, h))
    return rotated 

def get_flipped_images(img):
    """
    Flips an image vertically and horizontally
    
    Arguments: img - numpy array representing the image
               
    Returns: numpy array with the original image and flipped images
    """
    
    img_flipped = [img]
    img_flipped.append(cv.flip(img,0))
    img_flipped.append(cv.flip(img,1))
    
    return np.array(img_flipped)

def noisy(noise_type, img, corruption_ratio = 0.01, var = 50):
    """
    Adds noise to an image (either gaussian or salt and pepper noise)
    
    Arguments:  noise_type - either gaussian or salt and pepper
                img - numpy array representing the image
                corruption_ratio - the percentage of pixels to be changed to white or black in salt and pepper
                var - the variance of the gaussian distribution for gauss
                
    Returns: numpy array of the noisy image            
    """
    
    # Getting the dimensions of the image
    row, col, ch = img.shape
    if noise_type == "gauss":
        mean = 0
        sigma = var**0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = img + gauss
        return noisy
    elif noise_type == "s&p":
        number_of_pixels = corruption_ratio*row*col
        # Randomly pick some pixels in the
        # image for coloring them white
        
        for i in range(int(number_of_pixels)):

            # Pick a random y coordinate
            y_coord=random.randint(0, row - 1)

            # Pick a random x coordinate
            x_coord=random.randint(0, col - 1)

            # Color that pixel to white
            img[y_coord][x_coord] = 255

        # Randomly pick some pixels in
        # the image for coloring them black
        for i in range(int(number_of_pixels)):

            # Pick a random y coordinate
            y_coord=random.randint(0, row - 1)

            # Pick a random x coordinate
            x_coord=random.randint(0, col - 1)

            # Color that pixel to black
            img[y_coord][x_coord] = 0

        return img

def combine_dims(img, start, count):
    """
    Reshapes a numpy array a by combining count number of dimensions, starting at dimension index start
    
    Arguments: img - numpy array representing the image
               start - index of the first dimension to combine
               count - number of dimensions to combine
               
    Returns: numpy array resized
    """
    
    shape = img.shape
    return np.reshape(img, shape[:start] + (-1,) + shape[start+count:])


def padding(img, expected_size):
    desired_size = expected_size
    delta_width = desired_size - img.size[0]
    delta_height = desired_size - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding)

def resize_with_padding(img, expected_size):
    img.thumbnail((expected_size[0], expected_size[1]))
    # print(img.size)
    delta_width = expected_size[0] - img.size[0]
    delta_height = expected_size[1] - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding)
    
    
    
    
    
    
    
    