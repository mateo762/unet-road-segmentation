"""Some helper functions for project 2."""
import csv
import numpy as np
import cv2 as cv
import random
from PIL import Image, ImageOps
from tensorflow.keras.utils import Sequence
import math

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

def prediction_to_csv(predictions, test_ids, patch_size, threshold):
    """
    Creates a .csv file with the predictions of the test set in order to submit on AICrowd
    
    Arguments: predictions - numpy array of predictions for the test set (one prediction per patch)
               test_ids - list of all the test images names
               patch_size - the patch size that was used to train the model
               threshold - the percentage of road pixels in a 16x16 patch needed for this patch to be labeled as road
               
    Does not return anything but creates a predictions.csv file in the repo folder
    """
    submission = []
    # The constants with _SIDE mean how many patches fit per image in one dimension (one side)
    TEST_IMAGE_LENGTH = 608
    PATCHES_PER_IMAGE_SIDE = math.ceil(TEST_IMAGE_LENGTH/patch_size)
    PATCHES_PER_IMAGE = PATCHES_PER_IMAGE_SIDE**2
    SUBIMAGES_PER_PATCH_SIDE = patch_size/16
    for i, pred in enumerate(predictions):
        img_id = test_ids[i//PATCHES_PER_IMAGE]
        # Format the image id
        id = img_id.split('_')[1].zfill(3)
        # Make sure the patch size is a multiple of 16 otherwise this line won't work
        preds = split_into_patches(pred, 16)
        for j, img in enumerate(preds):
            # Calculate the index of each subimage (in terms of pixels)
            x = 16*(SUBIMAGES_PER_PATCH_SIDE*((i % PATCHES_PER_IMAGE) % PATCHES_PER_IMAGE_SIDE) + j % SUBIMAGES_PER_PATCH_SIDE)
            y = 16*(SUBIMAGES_PER_PATCH_SIDE*((i % PATCHES_PER_IMAGE) // PATCHES_PER_IMAGE_SIDE) + j // SUBIMAGES_PER_PATCH_SIDE)
            # Don't add the padding predictions
            if x < TEST_IMAGE_LENGTH and y < TEST_IMAGE_LENGTH:
                # For now we calculate the average over all the pixels and check if it's above 0.5
                submission.append((f"{id}_{x:.0f}_{y:.0f}", 1 if img.mean() > threshold else 0))
    np.savetxt("predictions.csv", np.asarray(submission), fmt="%s", delimiter=",", newline="\n", header="id,prediction", comments="")