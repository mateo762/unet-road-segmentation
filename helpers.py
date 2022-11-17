"""Some helper functions for project 2."""
import csv
import numpy as np
import cv2 as cv

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
    Rotates an image by 0, 90, 180 and 270 degrees.
    
    Arguments: img - numpy array representing the image
               
    Returns: numpy array with the original image and rotations
    """
    
    img_rotations = [img]
    img_rotations.append(cv.rotate(img, cv.ROTATE_90_CLOCKWISE))
    img_rotations.append(cv.rotate(img, cv.ROTATE_180))
    img_rotations.append(cv.rotate(img, cv.ROTATE_90_COUNTERCLOCKWISE))
    
    return np.array(img_rotations)
    
    
    
    
    
    
    
    