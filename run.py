import os
os.environ["CUDA_VISIBLE_DEVICES"]="0" # Makes visible cuda devices, -1 otherwise
import cv2 as cv
from tqdm import tqdm
from helpers import *
from models import *

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import math
import random

from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

import skimage.io as io
import skimage.transform as trans

import tensorflow as tf
print(f'This value has to be at most 2.10.x ---> {tf.__version__}')

# DATA AUGMENTATION
print("Augmenting the train set")
BASE_TRAINING = './base_training/'
BASE_TRAIN_IMAGES = BASE_TRAINING + 'images/'
BASE_TRAIN_GROUNDTRUTH = BASE_TRAINING + 'groundtruth/'

TRAINING = './training/'
TRAIN_IMAGES = TRAINING + 'images/'
TRAIN_GROUNDTRUTH = TRAINING + 'groundtruth/'

# Remove all images and groundtruths in the folders, do this whenever you want to try new augmentation sets
# We do this here to make sure all goes well if the user reruns the script twice in a row
for filename in tqdm(os.listdir(TRAIN_IMAGES),total=len(os.listdir(TRAIN_IMAGES))):
    file_path = os.path.join(TRAIN_IMAGES, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))
        
for filename in tqdm(os.listdir(TRAIN_GROUNDTRUTH),total=len(os.listdir(TRAIN_GROUNDTRUTH))):
    file_path = os.path.join(TRAIN_GROUNDTRUTH, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))

# Generate the rotated images (90, 180, 270)
for img_name in tqdm(os.listdir(BASE_TRAIN_IMAGES), total=len(os.listdir(BASE_TRAIN_IMAGES))):
    image = cv.imread(BASE_TRAIN_IMAGES + img_name)
    img_train_rotations = get_rotations_0_90_180_270(image)
    for i, rotated_image in enumerate(img_train_rotations[1:]): # Avoid original image
        cv.imwrite(os.path.join(TRAIN_IMAGES , img_name[:-4] + '_rotation_' + str((i+1)*90) + ".png"), rotated_image)
        
# Generate the rotated groundtruths (90, 180, 270)
for img_name in tqdm(os.listdir(BASE_TRAIN_GROUNDTRUTH), total=len(os.listdir(BASE_TRAIN_GROUNDTRUTH))):
    image = cv.imread(BASE_TRAIN_GROUNDTRUTH + img_name)
    image = image[:, :, 0]
    img_train_rotations = get_rotations_0_90_180_270(image)
    for i, rotated_image in enumerate(img_train_rotations[1:]): # Avoid original image
        cv.imwrite(os.path.join(TRAIN_GROUNDTRUTH , img_name[:-4] + '_rotation_' + str((i+1)*90) + ".png"), rotated_image)
        
ROTATIONS_PER_IMAGE = 10
degrees = []
centers = []
for i in range(ROTATIONS_PER_IMAGE*len(os.listdir(BASE_TRAIN_IMAGES))):
    degrees.append(random.randint(0, 90)) # random rotation of degree between 0 and 90
    centers.append((random.randint(45, 55), random.randint(45, 55))) # random center between 45% and 55% of image width & length
    
# Generate the rotated images
index = 0
for img_name in tqdm(os.listdir(BASE_TRAIN_IMAGES), total=len(os.listdir(BASE_TRAIN_IMAGES))):
    image = cv.imread(BASE_TRAIN_IMAGES + img_name)
    for i in range(ROTATIONS_PER_IMAGE):
        random_rotation = get_rotation_deg_n(image, degrees[index], centers[index])
        cv.imwrite(os.path.join(TRAIN_IMAGES , img_name[:-4] + '_random_rotation_' + str(i) + ".png"), random_rotation)
        index += 1
        
# Generate the rotated groundtruths
index = 0
for img_name in tqdm(os.listdir(BASE_TRAIN_GROUNDTRUTH), total=len(os.listdir(BASE_TRAIN_GROUNDTRUTH))):
    image = cv.imread(BASE_TRAIN_GROUNDTRUTH + img_name)
    image = image[:, :, 0]
    for i in range(ROTATIONS_PER_IMAGE):
        random_rotation = get_rotation_deg_n(image, degrees[index], centers[index])
        cv.imwrite(os.path.join(TRAIN_GROUNDTRUTH , img_name[:-4] + '_random_rotation_' + str(i) + ".png"), random_rotation)
        index += 1
        
flip_type = ['x', 'y'] # Types of wanted flips

# Generate the flipped images
for img_name in tqdm(os.listdir(BASE_TRAIN_IMAGES), total=len(os.listdir(BASE_TRAIN_IMAGES))):
    image = cv.imread(BASE_TRAIN_IMAGES + img_name)
    img_train_flips = get_flipped_images(image)
    for i, flipped_image in enumerate(img_train_flips[1:]): # Avoid original image
        cv.imwrite(os.path.join(TRAIN_IMAGES , img_name[:-4] + '_flipped_' + flip_type[i] + ".png"), flipped_image)
        
# Generate the flipped groundtruths
for img_name in tqdm(os.listdir(BASE_TRAIN_GROUNDTRUTH), total=len(os.listdir(BASE_TRAIN_GROUNDTRUTH))):
    image = cv.imread(BASE_TRAIN_GROUNDTRUTH + img_name)
    image = image[:, :, 0]
    img_train_flips = get_flipped_images(image)
    for i, flipped_image in enumerate(img_train_flips[1:]): # Avoid original image
        cv.imwrite(os.path.join(TRAIN_GROUNDTRUTH , img_name[:-4] + '_flipped_' + flip_type[i] + ".png"), flipped_image)
        
# Variance of the gaussian noise
VARIANCE = 50

# Generate the noisy images
for img_name in tqdm(os.listdir(BASE_TRAIN_IMAGES), total=len(os.listdir(BASE_TRAIN_IMAGES))):
    image = cv.imread(BASE_TRAIN_IMAGES + img_name)
    img_train_noise = noisy('gauss', image, var=VARIANCE)
    cv.imwrite(os.path.join(TRAIN_IMAGES , img_name[:-4] + '_noise_' + 'gauss_var_' + str(VARIANCE) + ".png"), img_train_noise)
    
# Copy the groundtruths as they do not change
for img_name in tqdm(os.listdir(BASE_TRAIN_GROUNDTRUTH), total=len(os.listdir(BASE_TRAIN_GROUNDTRUTH))):
    image = cv.imread(BASE_TRAIN_GROUNDTRUTH + img_name)
    image = image[:, :, 0]
    cv.imwrite(os.path.join(TRAIN_GROUNDTRUTH , img_name[:-4] + '_noise_' + 'gauss_var_' + str(VARIANCE) + ".png"), image)

# Variance of the gaussian noise
VARIANCE = 100

# Generate the noisy images
for img_name in tqdm(os.listdir(BASE_TRAIN_IMAGES), total=len(os.listdir(BASE_TRAIN_IMAGES))):
    image = cv.imread(BASE_TRAIN_IMAGES + img_name)
    img_train_noise = noisy('gauss', image, var=VARIANCE)
    cv.imwrite(os.path.join(TRAIN_IMAGES , img_name[:-4] + '_noise_' + 'gauss_var_' + str(VARIANCE) + ".png"), img_train_noise)
    
# Copy the groundtruths as they do not change
for img_name in tqdm(os.listdir(BASE_TRAIN_GROUNDTRUTH), total=len(os.listdir(BASE_TRAIN_GROUNDTRUTH))):
    image = cv.imread(BASE_TRAIN_GROUNDTRUTH + img_name)
    image = image[:, :, 0]
    cv.imwrite(os.path.join(TRAIN_GROUNDTRUTH , img_name[:-4] + '_noise_' + 'gauss_var_' + str(VARIANCE) + ".png"), image)
    
# Ratio of pixels to be corrupted
CORRUPTION_RATIO = 0.01

# Generate the noisy images
for img_name in tqdm(os.listdir(BASE_TRAIN_IMAGES), total=len(os.listdir(BASE_TRAIN_IMAGES))):
    image = cv.imread(BASE_TRAIN_IMAGES + img_name)
    img_train_noise = noisy('s&p', image, var=VARIANCE)
    cv.imwrite(os.path.join(TRAIN_IMAGES , img_name[:-4] + '_noise_' + 's&p_corrupt_' + str(CORRUPTION_RATIO) + ".png"), img_train_noise)

# Copy the groundtruths as they do not change
for img_name in tqdm(os.listdir(BASE_TRAIN_GROUNDTRUTH), total=len(os.listdir(BASE_TRAIN_GROUNDTRUTH))):
    image = cv.imread(BASE_TRAIN_GROUNDTRUTH + img_name)
    image = image[:, :, 0]
    cv.imwrite(os.path.join(TRAIN_GROUNDTRUTH , img_name[:-4] + '_noise_' + 's&p_corrupt_' + str(CORRUPTION_RATIO) + ".png"), image)
    
# Ratio of pixels to be corrupted
CORRUPTION_RATIO = 0.02

# Generate the noisy images
for img_name in tqdm(os.listdir(BASE_TRAIN_IMAGES), total=len(os.listdir(BASE_TRAIN_IMAGES))):
    image = cv.imread(BASE_TRAIN_IMAGES + img_name)
    img_train_noise = noisy('s&p', image, var=VARIANCE)
    cv.imwrite(os.path.join(TRAIN_IMAGES , img_name[:-4] + '_noise_' + 's&p_corrupt_' + str(CORRUPTION_RATIO) + ".png"), img_train_noise)

# Copy the groundtruths as they do not change
for img_name in tqdm(os.listdir(BASE_TRAIN_GROUNDTRUTH), total=len(os.listdir(BASE_TRAIN_GROUNDTRUTH))):
    image = cv.imread(BASE_TRAIN_GROUNDTRUTH + img_name)
    image = image[:, :, 0]
    cv.imwrite(os.path.join(TRAIN_GROUNDTRUTH , img_name[:-4] + '_noise_' + 's&p_corrupt_' + str(CORRUPTION_RATIO) + ".png"), image)
    
# Get the original images
for img_name in tqdm(os.listdir(BASE_TRAIN_IMAGES), total=len(os.listdir(BASE_TRAIN_IMAGES))):
    image = cv.imread(BASE_TRAIN_IMAGES + img_name)
    cv.imwrite(os.path.join(TRAIN_IMAGES + img_name), image)
    
# Get the original groundtruths
for img_name in tqdm(os.listdir(BASE_TRAIN_GROUNDTRUTH), total=len(os.listdir(BASE_TRAIN_GROUNDTRUTH))):
    image = cv.imread(BASE_TRAIN_GROUNDTRUTH + img_name)
    image = image[:, :, 0]
    cv.imwrite(os.path.join(TRAIN_GROUNDTRUTH + img_name), image)
    
# PREPARING THE DATA
print("Loading the data and preparing the model")

TRAIN_DIRECTORY_PATH = './training/'
TRAIN_IMAGES_PATH = TRAIN_DIRECTORY_PATH + 'images/'
TRAIN_GROUNDTRUTH_PATH = TRAIN_DIRECTORY_PATH + 'groundtruth/'

NEW_TRAIN_DIRECTORY_PATH = './new_training/'
NEW_TRAIN_IMAGES_PATH = NEW_TRAIN_DIRECTORY_PATH + 'images/'
NEW_TRAIN_GROUNDTRUTH_PATH = NEW_TRAIN_DIRECTORY_PATH + 'groundtruth/'

TEST_DIRECTORY_PATH = './test_set_images/'
TEST_IMAGES_PATH = [TEST_DIRECTORY_PATH + "test_" + str(i) + "/" for i in range(1,51)]

PATCH_SIZE = 96
NUMBER_NEW_TRAINING_TO_TAKE = 0 # Used even if 0
NUMBER_CHANNELS_INPUT = 3
BATCH_SIZE = 64 

MODEL_FUNCTION = fat_unet 
MODEL = MODEL_FUNCTION((PATCH_SIZE, PATCH_SIZE, NUMBER_CHANNELS_INPUT), verbose = False)

CHECKPOINT_PATH = "./check_points/" + str(MODEL_FUNCTION.__name__)
SAVE_MODEL_PATH = "./models/" + str(MODEL_FUNCTION.__name__) + ".h5"

RANDOM = np.random.randint(69)

tf.random.set_seed(RANDOM)

print("Loading training images")
train_images = []

for file in tqdm(os.listdir(TRAIN_IMAGES_PATH), total=len(os.listdir(TRAIN_IMAGES_PATH))):
    img = plt.imread(TRAIN_IMAGES_PATH + file)
    img_split = split_into_patches(img, PATCH_SIZE)
    train_images.append(img_split)

# New images
new_train_images = []
for num, file in enumerate(os.listdir(NEW_TRAIN_IMAGES_PATH)):
    if num == NUMBER_NEW_TRAINING_TO_TAKE:
        break
    img = plt.imread(NEW_TRAIN_IMAGES_PATH + file)
    img_split = split_into_patches(img, PATCH_SIZE)
    new_train_images.append(img_split)

train_images = np.array(train_images)
new_train_images = np.array(new_train_images)

# Below, this merges the first two dimensions. Instead of having x elements of y patches, we have x*y patches.
train_images = combine_dims(train_images, start = 0, count = 2)
new_train_images = combine_dims(new_train_images, start = 0, count = 2)
print(f'Base train shape: {train_images.shape}')
print(f'New train shape: {new_train_images.shape}')

# Add new training
if NUMBER_NEW_TRAINING_TO_TAKE:
    train_images = np.concatenate((train_images, new_train_images))
    print(f'Concatenated train shape: {train_images.shape}')
    
print("Loading training groundtruths")
train_labels = []

for file in tqdm(os.listdir(TRAIN_GROUNDTRUTH_PATH),total=len(os.listdir(TRAIN_GROUNDTRUTH_PATH))):
    img = plt.imread(TRAIN_GROUNDTRUTH_PATH + file)
    img_split = split_into_patches(img, PATCH_SIZE)
    train_labels.append(img_split)

# New images
new_train_labels = []
for num, file in enumerate(os.listdir(NEW_TRAIN_GROUNDTRUTH_PATH)):
    if num == NUMBER_NEW_TRAINING_TO_TAKE:
        break
    img = plt.imread(NEW_TRAIN_GROUNDTRUTH_PATH + file)
    img_split = split_into_patches(img, PATCH_SIZE)
    new_train_labels.append(img_split)
    
train_labels = np.array(train_labels)
new_train_labels = np.array(new_train_labels)

train_labels = combine_dims(train_labels, start = 0, count = 2)
new_train_labels = combine_dims(new_train_labels, start = 0, count = 2)
# Below, this adds a dimension at the end, such that the image is of size x*x*1, where 1 is the grayscale value of the pixel
train_labels = train_labels[:, :, :, np.newaxis]
if NUMBER_NEW_TRAINING_TO_TAKE:
    new_train_labels = new_train_labels[:, :, :, np.newaxis]
print(train_labels.shape)
print(new_train_labels.shape)

# Add new training
if NUMBER_NEW_TRAINING_TO_TAKE:
    train_labels = np.concatenate((train_labels, new_train_labels))
    print(train_labels.shape)

print("Loading test images")
test_images = []
test_ids = []

for directory in tqdm(TEST_IMAGES_PATH, total=len(TEST_IMAGES_PATH)):
    for file in os.listdir(directory):
        test_ids.append(file)
        img = plt.imread(directory + file)
        img_split = split_into_patches(img, PATCH_SIZE)
        test_images.append(img_split)

test_images = np.array(test_images)
# Below, this merges the first two dimensions. Instead of having x elements of y patches, we have x*y patches.
test_images = combine_dims(test_images, start = 0, count = 2)
print(test_images.shape)

test_ids = [x.split(".")[0] for x in test_ids]

X_train, X_test, y_train, y_test = train_test_split(train_images, train_labels, test_size = 0.20, random_state = RANDOM)
print(X_train.shape)

callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir='logs'),
        tf.keras.callbacks.ModelCheckpoint(filepath=CHECKPOINT_PATH, save_weights_only=True, verbose=1, save_best_only=True)]

train_gen = DataGenerator(X_train, y_train, BATCH_SIZE)
test_gen = DataGenerator(X_test, y_test, BATCH_SIZE)

print("Training the model, this will take a while..")
MODEL.fit(train_gen, verbose=True, epochs=400, validation_data=test_gen, shuffle=True, callbacks=callbacks)
MODEL.save(SAVE_MODEL_PATH)

MODEL.load_weights(CHECKPOINT_PATH) #Loads best model

print("Predicting test images and creating submission file")
test_predictions = MODEL.predict(test_images)
submission_thres = 0.2
prediction_to_csv(test_predictions, test_ids, PATCH_SIZE, submission_thres)