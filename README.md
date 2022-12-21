# CS-433 Machine Learning - Project 2
The goal of this project is to train a convolutional neural network to segment roads in satellite images. This repository contains all of our work, including different trainable models, data augmentation techniques, a report and an explanation on how to run the code and experiment with it.  
## Data  
The train and test datasets are available at AICrowd : https://www.aicrowd.com/challenges/epfl-ml-road-segmentation/dataset_files  
## File organization  
In order to run the code you will need the following file organization :   
Repo folder    
├── base_training  
│   └── groundtruth    
│   └── images  
├── training  
│   └── groundtruth  
│   └── images  
├── new_training  
│   └── groundtruth  
│   └── images  
├── test_set_images  
├── check_points  
├── models  
├── logs  
└── actual code files (.ipynb, .py)  
Where   
- the folders in base_training contain the train dataset from AICrowd.  
- the folders in training are empty (they will be populated with the augmented training set).  
- the folders in new_training contain images from another dataset (for example the one given in the report).  
- test_set_images contains the test set images, each in its own folder like in the zip file from AICrowd.  
## Code  
The following files are needed/useful for understanding the project and training the models :  
- models.py, definitions of the different U-Net models used.  
- helpers.py, helper methods used in the other files.  
- ExploreData.ipynb, visual representations of our images and groundtruths to provide more understanding on the data.  
- DataAugmentation.ipynb, file used for augmenting the data set in a customizable manner.  
- Unet.ipynb, file used to train a model and create predictions.  
## Running the code  
In order to train a model, please start by installing the required libraries.
- Tensorflow version 2.10.x
- Keras
- Numpy
- Math
- Random
- Tqdm
- Sklearn
- Matplotlib
- Skimage
- cudatoolkit version 11.2
- cudnn version 8.1.0  

Once this is done the following workflow can be used :   
1) Augment the data by running cells in DataAugmentation.ipynb until satisfied with the train set  
2) Tweak the constants at the start of Unet.ipynb  
3) Run the rest of Unet.ipynb in order to get a model, its predictions and a csv file  
If you simply want to run our best model with our best parameters, just run the run.py file with empty training folders. It will reproduce the workflow from above with set parameters.  
## Running the model the easier way
We also provide our best model [here](https://drive.google.com/file/d/1CEepjjulEFHvq4mVvU2LurGH4eFcsLn8/view?usp=sharing). The model chosen is the large unet, that was trained 300 epochs with a bacth size of 64 on an NVIDIA GeForce GTX 3070 which took about 4 hours.  
To use it, download it from the link above and place it in the models folder. You can then use the line  
```python
MODEL = tf.keras.models.load_model("./models/fat_unet.h5", custom_objects={'get_f1': get_f1})
```
to load and use the model with MODEL.predict().
Please make sure that you have imported helper.py (where get_f1 is defined) and that you have all the necessary libraries enumerated above installed and imported.
We recommend the following:  
 - Download and place the pre trained model into the models folder.
 - Open the Unet.ipynb file and run the following cells
 - 
