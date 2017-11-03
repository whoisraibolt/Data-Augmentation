#!/usr/bin/python
# -*- coding: utf-8 -*-

# Imports
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from os import walk
import numpy as np

# Data Augmentation
data_gen = ImageDataGenerator(# Amount of rotation
                              rotation_range=40,
    
                              # Amount of shift
                              width_shift_range=0.2,
                              height_shift_range=0.2,
    
                              # Shear angle in counter-clockwise direction as radians
                              shear_range=0.2,
    
                              # Range for random zoom
                              zoom_range=0.2,
    
                              # Boolean (True or False). Randomly flip inputs horizontally
                              horizontal_flip=True,
    
                              # Points outside the boundaries of the input are filled
                              # according to the given mode
                              fill_mode='nearest')

# Import folder
def load_dataset(path):
    f = []
    
    # Find all images in folder
    for (dir_path, dir_names, file_names) in walk(path):
        f.extend(file_names)

    # For each image in folder
    for item in f:
        image = Image.open(path+'/'+item).convert('L')
        
        # Create a numpy array with shape (1, 500, 500)
        x = img_to_array(image)
        #x = np.asarray(x)
        
        # Convert to a numpy array with shape (1, 1, 500, 500)
        x = x.reshape((1,) + x.shape)
        
        i = 0
        for batch in data_gen.flow(x, save_to_dir='Data Augmentation Generated', save_prefix='DA', save_format='jpg'):
            i += 1
            if i > 9:
                break 
                
    print('Done!\n')

load_dataset('Test')