# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 00:50:50 2021

@author: michel
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

#---------------------Usefull Functions-------------------------------

def image_preparation(img_path: str, img_size: tuple) -> np.ndarray:
    
    # Load your image
    img = image.load_img(img_path, target_size=img_size)
    
    # Convert your image pixels to a numpy array of values. 
    img = image.img_to_array(img)
    
    # reshape your image dimensions.
    img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
    # img = np.expand_dims(img, axis=0)
    
    # preprocess your image with preprocess_input
    prepared_img = preprocess_input(img)
    return prepared_img

def get_preprocessed_images(img_folder: str, 
                            img_size: tuple) -> np.ndarray:
    images = []
    for fname in os.listdir(img_folder):
        img_path = os.path.join(img_folder, fname)
        img = image_preparation(img_path, img_size)
        images.append(img)
    return np.vstack(images)

def get_images_labels(df: pd.DataFrame, img_folder: str) -> list:
    labels = []
    for fname in os.listdir(img_folder):
        label = df.loc[df['image_id'] == fname[:-4], 'dx'].iloc[0]
        labels.append(label)
    return labels

#---------------------Images loading and prepocessing---------------------

img_size = (224,224)
# Images folders
img_folder_1 = os.path.abspath("./Data/HAM10000_images_part_1")
img_folder_2 = os.path.abspath("./Data/HAM10000_images_part_2")

images_1 = get_preprocessed_images(img_folder_1, img_size)
images_2 = get_preprocessed_images(img_folder_2, img_size)

# Loading Images metadata from csv
df_meta = pd.read_csv("Data/HAM10000_metadata.csv")
image_class = df_meta['dx'].unique().tolist()

# Attaching label to each image
img_labels_1 = get_images_labels(df_meta, img_folder_1)
img_labels_2 = get_images_labels(df_meta, img_folder_2)

# Labels OneHotEncoding
#labels_1 = ...
#labels_2 = ...

# Concatenation
X = np.concatenate([images_1, images_2])
#y = np.concatenate([labels_1, labels_2])

#------------------------Splitting Data----------------------------------

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, 
    #y,
    test_size=0.2, 
    random_state=42, 
    shuffle=True
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, 
    y_train_val,
    test_size=0.2, 
    random_state=42, 
    shuffle=True
)
#------------------------Augmenting Data -------------------------------