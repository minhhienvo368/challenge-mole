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

def get_preprocessed_images(df_cancer: pd.DataFrame, 
                            df_not_cancer: pd.DataFrame, 
                            img_size: tuple, 
                            img_folder: str) -> np.ndarray:
    images_cancer = []
    images_not_cancer = []
    for fname in os.listdir(img_folder):
        if fname[:-4] in df_cancer['image_id'].tolist():
            img_path = os.path.join(img_folder, fname)
            img = image_preparation(img_path, img_size)
            images_cancer.append(img)
        elif fname[:-4] in df_not_cancer['image_id'].tolist():
            img_path = os.path.join(img_folder, fname)
            img = image_preparation(img_path, img_size)
            images_not_cancer.append(img)
    return np.vstack(images_cancer), np.vstack(images_not_cancer)

def cancer_or_not(dx_value: str) -> str:
    if dx_value in ['bcc', 'mel']:
        output = 'cancer'
    else:
        output = 'not_cancer'
    return output

#---------------------Images loading and prepocessing---------------------

# Loading Images metadata from csv
df_meta = pd.read_csv("Data/HAM10000_metadata.csv")
df_meta['target'] = df_meta['dx'].apply(cancer_or_not)
df_cancer = df_meta[df_meta['target']=='cancer']
df_not_cancer = df_meta[df_meta['target']=='not_cancer']

img_size = (224,224)
# Images folders
img_folder_1 = os.path.abspath("./Data/HAM10000_images_part_1")
img_folder_2 = os.path.abspath("./Data/HAM10000_images_part_2")

images_cancer_1, images_not_cancer_1 = get_preprocessed_images(
    df_cancer,
    df_not_cancer,
    img_size,
    img_folder_1)
images_cancer_2, images_not_cancer_2 = get_preprocessed_images(
    df_cancer,
    df_not_cancer,
    img_size,
    img_folder_2)

# Concatenate cancer images
images_cancer = np.concatenate([images_cancer_1, images_cancer_2])

# Concatenate not cancer images
images_not_cancer = np.concatenate([images_not_cancer_1, images_not_cancer_2])

# Make a numpy array for each of the class labels (one hot encoded).
labels_1 = np.tile([1, 0], (images_cancer.shape[0], 1))
labels_0 = np.tile([0, 1], (images_not_cancer.shape[0], 1))

# Concatenation
X = np.concatenate([images_cancer, images_not_cancer])
y = np.concatenate([labels_1, labels_0])

#------------------------Splitting Data----------------------------------

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, 
    y,
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