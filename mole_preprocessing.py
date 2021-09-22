# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 18:54:21 2021

@author: michel
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
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
                            img_folder: str) -> Tuple[np.ndarray]:
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

def splitting_data(X,y):
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
    return X_train, X_test, X_val, y_train, y_val, y_test

def load_metadata():
    
    df_meta = pd.read_csv("Data/HAM10000_metadata.csv")
    df_meta['target'] = df_meta['dx'].apply(cancer_or_not)
    df_cancer = df_meta[df_meta['target']=='cancer']
    df_not_cancer = df_meta[df_meta['target']=='not_cancer']
    return df_cancer, df_not_cancer

def split_cancer_from_not_cancer(df_cancer: pd.DataFrame,
                                 df_not_cancer: pd.DataFrame,
                                 img_size: tuple) -> Tuple[np.ndarray]:
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
    
    return images_cancer, images_not_cancer

def create_features_variable(images_cancer: np.ndarray,
                             images_not_cancer: np.ndarray,
                             subset_size=None) -> np.ndarray:

    # Make a numpy array for each of the class labels (one hot encoded).
    if subset_size:
        subset_cancer = images_cancer[:subset_size]
        subset_not_cancer = images_not_cancer[:subset_size]
        X = np.concatenate([subset_cancer, subset_not_cancer])
    else:
        X = np.concatenate([images_cancer, images_not_cancer])
    
    return X

def create_target_variable(images_cancer: np.ndarray,
                             images_not_cancer: np.ndarray,
                             subset_size=None) -> np.ndarray:
    if subset_size:
        subset_labels_1 = np.tile([1, 0], (images_cancer[:subset_size].shape[0], 1))
        subset_labels_0 = np.tile([0, 1], (images_not_cancer[:subset_size].shape[0], 1))
        y = np.concatenate([subset_labels_1, subset_labels_0])
    else:
        labels_1 = np.tile([1, 0], (images_cancer.shape[0], 1))
        labels_0 = np.tile([0, 1], (images_not_cancer.shape[0], 1))
        y = np.concatenate([labels_1, labels_0])
    return y

def augmenting_data(X_train: np.ndarray, X_val: np.ndarray,
                    y_train: np.ndarray, y_val: np.ndarray,
                    datagen_batch_size: int):
    
    # Make a datagenerator object using ImageDataGenerator.
    train_datagen = ImageDataGenerator(rotation_range=60,
                                        horizontal_flip=True)
    
    # Feed the generator your train data.
    train_generator = train_datagen.flow(X_train, y_train, 
                                         batch_size=datagen_batch_size)
    
    # Make a datagenerator object using ImageDataGenerator.
    validation_datagen = ImageDataGenerator(rotation_range=60,
                                            horizontal_flip=True)
    
    # Feed the generator your validation data.
    validation_generator = validation_datagen.flow(X_val, y_val, 
                                                 batch_size=datagen_batch_size)
    return train_generator, validation_generator

def plot_history(history : tensorflow.keras.callbacks.History):
    """ This helper function takes the tensorflow.python.keras.callbacks.
    History that is output from your `fit` method to plot the loss and 
    accuracy of the training and validation set.
    """
    fig, axs = plt.subplots(1,2, figsize=(12,6))
    axs[0].plot(history.history['accuracy'], label='training set')
    axs[0].plot(history.history['val_accuracy'], label = 'validation set')
    axs[0].set(xlabel = 'Epoch', ylabel='Accuracy', ylim=[0, 1])

    axs[1].plot(history.history['loss'], label='training set')
    axs[1].plot(history.history['val_loss'], label = 'validation set')
    axs[1].set(xlabel = 'Epoch', ylabel='Loss', ylim=[0, 1])
    
    axs[0].legend(loc='lower right')
    axs[1].legend(loc='lower right')