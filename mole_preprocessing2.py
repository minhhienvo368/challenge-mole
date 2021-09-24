# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 14:39:52 2021

@author: michel
"""

import os
import shutil
import itertools
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from typing import Tuple
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

def cancer_or_not(dx_value: str) -> str:
    if dx_value in ['bcc', 'mel']:
        output = 'cancer'
    else:
        output = 'not_cancer'
    return output

def load_metadata():
    
    df_meta = pd.read_csv("Data/HAM10000_metadata.csv")
    df_meta['target'] = df_meta['dx'].apply(cancer_or_not)
    return df_meta

def oversampling(img_folder: str):
    n = 1
    for fname in os.listdir(img_folder):
        img_path = os.path.join(img_folder,fname)
        im = Image.open(img_path)
        out_LR = im.transpose(Image.FLIP_LEFT_RIGHT)
        out_LR.save('cancer/fliph_'+str(n) + '.jpg')
        out_TB = im.transpose(Image.FLIP_TOP_BOTTOM)
        out_TB.save('cancer/flipv_'+str(n) + '.jpg')
        out_R90 = im.transpose(Image.ROTATE_90)
        out_R90.save('cancer/rot90_'+str(n) + '.jpg')
        out_R180 = im.transpose(Image.ROTATE_180)
        out_R180.save('cancer/rot180_'+str(n) + '.jpg')
        # out_R270 = im.transpose(Image.ROTATE_270)
        # out_R270.save('cancer/rot270_'+str(n) + '.jpg')
        n += 1

def spliting_fill_img(df_in: pd.DataFrame, img_folder: str):

    list_imgs = list(df_in['image_id'])
    folder = os.listdir(img_folder)

    for i in range(len(df_in.target.unique())):
        path1 = os.path.join('./', df_in.target.unique()[i])
        if not os.path.isdir(path1):
            os.mkdir(path1) 

    df_in = df_in.set_index('image_id')
    for index in list_imgs:
        fname = index + '.jpg'
        label = df_in.loc[index,'target']

        if fname in folder:
            # source path to image
            src = os.path.join(img_folder, fname)
            # destination path to image
            dst = os.path.join('./', label, fname)
            # copy the image from the source to the destination
            shutil.copyfile(src, dst)
            
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

def get_preprocessed_images(img_folder: str, img_size: tuple) -> list:
    images = []
    for item in os.listdir(img_folder):
        img_path = os.path.join(img_folder, item)
        img = image_preparation(img_path, img_size)
        images.append(img)
    return np.vstack(images)

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

def create_target_variable(cancer_labels: np.ndarray,
                           not_cancer_labels: np.ndarray,
                             subset_size=None) -> np.ndarray:
    if subset_size:
        subset_cancer = cancer_labels[:subset_size]
        subset_not_cancer = not_cancer_labels[:subset_size]
        y = np.concatenate([subset_cancer, subset_not_cancer])
    else:
        y = np.concatenate([cancer_labels, not_cancer_labels])
    return y

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
    
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
        print("Normalize confusion matrix")
    else:
        print("Confusion matrix, without Normalization.")
    print(cm)
    thresh = cm.max()/2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig("Visuals/confusion_matrix.png")
    plt.show()

#----------------------------------------------------------------------------
# loading metadata
df_meta = load_metadata()

# splitting images into 2 folders: cancer and not_cancer
# spliting_fill_img(df_meta, 'Data/ham10000_images_part_1')
# spliting_fill_img(df_meta, 'Data/ham10000_images_part_2')

# oversampling
# img_folder = os.path.abspath('./cancer')
# print(len(os.listdir(img_folder)))
# oversampling(img_folder)

# Load your images and preprocess them.
img_size = (224,224)
cancer_folder = os.path.abspath('./cancer')
not_cancer_folder = os.path.abspath('./not_cancer')

cancer_images = get_preprocessed_images(cancer_folder, img_size)
not_cancer_images = get_preprocessed_images(not_cancer_folder, img_size)

# one Hot encoding
cancer_labels = np.tile(1, (cancer_images.shape[0], 1))
not_cancer_labels = np.tile(0, (not_cancer_images.shape[0], 1))

# create features and target variables
X = create_features_variable(cancer_images, not_cancer_images)
y = create_target_variable(cancer_labels, not_cancer_labels)

#Splitting Data
X_train, X_test, X_val, y_train, y_val, y_test = splitting_data(X, y)

#------------------------Augmenting Data -------------------------------

train_generator, validation_generator = augmenting_data(X_train, X_val, 
                                                        y_train, y_val, 
                                                        datagen_batch_size=64)

#---------------------Importing an existing Model-----------------------------
# Make sure you exclude the top part. 
# set the input shape of the model to 224x224 pixels, with 3 color channels.
model = MobileNetV2(weights='imagenet', 
                    include_top=False, 
                    input_shape=(224,224,3))

# Freeze the imported layers so they cannot be retrained.
for layer in model.layers:
    layer.trainable = False
    
model.summary()

#---------------Adding flattening and dense layers-----------------------

new_model = Sequential()
new_model.add(model)
new_model.add(Flatten())
new_model.add(Dense(64, activation='relu'))
new_model.add(Dropout(0.5))
new_model.add(Dense(1, activation='sigmoid'))

# Summarize.
new_model.summary()

#----------------------Training and evaluating the model------------------
#Compiling
new_model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
# Stop the model training in case of overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=2)

# Training
history = new_model.fit(train_generator,
                        epochs=8, 
                        batch_size=64,
                        validation_data=validation_generator,
                        callbacks=[early_stop]) 
 
plot_history(history)

# saving the model
new_model.save('melanomia.h5')
new_model.save('mole_model')

# Model Evaluation
new_model.evaluate(X_test, y_test, verbose=2)

# predictions
predictions = new_model.predict(X_test)
preds = []
for n in predictions:
    if n > 0.5:
        preds.append(1)
    else:
        preds.append(0)
preds = np.array(preds).reshape(-1,1)

print(classification_report(y_test, preds))

cm = confusion_matrix(y_test, preds)
print(cm)

plot_confusion_matrix(cm, classes=['cancer','not_cancer'])

sample = X_test[:50]
print(np.argmax(new_model.predict(sample.reshape(50,224,224,3)), axis=-1))
print(y_test[:50])