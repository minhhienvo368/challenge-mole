import pandas as pd
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.models import Model
import tensorflow
import numpy as np
import os
from sklearn.model_selection import train_test_split
import os
import errno
import shutil

base_dir = 'base_dir'
image_class = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']

# 3 folders are made: base_dir, train_dir and val_dir

try:
    os.mkdir(base_dir)

except OSError as exc:
    if exc.errno != errno.EEXIST:
        raise
    pass

train_dir = os.path.join(base_dir, 'train_dir')
try:
    os.mkdir(train_dir)
except OSError as exc:
    if exc.errno != errno.EEXIST:
        raise
    pass
val_dir = os.path.join(base_dir, 'val_dir')
try:
    os.mkdir(val_dir)
except OSError as exc:
    if exc.errno != errno.EEXIST:
        raise
    pass

# make sub directories for the labels
try:
    for x in image_class:
        os.mkdir(train_dir + '/' + x)
    for x in image_class:
        os.mkdir(val_dir + '/' + x)
except OSError as exc:
    if exc.errno != errno.EEXIST:
        raise
    pass
df = pd.read_csv('..\challenge-mole\Data\HAM10000_metadata.csv')

# Set y as the labels
y = df['dx']
X = df.drop(columns=['dx'])

#split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=101)
print(type(y_val))
'''
let's try with a test set after mvp'
'''
#X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.3, random_state=101)# Transfer the images into folders, Set the image id as the index
image_index = df.set_index('image_id', inplace=True)
#X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1, random_state=101)

# Get a list of images in each of the two folders
folder_1 = os.listdir('..\challenge-mole\Data\HAM10000_images_part_1')
folder_2 = os.listdir('..\challenge-mole\Data\HAM10000_images_part_2')

# Get a list of train and val images
train_list = list(X_train['image_id'])
val_list = list(X_val['image_id'])
#print(val_list)

# Transfer the training images
try:
    for image in train_list:
        fname = image + '.jpg'
        if fname in folder_1:
            #print(fname)
            # the source path
            src = os.path.join('..\challenge-mole\Data\HAM10000_images_part_1', fname)
            #print(src)
            # the destination path
            dst = os.path.join(train_dir + '/' + df['dx'][image], fname)
            #print(dst)
            shutil.copyfile(src, dst)
except OSError as exc:
    if exc.errno != errno.EEXIST:
        raise
    pass
    if fname in folder_2:
        # the source path
        src = os.path.join('..\challenge-mole\Data\HAM10000_images_part_2', fname)
        # the destination path
        dst = os.path.join(train_dir, fname)

        shutil.copyfile(src, dst)

# Transfer the validation images

for image in val_list:
    fname = image + '.jpg'
    if fname in folder_1:
        # the source path
        src = os.path.join('..\challenge-mole\Data\HAM10000_images_part_1', fname)
        # the destination path
        dst = os.path.join(val_dir + '/' + df['dx'][image], fname)
        #print(src)
        shutil.copyfile(src, dst)

    if fname in folder_2:
        # the source path
        src = os.path.join('..\challenge-mole\Data\HAM10000_images_part_2', fname)
        # destination path to image
        dst = os.path.join(val_dir, fname)
        # copy the image from the source to the destination
        shutil.copyfile(src, dst)
        #print(df['dx'][image])
        #y_val.append(df['dx'][image])

# Check how many training images are in train_dir
print(len(os.listdir('base_dir/train_dir')))
print(len(os.listdir('base_dir/val_dir')))

# Check how many validation images are in val_dir
print(len(os.listdir('data/HAM10000_images_part_1')))
print(len(os.listdir('data/HAM10000_images_part_2')))

image_class = ['nv','mel','bkl','bcc','akiec','vasc','df']

train_path = 'base_dir/train_dir/'

valid_path = 'base_dir/val_dir/'

#print(os.listdir('base_dir/train_dir'))

#print(len(os.listdir('base_dir/val_dir')))



