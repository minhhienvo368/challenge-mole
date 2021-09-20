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
import matplotlib.pyplot as plt

image_shape = 224
image_class = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']

train_path = 'base_dir/train_dir/'
valid_path = 'base_dir/val_dir/'

train_datagen = ImageDataGenerator(rescale=1./255)

val_datagen = ImageDataGenerator(rescale=1./255)

#declares data generator for train and val batches
train_batches = train_datagen.flow_from_directory(train_path,
                                                        target_size = (image_shape,image_shape),
                                                        classes = image_class,
                                                        batch_size = 64
                                                        )
valid_batches = val_datagen.flow_from_directory(valid_path,
                                                        target_size = (image_shape,image_shape),
                                                        classes = image_class,
                                                        batch_size = 64)

mobile = tensorflow.keras.applications.mobilenet.MobileNet()

x = mobile.layers[-6].output
# Add a dropout and dense layer for predictions
x = Dropout(0.25)(x)
predictions = Dense(7, activation='softmax')(x)
print(mobile.input)
net = Model(inputs=mobile.input, outputs=predictions)
mobile.summary()
for layer in net.layers[:-23]:
    layer.trainable = False
    net.compile(optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'])

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.5, min_lr=0.00001)

history = net.fit(train_batches, epochs=5)

mobile.save('keras_Mobile_model')

def plot_history(history: tensorflow.python.keras.callbacks.History):
    """ This helper function takes the tensorflow.python.keras.callbacks.History
    that is output from your `fit` method to plot the loss and accuracy of
    the training and validation set.
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].plot(history.history['accuracy'], label='training set')
    axs[0].plot(history.history['val_accuracy'], label='validation set')
    axs[0].set(xlabel='Epoch', ylabel='Accuracy', ylim=[0, 1])

    axs[1].plot(history.history['loss'], label='training set')
    axs[1].plot(history.history['val_loss'], label='validation set')
    axs[1].set(xlabel='Epoch', ylabel='Loss', ylim=[0, 10])

    axs[0].legend(loc='lower right')
    axs[1].legend(loc='lower right')
plot_history(history)



