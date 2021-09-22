# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 18:57:25 2021

@author: michel
"""
#---------------------Importing libraries---------------------------------
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping
from mole_preprocessing import (splitting_data, create_features_variable,
     split_cancer_from_not_cancer, load_metadata, create_target_variable,
     augmenting_data, plot_history)
        
#---------------------Images loading and prepocessing---------------------

# Loading Images metadata from csv
df_cancer, df_not_cancer = load_metadata()

# Splitting cancer images from not cancer images
img_size = (224,224)
images_cancer, images_not_cancer = split_cancer_from_not_cancer(df_cancer,
                                                                df_not_cancer,
                                                                img_size)
# create features and target variables
X = create_features_variable(images_cancer, images_not_cancer)
y = create_target_variable(images_cancer, images_not_cancer)

# Taking a subset of the data
X_subset = create_features_variable(images_cancer, images_not_cancer, 1000)
y_subset = create_target_variable(images_cancer, images_not_cancer, 1000)

#Splitting Data
X_train, X_test, X_val, y_train, y_val, y_test = splitting_data(X, y)

# Normalization
X_train = X_train/255
X_val = X_val/255
X_test = X_test/255

#------------------------Augmenting Data -------------------------------

train_generator, validation_generator = augmenting_data(X_train, X_val, 
                                                        y_train, y_val, 
                                                        datagen_batch_size=16)

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
new_model.add(Dense(2, activation='sigmoid'))

# Summarize.
new_model.summary()

#----------------------Training and evaluating the model------------------
#Compiling
new_model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=3)

# Training
history = new_model.fit(train_generator,
                        epochs=10, 
                        batch_size=8,
                        validation_data=validation_generator,
                        callbacks=[early_stop])
# Accuracy and loss plots
plot_history(history)

# saving the model
new_model.save('Mobilenet_model')

# Model Evaluation
new_model.evaluate(X_test, y_test)

# Prediction
preds = np.argmax(new_model.predict(X_test), axis=-1)

sample = X_test[200]
print(np.argmax(new_model.predict(sample.reshape(1,224,224,3)), axis=-1)[0])
print(y_test[200])