# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 22:01:16 2021

@author: michel
"""

import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D
from sklearn.metrics import classification_report, confusion_matrix
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

(X_train, y_train),(X_test, y_test) = mnist.load_data()

single_image = X_train[0]
plt.imshow(single_image,cmap='gray')
plt.show()

# Making sure labels are categorical and not continuous variables
y_cat_train = to_categorical(y_train)
y_cat_test = to_categorical(y_test)

# Normalization
X_train = X_train/255
X_test = X_test/255

# Reshaping to let the network know that we are dealing with
# binary images (batch_size,width,height,color_channels)
X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],1)

# Model
model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(4,4),
                 input_shape=(28,28,1), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
# OUTPUT layer -> softmax because multiclassification
model.add(Dense(10, activation='softmax'))
# keras.io/metrics
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=1)

model.fit(X_train, y_cat_train, epochs=10,
          validation_data=(X_test, y_cat_test),
          callbacks=[early_stop])

metrics = pd.DataFrame(model.history.history)
print(metrics)

metrics[['loss', 'val_loss']].plot()
plt.show()

metrics[['accuracy', 'val_accuracy']].plot()
plt.show()

print(model.metrics_names)
model.evaluate(X_test, y_cat_test)

predictions = model.predict_classes(X_test)
# UserWarning: `model.predict_classes()` is deprecated and will be removed 
# after 2021-01-01. Please use instead:* `np.argmax(model.predict(x), axis=-1)`,   
# if your model does multi-class classification   (e.g. if it uses a `softmax` 
# last-layer activation).* `(model.predict(x) > 0.5).astype("int32")`,   
# if your model does binary classification   (e.g. if it uses a `sigmoid` 
# last-layer activation).
# warnings.warn('`model.predict_classes()` is deprecated and '
preds = np.argmax(model.predict(X_test), axis=-1)

print(classification_report(y_test, preds))
cm = confusion_matrix(y_test, preds)
print(cm)

plt.figure(figsize=(12,6))
sns.heatmap(cm, annot=True, cmap='coolwarm')
plt.show()

my_number = X_test[0]
plt.imshow(my_number.reshape(28,28), cmap='gray')
plt.show()

print(np.argmax(model.predict(my_number.reshape(1,28,28,1)), axis=-1))
