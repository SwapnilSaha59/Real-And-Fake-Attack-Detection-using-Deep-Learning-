#!/usr/bin/env python
# coding: utf-8

# In[65]:


import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os, PIL
from glob import glob
import tensorflow as tf
from io import StringIO 
from PIL import Image
import pydot
import imageio as iio
import cv2

from __future__ import print_function
import pandas as pd
import shutil
import os
import sys

#import seaborn as sns
from sklearn import model_selection


import keras
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import  img_to_array
from tensorflow.keras.preprocessing.image import array_to_img


from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tqdm import tqdm


# In[66]:


pip install opencv-python


# In[67]:


pip install tqdm


# In[68]:


""" Sequential Model Architecture """
Sequential = tf.keras.models.Sequential

""" Data Preprocessing Functions """
Resizing = tf.keras.layers.experimental.preprocessing.Resizing
Rescaling = tf.keras.layers.experimental.preprocessing.Rescaling

""" Data Augmentation Functions """
RandomFlip = tf.keras.layers.experimental.preprocessing.RandomFlip
RandomRotation = tf.keras.layers.experimental.preprocessing.RandomRotation
RandomZoom = tf.keras.layers.experimental.preprocessing.RandomZoom

""" Artificial Neural Network Layer Inventory """
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout

""" Convolutional Neural Network Layer Inventory """
Conv2D = tf.keras.layers.Conv2D
MaxPool2D = tf.keras.layers.MaxPool2D
Flatten = tf.keras.layers.Flatten

""" Residual Network Layer Inventory """
ResNet50 = tf.keras.applications.resnet50.ResNet50

""" Function to Load Images from Target Folder """
get_image_data = tf.keras.utils.image_dataset_from_directory


# In[69]:


from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

img_width, img_height = 100, 100


# In[70]:


if K.image_data_format() == 'channels_first':
	input_shape = (3, img_width, img_height)
else:
	input_shape = (img_width, img_height, 3)


# In[71]:


train_data_dir ='Train1'
validation_data_dir ='Dev'
nb_train_samples =400
nb_validation_samples = 100
epochs = 10
batch_size = 25


# In[94]:


class_names = ['Real','Attack']
class_names_label = {class_name:i for i, class_name in enumerate(class_names)}

nb_classes = len(class_names)

IMAGE_SIZE = (100, 100)


# In[95]:


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # warning disabling

training_dataset = get_image_data(
    directory=train_data_dir,
    seed=42,
    image_size=(img_width, img_height),
    batch_size=batch_size
)

validation_dataset = get_image_data(
    directory=validation_data_dir,
    seed=42,
    image_size=(img_width, img_height),
    batch_size=batch_size
)
class_names=training_dataset.class_names
print(class_names)


# In[96]:


def load_data():
    datasets = ['Train1', 'Dev']
    output = []
    
    # Iterate through training and test sets
    for dataset in datasets:
        
        images = []
        labels = []
        
        print("Loading {}".format(dataset))
        
        # Iterate through each folder corresponding to a category
        for folder in os.listdir(dataset):
            label = class_names_label[folder]
            
            # Iterate through each image in our folder
            for file in tqdm(os.listdir(os.path.join(dataset, folder))):
                
                # Get the path name of the image
                img_path = os.path.join(os.path.join(dataset, folder), file)
                
                # Open and resize the img
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, IMAGE_SIZE) 
                
                # Append the image and its corresponding label to the output
                images.append(image)
                labels.append(label)
                
        images = np.array(images, dtype = 'float32')
        labels = np.array(labels, dtype = 'int32')   
        
        output.append((images, labels))

    return output


# In[97]:


(train_images, train_labels), (test_images, test_labels) = load_data()


# In[98]:


n_train = train_labels.shape[0]
n_test = test_labels.shape[0]

print ("Number of training examples: {}".format(n_train))
print ("Number of testing examples: {}".format(n_test))
print ("Each image is of size: {}".format(IMAGE_SIZE))


# In[99]:


import pandas as pd

_, train_counts = np.unique(train_labels, return_counts=True)
_, test_counts = np.unique(test_labels, return_counts=True)
pd.DataFrame({'train': train_counts,
                    'test': test_counts}, 
             index=class_names
            ).plot.bar()
plt.show()


# In[100]:


plt.pie(train_counts,
        explode=(0, 0) , 
        labels=class_names,
        autopct='%1.1f%%')
plt.axis('equal')
plt.title('Proportion of each observed category')
plt.show()


# In[101]:


train_images = train_images / 255.0 
test_images = test_images / 255.0


# In[102]:


def display_random_image(class_names, images, labels):
    """
        Display a random image from the images array and its correspond label from the labels array.
    """
    
    index = np.random.randint(images.shape[0])
    plt.figure()
    plt.imshow(images[index])
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.title('Image #{} : '.format(index) + class_names[labels[index]])
    plt.show()


# In[103]:


display_random_image(class_names, train_images, train_labels)


# In[104]:


def display_examples(class_names, images, labels):
    """
        Display 25 images from the images array with its corresponding labels
    """
    
    fig = plt.figure(figsize=(10,10))
    fig.suptitle("Some examples of images of the dataset", fontsize=16)
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[labels[i]])
    plt.show()


# In[105]:


display_examples(class_names, train_images, train_labels)


# In[106]:


resizing_layer = layers.experimental.preprocessing.Resizing(img_width, img_height)
normalization_layer = layers.experimental.preprocessing.Rescaling(1./255, 
                                                                  input_shape=(img_width, img_height, 
                                                                               3))

def configure_performant_datasets(dataset, shuffling=None):
    """ Custom function to prefetch and cache stored elements 
    of retrieved image data to boost latency and performance 
    at the cost of higher memory usage. """    
    AUTOTUNE = tf.data.AUTOTUNE
    # Cache and prefetch elements of input data for boosted performance
    if not shuffling:
        return dataset.cache().prefetch(buffer_size=AUTOTUNE)
    else:
        return dataset.cache().shuffle(shuffling).prefetch(buffer_size=AUTOTUNE)
    
    
    
training_dataset = configure_performant_datasets(training_dataset, 
                                                 shuffling=1000)
validation_dataset = configure_performant_datasets(validation_dataset)


# In[107]:


def plot_training_results(history):
    """
    Visualize results of the model training using `matplotlib`.

    The visualization will include charts for accuracy and loss, 
    on the training and as well as validation data sets.

    INPUTS:
        history(tf.keras.callbacks.History): 
            Contains data on how the model metrics changed 
            over the course of training.
    
    OUTPUTS: 
        None.
    """
    accuracy = history.history['accuracy']
    accuracy
    validation_accuracy = history.history['val_accuracy']
    validation_accuracy

    loss = history.history['loss']
    loss
    validation_loss = history.history['val_loss']
    validation_loss

    epochs_range = range(epochs)
    epochs_range

    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, accuracy, label='Training Accuracy')
    plt.plot(epochs_range, validation_accuracy, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, validation_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


# In[108]:


from tensorflow.keras import optimizers
filters = [[16,2],[16,3],[32,2],[32,3],[64,2],[64,3],[128,2],[128,3]]
for x,y in filters:
    model = Sequential()
    model.add(layers.Conv2D(x, (y, y), activation='relu', input_shape=(img_width,img_height,3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(x, (y, y), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(x, (y, y), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2,activation = 'softmax'))
    model.summary()
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
			optimizer='Adam',
			metrics=['accuracy'])
    epochs = 10
    history = model.fit(train_images, train_labels, batch_size=25, epochs = epochs, validation_split = 0.2)
    plot_training_results(history)


# In[109]:


test_loss = model.evaluate(test_images, test_labels)


# In[110]:


from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

model = VGG16(weights='imagenet', include_top=False)


# In[111]:


train_features = model.predict(training_dataset)
test_features = model.predict(validation_dataset)


# In[112]:


n_train, x, y, z = train_features.shape
n_test, x, y, z = test_features.shape
numFeatures = x * y * z


# In[113]:


from sklearn import decomposition

pca = decomposition.PCA(n_components = 2)

X = train_features.reshape((n_train, x*y*z))
pca.fit(X)

C = pca.transform(X) # Représentation des individus dans les nouveaux axe
C1 = C[:,0]
C2 = C[:,1]


# In[114]:


plt.subplots(figsize=(10,10))

for i, class_name in enumerate(class_names):
    plt.scatter(C1[train_labels == i][:1000], C2[train_labels == i][:1000], label = class_name, alpha=0.4)
plt.legend()
plt.title("PCA Projection")
plt.show()


# In[116]:


resizing_layer = layers.experimental.preprocessing.Resizing(img_width, img_height)
normalization_layer = layers.experimental.preprocessing.Rescaling(1./255, 
                                                                  input_shape=(img_width, img_height, 
                                                                               3))

def configure_performant_datasets(dataset, shuffling=None):
    """ Custom function to prefetch and cache stored elements 
    of retrieved image data to boost latency and performance 
    at the cost of higher memory usage. """    
    AUTOTUNE = tf.data.AUTOTUNE
    # Cache and prefetch elements of input data for boosted performance
    if not shuffling:
        return dataset.cache().prefetch(buffer_size=AUTOTUNE)
    else:
        return dataset.cache().shuffle(shuffling).prefetch(buffer_size=AUTOTUNE)
    
    
    
training_dataset = configure_performant_datasets(training_dataset, 
                                                 shuffling=1000)
validation_dataset = configure_performant_datasets(validation_dataset)


# In[118]:


from tensorflow.keras import optimizers
filters = [[16,2],[16,3],[32,2],[32,3],[64,2],[64,3],[128,2],[128,3]]
for x,y in filters:
    model = Sequential()
    model.add(layers.Conv2D(x, (y, y), activation='relu', input_shape=(img_width,img_height,3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(x, (y, y), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(x, (y, y), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2,activation = 'softmax'))
    model.summary()
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
			optimizer='Adam',
			metrics=['accuracy'])
    epochs = 20
    history = model.fit(train_images, train_labels, batch_size=128, epochs=epochs, validation_split = 0.2)


# In[120]:


plot_training_results(history)


# In[123]:


test_loss = model2.evaluate(test_features, test_labels)


# In[124]:


np.random.seed(seed=1997)
# Number of estimators
n_estimators = 10
# Proporition of samples to use to train each training
max_samples = 0.8

max_samples *= n_train
max_samples = int(max_samples)


# In[135]:


models = list()
random = np.random.randint(50, 100, size = n_estimators)

for i in range(n_estimators):
    
    # Model
    model = tf.keras.Sequential([ tf.keras.layers.Flatten(input_shape = (3, 3, 512)),
                                # One layer with random size
                                    tf.keras.layers.Dense(random[i], activation=tf.nn.relu),
                                    tf.keras.layers.Dense(6, activation=tf.nn.softmax)
                                ])
    
    model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Store model
    models.append(model)


# In[136]:


histories = []

for i in range(n_estimators):
    train_idx = np.random.choice(len(train_features), size = max_samples)
    histories.append(models[i].fit(train_features[train_idx], train_labels[train_idx], batch_size=25, epochs=10, validation_split = 0.1))


# In[137]:


predictions = []
for i in range(n_estimators):
    predictions.append(models[i].predict(test_features))
    
predictions = np.array(predictions)
predictions = predictions.sum(axis = 0)
pred_labels = predictions.argmax(axis=1)


# In[138]:


from sklearn.metrics import accuracy_score
print("Accuracy : {}".format(accuracy_score(test_labels, pred_labels)))


# In[139]:


from keras.models import Model

model = VGG16(weights='imagenet', include_top=False)
model = Model(inputs=model.inputs, outputs=model.layers[-5].output)


# In[140]:


train_features = model.predict(train_images)
test_features = model.predict(test_images)


# In[177]:


from keras.layers import Input, Dense, Conv2D, Activation , MaxPooling2D, Flatten

model2 = VGG16(weights='imagenet', include_top=False)
input_shape = model2.layers[-4].get_input_shape_at(0) # get the input shape of desired layer
layer_input = Input(shape = (9, 9, 512)) # a new input tensor to be able to feed the desired layer
# https://stackoverflow.com/questions/52800025/keras-give-input-to-intermediate-layer-and-get-final-output
new_model = tf.keras.Sequential([ tf.keras.layers.Flatten(input_shape = (6, 6, 512)),
                                # One layer with random size
                                    tf.keras.layers.Dense(random[i], activation=tf.nn.relu),
                                    tf.keras.layers.Dense(1024, activation=tf.nn.softmax)
                                ])


# In[178]:


new_model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])


# In[179]:


new_model.summary()


# In[181]:


history5 = new_model.fit(train_features, train_labels, batch_size=128, epochs=20, validation_split = 0.2)


# In[183]:


plot_training_results(history5)


# In[184]:


from sklearn.metrics import accuracy_score

predictions = new_model.predict(test_features)    
pred_labels = np.argmax(predictions, axis = 1)
print("Accuracy : {}".format(accuracy_score(test_labels, pred_labels)))

