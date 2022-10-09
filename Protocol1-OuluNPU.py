#!/usr/bin/env python
# coding: utf-8

# In[31]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# pytorch imports

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F


# In[32]:


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


# In[33]:


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


# In[34]:


from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

img_width, img_height = 100, 100


# In[35]:


mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

transformations = {
    'train': transforms.Compose([
        transforms.Resize((100,100)),
        transforms.CenterCrop((100,100)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize(mean,std)
    ]),
    'test': transforms.Compose([
        transforms.Resize((100,100)),
        transforms.CenterCrop((100,100)),
        transforms.ToTensor(),
        transforms.Normalize(mean,std)
    ])
}


# In[36]:


learning_rate = 0.001
batch_size = 8
num_epochs = 5
num_classes = 2

# device
device = None
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
print(device)


# In[37]:


DIR_PATH='C:\\Users\\SWAPNIL\\Desktop\\New folder\\'


# In[38]:


#total_dataset = torchvision.datasets.ImageFolder(DIR_PATH,transform=transformations['train'])

#len(total_dataset),total_dataset[0][0].shape,total_dataset.class_to_idx


# In[72]:


train ='C:\\Users\\SWAPNIL\\Desktop\\New folder\\Train1\\'
validation ='C:\\Users\\SWAPNIL\\Desktop\\New folder\\Dev1\\'
test ='C:\\Users\\SWAPNIL\\Desktop\\New folder\\Test1\\'
nb_train_samples =400
nb_validation_samples = 100
epochs = 4
batch_size = 20


# In[40]:


train = torchvision.datasets.ImageFolder(train,transform=transformations['train'])

len(train),train[0][0].shape,train.class_to_idx


# In[41]:


val = torchvision.datasets.ImageFolder(validation,transform=transformations['train'])

len(val),val[0][0].shape,val.class_to_idx


# In[73]:


test = torchvision.datasets.ImageFolder(test,transform=transformations['test'])

len(test),test[0][0].shape,test.class_to_idx


# In[74]:


test_loader = DataLoader(dataset=test, batch_size=1, shuffle=True, num_workers=4)
print(len(test_loader))


# In[56]:


val_loader = DataLoader(dataset=val,
                       batch_size=1,
                       shuffle=True,
                       num_workers=4)
print(len(val_loader))


# In[43]:


# dataloaders
train_loader = DataLoader(dataset=train,
                         batch_size=batch_size,
                         shuffle=True,
                         num_workers=4)

val_loader = DataLoader(dataset=validation,
                       batch_size=1,
                       shuffle=True,
                       num_workers=4)
test_loader = DataLoader(dataset=test, batch_size=1, shuffle=True, num_workers=4)


# In[44]:


print(len(train_loader))
print(len(val_loader))
print(len(test_loader))


# In[45]:


# testing dataloading 

examples = iter(train_loader)
samples,labels = examples.next()
print(samples.shape,labels.shape) # batch_size=8
len(train_loader)
len(val_loader)


# In[46]:


# custom CNN model class

class ConvNet(nn.Module):
    def __init__(self,model,num_classes):
        super(ConvNet,self).__init__()
        self.base_model = nn.Sequential(*list(model.children())[:-1]) # model excluding last FC layer
        self.linear1 = nn.Linear(in_features=2048,out_features=512)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(in_features=512,out_features=num_classes)
    
    def forward(self,x):
        x = self.base_model(x)
        x = torch.flatten(x,1)
        lin = self.linear1(x)
        x = self.relu(lin)
        out = self.linear2(x)
        return lin, out


# In[47]:


model = torchvision.models.resnet50(pretrained=True) # base model

model = ConvNet(model,num_classes)

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate,momentum=0.9)


# In[48]:


print(model)


# In[49]:


num_epochs=10


# In[50]:


print(len(train_loader))


# In[51]:


# training loop

n_iters = len(train_loader)

for epoch in range(num_epochs):
    model.train()
    for ii,(images,labels) in enumerate(train_loader):
        print(ii)
        images = images.to(device)
        labels = labels.to(device)
        
        _,outputs = model(images)
        loss = criterion(outputs,labels)
        
        # free_gpu_cache()
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if (ii+1)%108 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{ii+1}/{n_iters}], Loss = {loss.item():.6f}')
            
    print('----------------------------------------')
    


# In[52]:


# evaluating model and getting features of every image

def eval_model_extract_features(features,true_labels,model,dataloader,phase):

    with torch.no_grad():
        # for entire dataset
        n_correct = 0
        n_samples = 0

        model.eval()

        for images,labels in dataloader:

            images = images.to(device)
            labels = labels.to(device)

            true_labels.append(labels)
            
            ftrs,outputs = model(images)
            features.append(ftrs)

            _,preds = torch.max(outputs,1)
            n_samples += labels.size(0)
            n_correct += (preds == labels).sum().item()
                
        accuracy = n_correct/float(n_samples)

        print(f'Accuracy of model on {phase} set = {(100.0 * accuracy):.4f} %')

    return features,true_labels


# In[53]:


features = []
true_labels = []


# In[54]:


train_loader = DataLoader(dataset=train,
                         batch_size=1,
                         shuffle=False,
                         num_workers=4)

features,true_labels = eval_model_extract_features(features,true_labels,model,dataloader=train_loader,phase='training')

print(len(features),len(true_labels))


# In[57]:


features,true_labels = eval_model_extract_features(features,true_labels,model,dataloader=val_loader,phase='validation')

print(len(features),len(true_labels))


# In[58]:


ftrs = features.copy() 
lbls = true_labels.copy()


# In[59]:


for i in range(len(ftrs)):
    ftrs[i]=ftrs[i].cpu().numpy()

ftrs[0].shape


# In[60]:


for i in range(len(lbls)):
    lbls[i]=lbls[i].cpu().numpy()

lbls[0].shape


# In[61]:


type(ftrs),type(lbls)


# In[62]:


ftrs = np.array(ftrs)
lbls = np.array(lbls)

ftrs.shape,lbls.shape


# In[63]:


n_samples = ftrs.shape[0]*ftrs.shape[1]
n_features = ftrs.shape[2]
ftrs = ftrs.reshape(n_samples,n_features)

print(ftrs.shape)


# In[64]:


n_lbls = lbls.shape[0]
lbls = lbls.reshape(n_lbls)

print(lbls.shape)


# In[65]:


# save to csv
ftrs_df = pd.DataFrame(ftrs)
ftrs_df.to_csv('Protocol1_features.csv',index=False)

# reloading the saved csv into a df

ftrs_df = pd.read_csv('Protocol1_features.csv')
ftrs_df


# In[66]:


# appending labels to the feature set
ftrs_df['label'] = lbls

ftrs_df.head()


# In[67]:


ftrs_df.to_csv('olunpu_protocol1.csv',index=False)

print('feature set saved successfully !')


# In[68]:


# save model
MODEL_PATH = 'oulunpu_protocol1.pth'
torch.save(model.state_dict(),MODEL_PATH)


# In[75]:


features,true_labels = eval_model_extract_features(features,true_labels,model,dataloader=test_loader,phase='testing')

print(len(features),len(true_labels))


# In[ ]:




