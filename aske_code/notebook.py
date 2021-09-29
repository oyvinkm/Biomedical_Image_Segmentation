#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from Image_Functions import slicing, crop_images_to_brain, crop_to_size
from datasetModule import Set
from torch import nn
import matplotlib.pyplot as plt
import Model
import tensorflow as tf

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Most of the functionallity is stored in module files. 
# The data consist of Images with 3 channels and segmentation images with 2 channels.
# 

# In[2]:


#hyper parameters
batch_size = 2
learning_rate = 0.01
num_epochs = 4


# In[3]:


"Need to specify the local path on computer"
dir_path = "../Cropped_Task3/"

'Splitting the data into 30% test and 70% training.'
train_set, test_set = train_test_split(Set(dir_path, sub_dir = 'crop_sub-23'), test_size=0.3, random_state=25)

#X_train, X_test = crop_images_to_brain(X_train), crop_images_to_brain(X_test)
size = (256,288,176)
train_set = crop_to_size(train_set, size)
test_set = crop_to_size(test_set, size)

'Load training and test set, batch size my vary'
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)
test_set = None
train_set = None


# In[4]:


"""tmp = next(iter(train_set))
image0 = tmp['seg']
"""
examples = iter(train_loader)
samples = examples.next()
image1 = samples['seg']

image1[0].shape


# In[5]:


'Run the CNN'
model = Model.CNN(3).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, image_set in enumerate(train_loader):
        image = image_set['data'].to(device)

        #We have 2 segementations for every sample, so we must add them together to a single segmentation
        labels = image_set['seg'].to(device)
        print("labels shape", labels.shape)
        outputs = model(image)

        print("outputs shape = ", outputs.shape)
        print("squeeze = ", torch.squeeze(labels,1).shape)
        loss = criterion(outputs, torch.squeeze(labels, 1))
        print(1)
        optimizer.zero_grad()
        print(2)
        loss.backward()
        print(3)
        optimizer.step()
        print(4)
        if (i+1) % 1 == 0:
            print(f'epoch {epoch+1} / {num_epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():.4f}')


# In[ ]:



print(out_img[0][0].shape)
img = out_img[0][0]
img = img.detach().numpy()
for imgur in image[0]:
    slicing(imgur)
#slicing(img)


# In[ ]:




