#!/usr/bin/env python
# coding: utf-8

# In[1]:


from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from Image_Functions import slicing, crop_images_to_brain, crop_to_size
from datasetModule import Set
import os


# Most of the functionallity is stored in module files. 
# The data consist of Images with 3 channels and segmentation images with 2 channels.
# 

# In[2]:


"Need to specify the local path on computer"
dir_path = os.path.join(os.getcwd(), 'Cropped_Task3')
print(dir_path)


# In[3]:


'Splitting the data into 30% test and 70% training.'
X_train, X_test = train_test_split(Set(dir_path, sub_dir = 'crop_sub-23'), test_size=0.3, random_state=25)


# In[4]:


#X_train, X_test = crop_images_to_brain(X_train), crop_images_to_brain(X_test)
size = (256,288,176)
#size = (128,128,128)
X_train, X_test = crop_to_size(X_train, size), crop_to_size(X_test,size)


# In order to access the set after they have been parsed through the dataloader:
# 

# To access a batch: batch = next(iter(<<Insert name here>>))
# 

# To access the data: batch['batchnumber']['data']
# 

# To access the segmentation: batch['batchnumber']['seg]

# In[ ]:


'Load training and test set, batch size may vary'
train_set, test_set = DataLoader(X_train, batch_size=1), DataLoader(X_test, batch_size=1)


# In[ ]:


tmp = next(iter(train_set))
image = tmp['data']
print(image.shape)


# In[ ]:


import torch
from torch import nn
from torch.nn import parameter
import Model


# In[ ]:


'Run the CNN'
model = Model.CNN(3, None)
print(model)


# In[ ]:


out_img = model(image)


# In[ ]:



print(out_img[0][0].shape)
img = out_img[0][0]
img = img.detach().numpy()
slicing(img)

