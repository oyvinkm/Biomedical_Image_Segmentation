#!/usr/bin/env python
# coding: utf-8

# In[1]:


from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from Image_Functions import slicing, crop_images_to_brain, crop_to_size
from datasetModule import Set
from Model import CNN
from Dice_Loss import DiceLoss
import torch
import os

#hyper parameters
batch_size = 2
learning_rate = 0.01
num_epochs = 4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

from matplotlib import pyplot as plt

"Need to specify the local path on computer"
dir_path = os.path.join(os.getcwd(), 'Cropped_Task3')
print(dir_path)


'Splitting the data into 30% test and 70% training.'
X_train, X_test = train_test_split(Set(dir_path, sub_dir = 'crop_sub-23'), test_size=0.3, random_state=25)
size = (256,288,176)
X_train, X_test = crop_to_size(X_train, size), crop_to_size(X_test,size)

'Load training and test set, batch size may vary'
train_loader, test_loader = DataLoader(X_train, batch_size=1), DataLoader(X_test, batch_size=1)

'Run the CNN'
'Run the CNN'
model = CNN(3,base_features=16).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
n_total_steps = len(train_loader)
loss_func_2 = DiceLoss()

losses = []
for epoch in range(2):
    for i, image_set in enumerate(train_loader):
        image = image_set['data'].to(device)
        labels = image_set['seg'].to(device)
        optimizer.zero_grad()
        outputs = model(image)

        print("outputs shape = ", outputs.shape)
        print("labels shape = ", labels.shape)
        loss = loss_func_2(outputs, labels)
        print(loss.item())
        losses.append(loss.item())
        print(2)
        loss.backward()
        print(3)
        optimizer.step()
        print(4)
        if (i+1) % 1 == 0:
            print(f'epoch {epoch+1} / {num_epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():.4f}')

plt.plot(losses)
plt.savefig('Losses')
print(losses)

test_pred = iter(test_loader)
test_img = test_pred.next()

prediction = model(test_img)

slicing(prediction, 100, 150, 80)




