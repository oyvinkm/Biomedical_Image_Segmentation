from sklearn.utils import shuffle
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from preprocessing.datasetModule import Set
from Model import CNN
import torch
import os
from datetime import datetime
import nibabel as nib
import numpy as np
import csv
from torchvision import transforms
import torch.nn as nn
from matplotlib import pyplot as plt
from preprocessing.DataLoader3D import DataLoader3D
from loss import WeightedTverskyLoss, DiceLoss, BinaryFocalLoss
from Image_Functions import crop_to_size, save_slice, slicing
try:
    from preprocessing.DataAugmentation import (AddGaussianNoise, 
                                                RotateRandomTransform, 
                                                FlipTransform, 
                                                AddRicianNoise,
                                                SpatialTransformsRotate)
except:
    print('Path not good enough')
#hyper parameters
batch_size = 2
learning_rate = 3e-4
num_epochs = 50
base_features = 16 
patch_size = (128,128,128)
maxpool = nn.MaxPool3d
def CreateSeg(seg, x):
    segs = [None for _ in range(len(x))]
    segs[-1] = seg
    for i in range(len(x)-2, 0, -1):
        max = maxpool((2,2,2))
        segs[i] = max(seg[i+1])
    return segs     


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"Need to specify the local path on computer"
test_path = os.getcwd()
test_imgur = nib.load(os.path.join(test_path, "Cropped_Task3/crop_sub-233/crop_sub-233_space-T1_desc-masked_T1.nii.gz"))

def save_image(data, affine, name):
    cropped_img = nib.Nifti1Image(data, affine)
    nib.save(cropped_img, ('name'+'.nii.gz'))   

'Splitting the data into 30% test and 70% training.'

dir_path = os.path.join(os.getcwd(), "Cropped_Task3/Segmentations")
sub_dir = 'crop_sub-2'
data_folders = sorted([folder for folder  in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, folder)) and sub_dir in folder])
train, test = train_test_split(data_folders)
print(train[:2])

X_train = Set(dir_path, train)
X_test = Set(dir_path, test)
""" FlipTransform(), 
AddRicianNoise(),
AddGaussianNoise() """
#X_train = Set(dir_path, train)
#X_test = Set(dir_path, test)





'Load training and test set, batch size may vary'
train_loader= DataLoader3D(X_train, patch_size, BATCH_SIZE=batch_size, device=device)
test_loader = DataLoader(X_train, shuffle=True)


""" for i, img in enumerate(train_loader):
    save_slice(img['seg'][0][0], ('seg_sub2' + str(i) +'.png'))
    if not train_loader.GetDialate():
        train_loader.EnableDialate()
    train_loader.UpdateDialPad(1)
    if i == 15:
        break """


'Run the CNN'
model = CNN(3,base_features=base_features)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

n_total_steps = 1
loss_func_2 =BinaryFocalLoss()

epoch_losses = []
folder = '{}_{}'.format('Loss_func', str(num_epochs))
folder_path = os.path.join(os.getcwd(),'{}_{}'.format('Loss_func', str(num_epochs)))
for epoch in range(num_epochs):
    loss_here = []
    if epoch <= 20 and epoch % 2 == 0:
        train_loader.UpdateDialPad(-1)
    elif epoch > 20:
        train_loader.EnableDialate     
    for i, image_set in enumerate(train_loader):
        image = image_set['data'].to(device)
        labels = image_set['seg'].to(device)
        print(image_set['keys'])
        outputs = model(image)
        optimizer.zero_grad(set_to_none=True) #I'd put it further down, it might not make a difference
        loss = loss_func_2(outputs, labels)
        loss_here.append(loss.item())
        loss.backward()
        optimizer.step()
        if (i+1) % 1 == 0:
            print(f'epoch {epoch+1} / {num_epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():.4f}')
        if i >= n_total_steps-1:
            np.savetxt((f"file_name_{epoch}.csv"), np.array(loss_here), delimiter=",", fmt='%s')
            if not os.path.exists(folder_path):
                os.mkdir(folder_path)
            save_slice(outputs[0][0].detach().cpu().numpy(), os.path.join(folder, str(epoch)))
            break
    epoch_losses.append(np.mean(loss_here))        



test_pred = iter(test_loader)
test_img = test_pred.next()

prediction = model(test_img['data'].to(device))

save_image(torch.squeeze(prediction.detach()).cpu().numpy(), test_imgur.affine, 'pred_seg')
save_image(torch.squeeze(test_img['seg'].detach()).cpu().numpy(), test_imgur.affine, 'true_seg')
#time = datetime.now().replace(microsecond=0).strftime("kl%H%M%S-%d.%m.%Y")
#print("The saved name of the file was = ", str(time) , ".csv")
#torch.save(prediction, str(time) + ".csv")

plt.plot(epoch_losses)
plt.savefig(('Losses_' + str(num_epochs)))
print(epoch_losses)
