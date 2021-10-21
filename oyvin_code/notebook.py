from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from preprocessing.datasetModule import Set
from Model import CNN
import torch
import os
from datetime import datetime
import nibabel as nib
import numpy as np
import torch.nn as nn
from matplotlib import pyplot as plt
from preprocessing.DataLoader3D import DataLoader3D
from loss import WeightedTverskyLoss, DiceLoss
from Image_Functions import crop_to_size

#hyper parameters
batch_size = 2
learning_rate = 3e-4
num_epochs = 1
base_features =16
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
print(device)

"Need to specify the local path on computer"
test_path = os.getcwd()
print(test_path)
test_imgur = nib.load(os.path.join(test_path, "Cropped_Task3/crop_sub-233/crop_sub-233_space-T1_desc-masked_T1.nii.gz"))

def save_image(data, affine):
    cropped_img = nib.Nifti1Image(data, affine)
    nib.save(cropped_img, "test.nii.gz")

'Splitting the data into 30% test and 70% training.'
dir_path = os.path.join(os.getcwd(), "Cropped_Task3/Segmentations")
X_train, X_test = train_test_split(Set(dir_path, sub_dir = 'crop_sub-2'), test_size=0.3, random_state=25)
X_test = crop_to_size(X_test, (176,176,176))
'Load training and test set, batch size may vary'
train_loader, test_loader = DataLoader3D(X_train, patch_size, BATCH_SIZE=batch_size, device=device), DataLoader(X_test, batch_size=1)

'Run the CNN'
model = CNN(3,base_features=base_features)
#model = nn.DataParallel(model)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

n_total_steps = 10 if 10 > train_loader.get_data_length() else train_loader.get_data_length()
loss_func_2 = DiceLoss()

epoch_losses = []
for epoch in range(num_epochs):
    loss_here = []
    if epoch > 30:
        if (np.max(epoch_losses[-10:] - np.min(epoch_losses[-10:]))) < 1e-3:
            for g in optimizer.param_groups:
                old_lr = g['lr']
                print('Changing learning rate from', old_lr, ' to', old_lr*10)
                g['lr'] = old_lr * 10
    for i, image_set in enumerate(train_loader):
        image = image_set['data'].to(device)
        labels = image_set['seg'].to(device)
        outputs = model(image)
        optimizer.zero_grad(set_to_none=True) #I'd put it further down, it might not make a difference
        loss = loss_func_2(outputs, labels)
        loss_here.append(loss.item())
        loss.backward()
        optimizer.step()
        if i >= n_total_steps:
            break
        if (i+1) % 1 == 0:
            print(f'epoch {epoch+1} / {num_epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():.4f}')
    epoch_losses.append(np.mean(loss_here))        



test_pred = iter(test_loader)
test_img = test_pred.next()

prediction = model(test_img['data'].to(device))

save_image(torch.squeeze(prediction.detach()).cpu().numpy(), test_imgur.affine)
time = datetime.now().replace(microsecond=0).strftime("kl%H%M%S-%d.%m.%Y")
print("The saved name of the file was = ", str(time) , ".csv")
torch.save(prediction, str(time) + ".csv")

plt.plot(epoch_losses)
plt.savefig(('Losses_' + str(num_epochs)))
print(epoch_losses)
