from operator import delitem
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from image_Functions import slicing, crop_images_to_brain, crop_to_size, save_image, save_slice, ContinuoslySaving
from preprocessing.datasetModule import Set
from preprocessing.DataLoader3D import DataLoader3D
from model import CNN
from loss import DiceLoss, WeightedTverskyLoss, _BCEWithLogitsLoss, BinaryFocalLoss
import torch
import os
import numpy as np
from datetime import datetime
import nibabel as nib
import torch.nn as nn
from matplotlib import pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cluster_path = os.path.join(os.getcwd(), 'Biomedical_Image_Segmentation/Cropped_Task3/crop_sub-233/crop_sub-233_space-T1_desc-masked_T1.nii.gz')
pc_path = os.path.join(os.getcwd(), 'Cropped_Task3/crop_sub-233/crop_sub-233_space-T1_desc-masked_T1.nii.gz')
test_path = cluster_path if torch.cuda.is_available() else pc_path


#hyper parameters
batch_size = 2
learning_rate = 0.1
num_epochs = 10
base_features = 2
patch_size = (128,128,128)
n_total_steps = 1

"Need to specify the local path on computer"
dir_path = os.path.join(os.getcwd(), "Segmentations")
sub_dir = 'crop_sub-2'

data_folders = sorted([folder for folder  in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, folder)) and sub_dir in folder])
train, test = train_test_split(data_folders)
test_imgur = nib.load(test_path)

'Splitting the data into 30% test and 70% training.'
train_set, test_set = train_test_split(data_folders)

train_set = Set(dir_path, train_set[:1])
print(train_set)
test_set = Set(dir_path, test_set)

'Load training and test set, batch size my vary'
train_loader = DataLoader3D(train_set, patch_size, BATCH_SIZE=batch_size, to_tensor=True, device=device, iterations=50)
#test_set = crop_to_size(test_set, (280, 352, 184))
test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False)
test_set = None
train_set = None

folder_path = os.path.join(os.getcwd(),'{}_{}'.format('Loss_func', str(num_epochs)))
TverskyAlpha = 0.0
for i in range(9):
	model = CNN(3, base_features=base_features)
	model = nn.DataParallel(model)
	model.to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

	TverskyAlpha += 0.1
	TverskyBeta = round(1 - TverskyAlpha, 1)
	LossFunc = WeightedTverskyLoss((TverskyAlpha, TverskyBeta))
	folder = '{}_{}_{}'.format(LossFunc.get_name(), str(num_epochs), str(TverskyAlpha))
	'Run the CNN'
	losses = []
	epoch_losses = []
	for epoch in range(num_epochs):
		loss_lst = []
		for i, image_set in enumerate(train_loader):
			if epoch <= 50:
				train_loader.UpdateDialPad(-1)
			if epoch == 20:
				train_loader.ToggleDialate()
			image = image_set['data'].to(device)
			labels = image_set['seg'].to(device)
			outputs = model(image)
			loss = LossFunc(outputs, labels)
			loss_lst.append(loss.item())
			losses.append(round(loss.item(), 2))
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			if (i+1) % 1 == 0:
				print(f'epoch {epoch+1} / {num_epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():.4f}')
			ContinuoslySaving(epoch, losses, folder_path, outputs, folder)
			if i >= n_total_steps-1:
				break
		epoch_losses.append(np.mean(loss_lst))

"""test_pred = iter(test_loader)
test_img = test_pred.next()
test_img = crop_to_size(test_img, (test_img['data'].shape[2], test_img['data'].shape[3], test_img['data'].shape[4]))

prediction = model(test_img['data'].to(device))
time = datetime.now().replace(microsecond=0).strftime("kl%H%M%S-%d.%m.%Y")
name = str(num_epochs)+"epoch"+str(TverskyAlpha)+"-"+str(TverskyBeta)+time+".nii.gz"
save_image(torch.squeeze(prediction.detach()).cpu().numpy(), test_imgur.affine, name=name)"""
