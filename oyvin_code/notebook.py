from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from Image_Functions import slicing, crop_images_to_brain, crop_to_size
from datasetModule import Set
from Model import CNN
from Dice_Loss import DiceLoss
import torch
import os
from datetime import datetime
import nibabel as nib
import torch.nn as nn
from matplotlib import pyplot as plt

#hyper parameters
batch_size = 8
learning_rate = 0.01
num_epochs = 50
base_features = 8

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

"Need to specify the local path on computer"
test_path = os.path.join(os.getcwd(), 'Biomedical_Image_Segmentation')
print(test_path)
test_imgur = nib.load(os.path.join(test_path, "Cropped_Task3/crop_sub-233/crop_sub-233_space-T1_desc-masked_T1.nii.gz"))

def save_image(data, affine):
    cropped_img = nib.Nifti1Image(data, affine)
    nib.save(cropped_img, "test.nii.gz")

'Splitting the data into 30% test and 70% training.'
dir_path = os.path.join(os.getcwd(), "Biomedical_Image_Segmentation/Cropped_Task3")
X_train, X_test = train_test_split(Set(dir_path, sub_dir = 'crop_sub-2'), test_size=0.3, random_state=25)
size = (256,288,176)
X_train, X_test = crop_to_size(X_train, size), crop_to_size(X_test,size)

'Load training and test set, batch size may vary'
train_loader, test_loader = DataLoader(X_train, batch_size=batch_size), DataLoader(X_test, batch_size=1)

'Run the CNN'
model = CNN(3,base_features=base_features)
model = nn.DataParallel(model)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
n_total_steps = len(train_loader)
loss_func_2 = DiceLoss()

losses = []
for epoch in range(num_epochs):
    for i, image_set in enumerate(train_loader):
        image = image_set['data'].to(device)
        labels = image_set['seg'].to(device)
        optimizer.zero_grad() #I'd put it further down, it might not make a difference
        outputs = model(image)
        loss = loss_func_2(outputs, labels)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        if (i+1) % 1 == 0:
            print(f'epoch {epoch+1} / {num_epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():.4f}')

plt.plot(losses)
plt.savefig('Losses')
print(losses)

test_pred = iter(test_loader)
test_img = test_pred.next()

prediction = model(test_img['data'].to(device))

save_image(torch.squeeze(prediction.detach()).cpu().numpy(), test_imgur.affine)
time = datetime.now().replace(microsecond=0).strftime("kl%H%M%S-%d.%m.%Y")
print("The saved name of the file was = ", str(time) , ".csv")
torch.save(prediction, str(time) + ".csv")
