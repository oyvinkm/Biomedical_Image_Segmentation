Explanation of the different files: 

Preprocessing:
    - cropping.py: Does not really do anything at the moment, cropping functionality.
    - DataLoaderModule.py: Module for self implemented DataLoader
    - DataLoader3D.py: Main dataloader, which creates patches
    - preprocessing.py: Tried to resample images, but we diteriated from this due to information loss.

Dataset Module: Loads, the data
Image_functions: collections of functions to work with images; show images, crop, create masks etc. 
Model:
    - Model_Aske