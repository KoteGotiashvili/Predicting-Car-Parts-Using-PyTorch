import os
import torch
from torch.utils.data import  DataLoader
from torchvision import datasets, transforms
from VisualizeDataset import plot_tranformed_images, get_image_path_list
from torchvision import datasets
from pathlib import Path
from torch.utils.data import DataLoader

image_path = Path("data")
train_dir = image_path / "train"
test_dir = image_path / "test"
BATCH_SIZE = 32
# Transform data into tensors(numerical representation of images)
# docs : https://pytorch.org/vision/stable/transforms.html
# visualized docs: https://pytorch.org/vision/main/auto_examples/transforms/plot_transforms_illustrations.html
data_transform = transforms.Compose([
    # Resize images to 64x64
    transforms.Resize(size=(64, 64)),
    # Flip the images randomly on the horizontal
    transforms.RandomHorizontalFlip(p=0.5),
    # Turn the image into a Tensor (PIL image, or numpy array into Tensor)
    transforms.ToTensor()
])

# plot_tranformed_images(image_paths=get_image_path_list(),
#                        tranform=data_transform,
#                        n=3,
#                        seed=None)

# Load image classification data using ImageFolder
train_data = datasets.ImageFolder(root=train_dir,
                                  transform=data_transform, # transform for data
                                  target_transform=None) # transform for label, infered as target directory, do not need some label, if neccessary can pass target labels
test_data = datasets.ImageFolder(root=test_dir,
                                  transform=data_transform) # transform for data

# Find out details about transformation and image path
# print(train_data, test_data)

class_names = train_data.classes # get classes
class_dict = train_data.class_to_idx # get as dictionary

# Turn loaded images into DataLoaders, turn into iterables, specify batch size
train_dataloader = DataLoader(dataset=train_data,
                              batch_size=BATCH_SIZE,
                              num_workers=os.cpu_count()//2,# How many cpu cors used to load data, all count divide by 2
                              shuffle=True) # shuffle order
test_dataloader=DataLoader(dataset=test_data,
                           batch_size=BATCH_SIZE,
                           num_workers=os.cpu_count()//2,
                           shuffle=False) # Order
# Other forms of transforms -> Data Augmentation -> Change image location, bright, colors, rotation -
# adding slightly modified copies helps to learn better AKA Data Augmentation
# Helps view images from different angles, prespectives

train_transform = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.TrivialAugmentWide(num_magnitude_bins=5), # Selects augmentation types and apply some of them randomly on data
    transforms.ToTensor()
])
test_transform = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor()
])

plot_tranformed_images(image_paths=get_image_path_list(),
                       tranform=train_transform,
                       n=3,
                       seed=None)
