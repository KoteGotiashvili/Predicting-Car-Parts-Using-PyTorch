import os
import torch
from torch.utils.data import  DataLoader
from torchvision import datasets, transforms
from VisualizeDataset import plot_tranformed_images, get_image_path_list
from torchvision import datasets
from pathlib import Path
from torch.utils.data import DataLoader
from torch import nn
from torchinfo import summary
from TrainTest import train
from timeit import default_timer as timer
from ultralytics import YOLO

image_path = Path("data")
train_dir = image_path / "train"
test_dir = image_path / "test"
BATCH_SIZE = 16
NUM_EPOCHS = 100
device = "cuda" if torch.cuda.is_available() else "cpu"
# Transform data into tensors(numerical representation of images)
# docs : https://pytorch.org/vision/stable/transforms.html
# visualized docs: https://pytorch.org/vision/main/auto_examples/transforms/plot_transforms_illustrations.html
data_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.TrivialAugmentWide(num_magnitude_bins=5),
    transforms.ToTensor(),
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
                              # num_workers=os.cpu_count()//2,# How many cpu cors used to load data, all count divide by 2
                              shuffle=True) # shuffle order
test_dataloader=DataLoader(dataset=test_data,
                           batch_size=BATCH_SIZE,
                           shuffle=False) # Order
# Other forms of transforms -> Data Augmentation -> Change image location, bright, colors, rotation -
# adding slightly modified copies helps to learn better AKA Data Augmentation
# Helps view images from different angles, prespectives

# train_transform = transforms.Compose([
#     transforms.Resize(size=(224, 224)),
#     transforms.TrivialAugmentWide(num_magnitude_bins=5), # Selects augmentation types and apply some of them randomly on data
#     transforms.ToTensor()
# ])
# test_transform = transforms.Compose([
#     transforms.Resize(size=(224, 224)),
#     transforms.ToTensor()
# ])

 # Visualization
# plot_tranformed_images(image_paths=get_image_path_list(),
#                        tranform=train_transform,
#                        n=3,
#                        seed=None)


# Start Building Model
class CarPartsClassification(nn.Module):

    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=5,  # how big is the square that's going over the image?
                      stride=2,  # default
                      padding=1),
            # options = "valid" (no padding) or "same" (output has same shape as input) or int for specific number
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=5,
                      stride=2,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=5,
                         stride=2)  # default stride value is same as kernel_size
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Where did this in_features shape come from?
            # It's because each layer of our network compresses and changes the shape of our inputs data.
            nn.Linear(in_features=hidden_units*9,
                      out_features=output_shape)
        )
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        x = self.dropout(x)
        # print(x.shape)
        return x
        # return self.classifier(self.conv_block_2(self.conv_block_1(x))) # <- leverage the benefits of operator fusion


torch.manual_seed(42)
model = CarPartsClassification(input_shape=3, # number of color channels in image
                                 hidden_units=64,
                                 output_shape=len(class_names)).to(device)

# # Test model, dummy forward pass
# image_batch, label_batch = next(iter(train_dataloader))
# image_batch, label_batch = image_batch.to(device), label_batch.to(device)
# print(image_batch.shape, label_batch.shape)
# # Try forward pass
# model(image_batch)
# summary(model, input_size=[1,3,64,64])




# Create Loss function and Optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(),
                             lr = 0.01)

# create start time
start_time = timer()
torch.manual_seed(42)
torch.cuda.manual_seed(42)
# Train model
model_results = train(model=model,
                      train_dataloader=train_dataloader,
                      test_dataloader=test_dataloader,
                      optimizer=optimizer,
                      loss_fn=loss_fn,
                      epochs=NUM_EPOCHS,
                      device=device)
# End timer and print time it take
end_time = timer()
print(f" Total time {end_time-start_time:.3f} seconds")

