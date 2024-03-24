import torch
from torch.utils.data import  DataLoader
from torchvision import datasets, transforms
from VisualizeDataset import plot_tranformed_images, get_image_path_list

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

plot_tranformed_images(image_paths=get_image_path_list(),
                       tranform=data_transform,
                       n=3,
                       seed=42)