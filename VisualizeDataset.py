import os
from pathlib import Path
import random
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

image_path = Path("data")

# create new class for vizualaziation and make different class for random image generation for no code duplicate
def walk_through_dir(dir_path):
    """ Walk through dir_path and return what it contains, its contents."""
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f" Length of dirnames: {len(dirnames)} Directories {len(filenames)} and images in {dirpath}")


# Setup training and testing paths
train_dir = image_path / "train"
test_dir = image_path / "test"

def get_image_path_list():
    image_path_list = list(image_path.glob("*/*/*.jpg"))

    return image_path_list
def get_random_image():
    # get paths */ means get everything from files and last get jpg images
    image_path_list = list(image_path.glob("*/*/*.jpg"))

    # pick a random image path
    random_image_path = random.choice(image_path_list)

    # Get image class from path name(directory name, label)
    image_class = random_image_path.parent.stem
    # print(image_class)

    # Open image
    img = Image.open(random_image_path)

    return [img, image_class]


# Visualize images - 1 -> Get paths, 2 -> Pick random image, 3 -> get image class name
def visualize_using_PIL():

    img = get_random_image()
    # print metadata
    print(f" Image class {img[1]} Image height-width {img[0].height} {img[0].width}")
    img.show()

# visualize_using_pil()

# same steps
def visualize_using_matplotlib():

    img =  img = get_random_image()
    # turn image into array
    image_as_array=np.asarray(img[0])
    plt.figure(figsize=(12,6))
    plt.imshow(image_as_array)
    plt.title(F" Image class {img[1]}  Shape: {image_as_array.shape} -> Height, Width, Color Channels ")
    plt.show()

def plot_tranformed_images(image_paths, tranform, n=5, seed=None):
    """
    Selects random images from a path of images and loads/transforms them,
    then plots the original vs the transformed version, """
    if seed:
        random.seed(seed)
    random_image_paths = random.sample(image_paths, k=n)
    for image_path in random_image_paths:
        with Image.open(image_path) as f:
            fig, ax = plt.subplots(nrows=1, ncols=2) # Do manually, can be used for loop
            ax[0].imshow(f)
            ax[0].set_title(f" Original\n Size: {f.size}")
            ax[0].axis(False)

            # Transform and plot target image (C,H,W) -> (H,W,C)
            transformed_image = tranform(f).permute(1,2,0) # permute for swap color channel RGD from first to last
            ax[1].imshow(transformed_image)
            ax[1].set_title(f" Tranformed \n Shape: {transformed_image.shape}")
            ax[1].axis(False)

            fig.suptitle(f" Class: {image_path.parent.stem}", fontsize=16)
            plt.show()






