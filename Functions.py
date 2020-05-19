import matplotlib.pyplot as plt
import scipy.io as sio
from ipywidgets import Dropdown
from Classes import Image

def createNamesDropdownMenu():
    names = sio.loadmat('Names.mat')['MaskDatabaseNames']
    tags = []
    for i in range(names.shape[1]):
        tags.append((names[0, i][0], i + 1))
    return Dropdown(options=tags, value=1, description='Region:')

def blendImages(z, alpha, im1, im2, color=False):

    if color is True:
        img = (1.0 - alpha)*Image.convertToRGB(im1[:, :, z]) + alpha*Image.convertToRGB(im2[:, :, z], color='red')
        plt.figure(figsize=(15,15))
        plt.imshow(img)
    else:
        img = (1.0 - alpha) * im1[:, :, z]+ alpha * im2[:, :, z]
        plt.figure(figsize=(15, 15))
        plt.imshow(img, cmap=plt.cm.Greys_r)
    plt.axis('off')
    plt.show()

def displayImages(fixedZ, movingZ, fixed, moving):
    # Create a figure with two subplots and the specified size.
    plt.subplots(1, 2, figsize=(15, 12))

    # Draw the fixed image in the first subplot.
    plt.subplot(1, 2, 1)
    plt.imshow(fixed[:, :, fixedZ], cmap=plt.cm.Greys_r)
    plt.title('Fixed Image')
    plt.axis('off')

    # Draw the moving image in the second subplot.
    plt.subplot(1, 2, 2)
    plt.imshow(moving[:, :, movingZ], cmap=plt.cm.Greys_r)
    plt.title('Moving Image')
    plt.axis('off')

    plt.show()

def displayImagesSamePlane(Z, fixed, moving):
    # Create a figure with two subplots and the specified size.
    plt.subplots(1, 2, figsize=(15, 12))

    # Draw the fixed image in the first subplot.
    plt.subplot(1, 2, 1)
    plt.imshow(fixed[:, :, Z], cmap=plt.cm.Greys_r)
    plt.title('Fixed Image')
    plt.axis('off')

    # Draw the moving image in the second subplot.
    plt.subplot(1, 2, 2)
    plt.imshow(moving[:, :, Z], cmap=plt.cm.Greys_r)
    plt.title('Moving Image')
    plt.axis('off')

    plt.show()

def displayStack(stack, z):
    plt.figure(figsize=(15, 15))
    plt.imshow(stack[:, :, z], cmap=plt.cm.Greys_r)
    plt.axis('off')
    plt.show()
