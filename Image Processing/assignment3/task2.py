from scipy import misc, ndimage
import matplotlib.pyplot as plt
import numpy as np

def task2a(): # Opening/closing
    img = misc.imread("images/noisy.tiff") # Loading image
    spanishdisc = np.array([[0,1,0],[1,1,1],[0,1,0]]) # Structuring element

    # Open/close
    img = ndimage.binary_closing(img, structure=spanishdisc, iterations=6)
    img = ndimage.binary_opening(img, structure=spanishdisc, iterations=8)

    # Display result
    plt.imshow(img, cmap="gray")
    plt.show()

    # Save
    misc.imsave("images/noisefree.png", img)


def task2b(): # Distance transform
    # Loading and converting to binary image
    img = misc.imread("images/noisefree.png")//255
    structure = np.ones((3,3)) # Structuring element
    newimg = np.zeros((img.shape[0], img.shape[1])) # New image
    while not (sum(img.flatten()) == 0): # While not eroded to zero

        # Save the eroded image
        img = ndimage.binary_erosion(img, structure=structure, iterations=1)
        newimg += img # Add erosion to empty image
    plt.imshow(newimg, cmap="gray") # Display result
    plt.show()
    misc.imsave("images/distancetransform.png", newimg) # Save image

def task2c(): # Boundary
    # Loading and converting to binary image
    img = misc.imread("images/noisefree.png")/255
    structure = np.ones((3,3)) # Structuring element
    newimg = np.zeros((img.shape[0], img.shape[1])) # New image

    # Boundary extraction
    newimg = img - ndimage.morphology.binary_erosion(img, structure=structure, iterations=1)
    plt.imshow(newimg, cmap="gray") # Display result
    plt.show()

