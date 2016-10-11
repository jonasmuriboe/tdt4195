import numpy as np
from scipy import misc
import math

#Opens image and extracts height and width
image = misc.imread('greyscale.tiff')
s = image.shape
x = s[0]
y = s[1]

#The first, simple intensity transformation
def basic_gamma():
    new_image = np.zeros((x,y))
    for i in range(0,x):
        for j in range(0,y):
            new_image[i,j] = 255 - image[i,j]
    misc.imsave('gamma.png', new_image)

#This function normalizes the image and saves it in the folder the script is run from
def normalize():
    new_image = np.zeros((x,y))
    for i in range(0,x):
        for j in range(0,y):
            new_image[i,j] = float(image[i,j])/255
    misc.imsave('normalized.png', new_image)

#Applies a gamma transformation to the power of the coefficient
def greyscale_level(coefficient):
    image = misc.imread('normalized.png')
    s = image.shape
    x = s[0]
    y = s[1]
    new_image = np.zeros((x,y))
    for i in range(0,x):
        for j in range(0,y):
            new_image[i,j] = math.pow(image[i,j], coefficient)
    misc.imsave(('gammafied'+str(coefficient)+'.png'), new_image)

if __name__ == '__main__':
    basic_gamma()
