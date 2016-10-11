import numpy as np
import math
from scipy import misc

#Converts image into greyscale and normalizes it
def greyscale(filename):
    image = misc.imread(filename)
    imgshape = image.shape
    x = imgshape[0]
    y = imgshape[1]
    new_image = np.zeros((x,y))

    for i in range(0,x):
        for j in range(0,y):
            rgb = np.split(image[i,j],3)
            new_image[i,j] = (0.2126*rgb[0] + 0.7152*rgb[1] + 0.0722*rgb[2]) / 256.0

    return new_image

#Performs a greyscale convolution on an image with the desired convolution
def greyscale_convolution(convolution, filename):

    image = greyscale(filename)
    imgshape = image.shape
    x = imgshape[0]
    y = imgshape[1]
    convolution = np.rot90(convolution, 2)

    convshape = convolution.shape
    padding = int((convshape[0] - 1)/2.0)

    image = np.pad(image,padding,'symmetric')

    new_image = np.zeros((x + 2*padding,y + 2*padding))
    for i in range(padding, x - padding):
        for j in range(padding, y - padding):
            sum = 0.0
            for l in range(-padding, padding + 1):
                for k in range(-padding, padding + 1):
                    sum += image[i + l ,j + k] * convolution[l + padding,k + padding]
            new_image[i,j] = sum
    misc.imsave('gaussian.png', new_image)

    return new_image

#Convolves a colored image with the desired convolution
def color_convolution(convolution):
    #Imports image and gets shape and depth
    image = misc.imread('jelly.tiff')
    new_image = image
    imgshape = image.shape
    x = imgshape[0]
    y = imgshape[1]
    depth = imgshape[2]

    #Convolution not correlation, so gotta rotate that filter
    convolution = np.rot90(convolution, 2)

    #Applies padding
    convshape = convolution.shape
    padding = int((convshape[0] - 1)/2.0)
    image = np.pad(image,padding,'symmetric')

    #Calculates every pixel color by color with M*M*N*N runtime
    for color in range(0,depth-1):
        for i in range(padding, x - padding):
            for j in range(padding, y - padding):
                su = 0.0
                for l in range(-padding, padding + 1):
                    for k in range(-padding, padding + 1):
                        su += new_image[i + l ,j + k,color] * convolution[l + padding,k + padding]
                new_image[i,j,color] = np.uint8(su)
    misc.imsave('box.png', new_image)

#Creates a greyscale gradient of an image by applying the horizontal and vertical filter given in
#the assigment
def greyscale_gradient(filename):
    hori = greyscale_convolution(horizontal, filename)
    verti = greyscale_convolution(vertical, filename)
    new_image = hori
    for i in range(hori.shape[0]):
        for j in range(hori.shape[1]):
            new_image[i,j] = math.fabs(math.sqrt((hori[i,j]**2)+(verti[i,j]**2)))
    misc.imsave('magnitude.png', new_image)




#Defines the different convolutions and starts the program
if __name__=="__main__":
    box = np.array([[1,1,1],[1,1,1],[1,1,1]]) / 9.0
    gaussian = np.array([[1,4,6,4,1],[4,16,24,16,4],[6,24,36,24,6],[4,16,24,16,4],[1,4,6,4,1]]) / 256.0
    vertical = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    horizontal = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    greyscale_gradient('jelly.png')
