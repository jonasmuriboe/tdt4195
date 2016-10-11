import numpy as np
from scipy import misc
import matplotlib.pyplot as plt

#Imports image and extracts the height and width
image = misc.imread('color.jpg')
s = image.shape
x = s[0]
y = s[1]

#The first of the equations
def equation_1():
    new_image = np.zeros((x,y))

    for i in range(0,x):
        for j in range(0,y):
            rgb = np.split(image[i,j],3)
            new_image[i,j] = (rgb[0] + rgb[1] + rgb[2]) / 3

    misc.imsave('equation_1.png', new_image)

#The second equation
def equation_2():
    new_image = np.zeros((x,y))

    for i in range(0,x):
        for j in range(0,y):
            rgb = np.split(image[i,j],3)
            new_image[i,j] = (0.2126*rgb[0] + 0.7152*rgb[1] + 0.0722*rgb[2])

    misc.imsave('equation_2.png', new_image)

#
if __name__ == '__main__':
    equation_1()
    equation_2()
