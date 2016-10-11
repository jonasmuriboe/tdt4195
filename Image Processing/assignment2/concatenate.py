import numpy as np
from scipy import misc

def main():
    image1 = misc.imread('images/clinton.tiff')
    image2 = misc.imread('images/bush.tiff')
    misc.imsave('bush.png', image2)
    misc.imsave('clinton.png', image1)
main()
