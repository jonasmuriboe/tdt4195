from scipy import misc

image = misc.imread("images/noisy.tiff")
misc.imsave('originalnoisy.png',image)
