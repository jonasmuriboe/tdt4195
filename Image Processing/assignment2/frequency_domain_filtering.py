import numpy as np
from scipy import misc
from scipy import signal
import matplotlib.pyplot as plt
import math

highpass = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
laplacian = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
lowpass = np.array([[1,1,1],[1,1,1],[1,1,1]]) / 9

gaussian1 = np.array([[1,4,1],[4,16,4],[1,4,1]]) / 36
gaussian2 = np.array([[1,4,6,4,1],[4,16,24,16,4],[6,24,36,24,6],[4,16,24,16,4],[1,4,6,4,1]]) / 256
gaussian3 = np.array([[0,0,1,2,1,0,0],[0,3,13,22,13,3,0],[1,13,59,97,59,13,1],[2,22,97,159,97,22,2],[1,13,59,97,59,13,1],[0,3,13,22,13,3,0], [0,0,1,2,1,0,0]]) / 1003

#Makes image greyscale
def greyscale(filename):
    image = misc.imread(filename)
    imgshape = image.shape
    x = imgshape[0]
    y = imgshape[1]
    new_image = np.zeros_like(image)

    for i in range(x):
        for j in range(y):
            rgb = image[i,j]
            new_image[i,j] = (0.2126*rgb[0] + 0.7152*rgb[1] + 0.0722*rgb[2])
    return new_image[..., 0]


def pass_filtering(filename, kernel):
    #Import greyscale image
    if(not misc.imread(filename)[0,0].size > 1):
        image = misc.imread(filename)
    else:
        image = greyscale(filename)

    #Fourier transform the images
    image = np.fft.rfft2(image)
    kernel = np.fft.rfft2(kernel, s =(image.shape[0], image.shape[0]))

    #Matrix multiplication
    image = image * kernel

    #Shift the image back to normal
    image = np.fft.irfft2(image)

    #Show image and save
    plt.imshow(image,cmap='gray')
    plt.show()
    misc.imsave('lowpass.png',image)

def amplitude(filename):
    #Import greyscale image
    if(not misc.imread(filename)[0,0].size > 1):
        image = misc.imread(filename)
    else:
        image = greyscale(filename)

    #Fourier transform
    image = np.fft.fft2(image)
    image = np.fft.fftshift(image)
    image = np.abs(image)

    #Show and save
    plt.imshow(np.log10(image),cmap='gray')
    plt.show()
    misc.imsave('before_lowpass.png', np.log10(image))

def laplacian(filename, kernel):
    #Imports image and saves greyscale of it
    image = greyscale(filename)
    misc.imsave('before_laplacian.png', image)

    #Save original for later
    original = image

    #Convolution
    image = np.fft.fft2(image)
    kernel = np.fft.fft2(kernel, s = (image.shape[0], image.shape[0]))
    image *= kernel
    image = np.fft.ifft2(image)

    #Add convolved image to original
    image += original
    #image = np.abs(image)

    #Show and save image
    plt.imshow(image,cmap='gray')
    plt.show()
    misc.imsave('laplacian.png', image)

def hybrid(filename1, filename2, kernel):
    #Import greyscale image 1
    if(not misc.imread(filename1)[0,0].size > 1):
        image1 = misc.imread(filename1)
    else:
        image1 = greyscale(filename1)

    #Import greyscale image 2
    if(not misc.imread(filename2)[0,0].size > 1):
        image2 = misc.imread(filename2)
    else:
        image2 = greyscale(filename2)

    #Convolves image 1
    image1 = np.fft.rfft2(image1)
    kernel1 = np.fft.rfft2(kernel, s =(image1.shape[0], image1.shape[0]))
    image1 *= kernel1

    #Convolves image 2
    kernel2 = 1.0 - kernel
    image2 = np.fft.rfft2(image2)
    kernel2 = np.fft.rfft2(kernel, s =(image2.shape[0], image1.shape[0]))
    image2 *= (kernel2)

    #Inverse Fourier transform
    #image1 = np.fft.irfft2(image1)
    #image2 = np.fft.irfft2(image2)
    image = image1 + image2
    image = np.fft.irfft2(image)

    #Show and save image
    plt.imshow(image,cmap='gray')
    plt.show()
    misc.imsave('clintbush.png', image)

def downsampling(filename):
    #Import greyscale image
    if(not misc.imread(filename)[0,0].size > 1):
        image = misc.imread(filename)
    else:
        image = greyscale(filename)
    #misc.imsave('before_downsampling.png', image)

    #Creates base for new image
    size = image.shape[0]
    new_image = np.zeros((math.ceil(size / 2),(math.ceil(size / 2))))

    #Get second row and col
    for i in range(0,size,2):
        for j in range(0,size,2):
            new_image[i/2, j/2] = image[i,j]

    #Show and save image
    plt.imshow(new_image,cmap='gray')
    plt.show()
    misc.imsave('downsampled_gaussian3.png', new_image)

def gaussian_smoothing(filename,convolution):
    #Imports greyscale image
    image = greyscale(filename)

    #Convolves image
    padded_convolution = np.pad(convolution, int(image.shape[0] / 2) + 1,'constant')
    image = signal.fftconvolve(padded_convolution,image,mode='full')

    #Crops image
    pad = int(image.shape[0]/4)
    image = image[pad:-pad,pad:-pad]

    #Show and save image
    plt.imshow(image,cmap='gray')
    plt.show()
    misc.imsave('gaussian3.png', image)
misc.imsave('bricks.png', greyscale('images/bricks.tiff'))
#downsampling('images/bricks.tiff')
#gaussian_smoothing('images/lake.tiff',gaussian3)
#downsampling('gaussian3.png')
#amplitude('lake.png')
#amplitude('images/lake.tiff')
#pass_filtering('images/lake.tiff', lowpass)
#hybrid('images/bush.tiff','images/clinton.tiff',lowpass)
