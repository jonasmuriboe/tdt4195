import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
import math

#Seed points from noise.tiff
noise_seed = [[127,204],[150,229],[45,106],[55,335],[49,176],[91,140],[91,250],[23,270],[38,383],[106,67],[176,54],[238,35],[286,72],[303,180],[245,242],[314,260],[278,332],[203,338],[274,401],[134,396]]

def algorithm_1(filename,threshold,deltaT):
    #Imports image and sets defaults
    image = misc.imread(filename, mode='L')
    threshold_prev = 0
    x = image.shape[0]
    y = image.shape[1]

    #While the difference between threshold and previous threshold
    #is bigger than deltaT keep going. Directly implements function
    #described in the assignment
    while math.fabs(threshold - threshold_prev) > deltaT:
        set1 = image[image > threshold]
        set2 = image[image <= threshold]
        mean1 = np.mean(set1)
        mean2 = np.mean(set2)
        threshold_prev = threshold
        threshold = 0.5 * (mean1 + mean2)

    #Prints the threshold of the image and then segments it by calling the basic_segment function with the image and threshold
    print("The threshold for this image is", threshold)
    basic_segment(image, threshold)

#Creates a black and white image from the input and based on the threshold
def basic_segment(image, threshold):
    new_image = np.empty_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i,j] > threshold:
                new_image[i,j] = 255
            else:
                new_image[i,j] = 0

    misc.imsave('weld.png',new_image)

def segment_growing(filename, threshold,seedPoints):
    #imports image,creates empty lists and a black image the size of the input
    image = misc.imread(filename)
    candidatePoints = []
    visitedPoints = []
    new_image = np.zeros_like(image)

    #Goes through all the seed points provided
    for seed in seedPoints:
        candidatePoints.append(seed)

        #As long as there are candidate points it will keep going
        while len(candidatePoints) > 0:
            currentPoint = candidatePoints.pop(0)

            #Checks if the point has been visited before
            if not currentPoint in visitedPoints:
                visitedPoints.append(currentPoint)
                row = currentPoint[0]
                col = currentPoint[1]

                #Checks if the current point is within the threshold
                if image[row,col] > threshold:
                    new_image[row,col] = image[row,col]

                    #Uses 4-connectedness anc checks if the four points are within the threshold and adds them to candidate points if they are
                    for i in ([row,col-1],[row-1,col],[row,col+1],[row+1,col]):
                        if image[row,col] > threshold and not (i in visitedPoints):
                                candidatePoints.append(i)
                                new_image[i[0],i[1]] = image[i[0],i[1]]

    #Saves image
    misc.imsave("noisy.png",new_image)

if __name__ == '__main__':
    #algorithm_1('images/weld.tif', 128,10)
    #segment_growing('images/noisy.tiff',[124],noise_seed)
    segment_growing('images/noisy.tiff',[124],[[127,204]])
