# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 22:38:03 2020

@author: Namra Rehman
"""
#Sobel Filter implementation in 2D Convolution

#import cv2
import numpy as np
import matplotlib.pyplot as plt
#import time
#from matplotlib import pyplot as plt

#image = cv2.imread('car.jpg', 0)
#image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
sobelx = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype = np.float)
sobely = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype = np.float)
image=np.array([[0,0,1,3,2], [2, 3, 6,0,3], [5,4,2,5,7],[1,2,0,4,4]], dtype = np.float)


row,col =image.shape

sobelxImage = np.zeros([row,col])
sobelyImage = np.zeros((row,col))
sobelGrad = np.zeros((row,col))
thetaImage  = np.zeros([row,col])

#Start time
#timestart = time.clock()

#Surrounds array with 0's on the outside perimeter
image = np.pad(image, (1,1), 'edge')

for i in range(1, row-1):
    for j in range(1, col-1):        
        #Calculate gx and gy using Sobel (horizontal and vertical gradients)
        gx = (sobelx[0][0] * image[i-1][j-1]) + (sobelx[0][1] * image[i-1][j]) + \
             (sobelx[0][2] * image[i-1][j+1]) + (sobelx[1][0] * image[i][j-1]) + \
             (sobelx[1][1] * image[i][j]) + (sobelx[1][2] * image[i][j+1]) + \
             (sobelx[2][0] * image[i+1][j-1]) + (sobelx[2][1] * image[i+1][j]) + \
             (sobelx[2][2] * image[i+1][j+1])

        gy = (sobely[0][0] * image[i-1][j-1]) + (sobely[0][1] * image[i-1][j]) + \
             (sobely[0][2] * image[i-1][j+1]) + (sobely[1][0] * image[i][j-1]) + \
             (sobely[1][1] * image[i][j]) + (sobely[1][2] * image[i][j+1]) + \
             (sobely[2][0] * image[i+1][j-1]) + (sobely[2][1] * image[i+1][j]) + \
             (sobely[2][2] * image[i+1][j+1])     

        sobelxImage[i-1][j-1] = gx
        sobelyImage[i-1][j-1] = gy

        #Calculate the gradient magnitude
        g = np.sqrt(gx * gx + gy * gy)
        sobelGrad[i-1][j-1] = g
        
        thetaImage[i-1][j-1] = np.arctan2(gy, gx)

#End time
#timeend = time.clock() - timestart
#print("2D Convolution with Sobel Filters: ", timeend)

print("Horizontal Sobel")
plt.imshow(sobelxImage)
plt.show()
print("Vertical sobel")
plt.imshow(sobelyImage)
plt.show()
print("Sobel magnitude")
plt.imshow(sobelGrad)
plt.show()
