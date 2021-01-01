# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 19:29:49 2020

@author: Namra Rehman
"""

import imageio as io
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

def checkWeak(edge,weakEdges,weakIndex):
    for i in range(0,1):
        for j in range(0,weakIndex):
            if(weakEdges[i,j]==edge):
                return 1
            else:
                return 0
    
def canny():
    img = io.imread('original.png')
    #img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)    #convert image to gray if you have color image.
    
    print("Original")
    plt.imshow(img,cmap="gray")
    plt.show()
    
    #step1  ================================================================
    gussianImage = cv.GaussianBlur(img,(5,5),cv.BORDER_DEFAULT)
    print("Gussian")
    plt.imshow(gussianImage,cmap="gray")
    plt.show()

    #step2  ================================================================
    sobelx = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype = np.float)
    sobely = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype = np.float)
    
    row,col =img.shape
    #row = img.shape[0] #row 
    #col = img.shape[1] #column

    sobelxImage = np.zeros([row,col])
    sobelyImage  = np.zeros([row,col])
    sobelMagImage  = np.zeros([row,col])
    thetaImage  = np.zeros([row,col])
    
    #padding 0's 
    gussianImage1 = np.pad(gussianImage, (1,1), 'edge')
    
    for i in range(1, row-1):
        for j in range(1, col-1):   
            gx = (sobelx[0][0] * gussianImage1[i-1][j-1]) + (sobelx[0][1] * gussianImage1[i-1][j]) +\
                 (sobelx[0][2] * gussianImage1[i-1][j+1]) + (sobelx[1][0] * gussianImage1[i][j-1]) + \
                 (sobelx[1][1] * gussianImage1[i][j]) + (sobelx[1][2] * gussianImage1[i][j+1]) + \
                 (sobelx[2][0] * gussianImage1[i+1][j-1]) + (sobelx[2][1] * gussianImage1[i+1][j]) +\
                 (sobelx[2][2] * gussianImage1[i+1][j+1])  
                 
            gy = (sobely[0][0] * gussianImage1[i-1][j-1]) + (sobely[0][1] * gussianImage1[i-1][j]) + \
                 (sobely[0][2] * gussianImage1[i-1][j+1]) + (sobely[1][0] * gussianImage1[i][j-1]) + \
                 (sobely[1][1] * gussianImage1[i][j]) + (sobely[1][2] * gussianImage1[i][j+1]) + \
                 (sobely[2][0] * gussianImage1[i+1][j-1]) + (sobely[2][1] * gussianImage1[i+1][j]) + \
                 (sobely[2][2] * gussianImage1[i+1][j+1]) 
                 
            sobelxImage[i-1][j-1] = gx
            sobelyImage[i-1][j-1] = gy
            
            #Calculate the gradient magnitude
            g = np.sqrt(gx * gx + gy * gy)
            sobelMagImage[i-1][j-1] = g
            
            thetaImage[i-1][j-1] = np.arctan2(gy, gx)
    
    print("Horizontal Sobel")
    plt.imshow(sobelxImage,cmap="gray")
    plt.show()
    print("Vertical sobel")
    plt.imshow(sobelyImage,cmap="gray")
    plt.show()
    print("Sobel magnitude")
    plt.imshow(sobelMagImage,cmap="gray")
    plt.show() 
    
    #step3  ================================================================
    nonMaximaImage  = np.zeros([row,col])
    for i in range(0,row-1):
        for j in range(0,col-1):  
            if (thetaImage[i,j]>=-22.5 and thetaImage[i,j]<=22.5) or (thetaImage[i,j]<-157.5 and thetaImage[i,j]>=-202):
                if (sobelMagImage[i,j] >= sobelMagImage[i,j+1]) and (sobelMagImage[i,j] >= sobelMagImage[i,j-1]):
                    nonMaximaImage[i,j]= sobelMagImage[i,j]
                else:
                    nonMaximaImage[i,j]=0
                
            elif (thetaImage[i,j]>=22.5 and thetaImage[i,j]<=67.5) or (thetaImage[i,j]<-112.5 and thetaImage[i,j]>=-157.5):
                if (sobelMagImage[i,j] >= sobelMagImage[i+1,j+1]) and (sobelMagImage[i,j] >= sobelMagImage[i-1,j-1]):
                    nonMaximaImage[i,j]= sobelMagImage[i,j]
                else:
                    nonMaximaImage[i,j]=0
               
            elif (thetaImage[i,j]>=67.5 and thetaImage[i,j]<=112.5) or(thetaImage[i,j]<-67.5 and thetaImage[i,j]>=-112.5):
                if (sobelMagImage[i,j] >= sobelMagImage[i+1,j]) and (sobelMagImage[i,j] >= sobelMagImage[i-1,j]):
                    nonMaximaImage[i,j]= sobelMagImage[i,j]
                else:
                    nonMaximaImage[i,j]=0
                
            elif (thetaImage[i,j]>=112.5 and thetaImage[i,j]<=157.5) or (thetaImage[i,j]<-22.5 and thetaImage[i,j]>=-67.5):
                if (sobelMagImage[i,j] >= sobelMagImage[i+1,j-1]) and (sobelMagImage[i,j] >= sobelMagImage[i-1,j+1]):
                    nonMaximaImage[i,j]= sobelMagImage[i,j]
                else:
                    nonMaximaImage[i,j]=0
             
    print("After non maxima suppression")
    plt.imshow(nonMaximaImage,cmap="gray")
    plt.show()
    
    #step4  ================================================================
    highThreshold = nonMaximaImage.max()*0.09;
    lowThreshold = highThreshold*0.05;
    
    strongEdgesRow = np.zeros([1,row*col])
    strongEdgesCol = np.zeros([1,row*col])
    weakEdgesRow = np.zeros([1,row*col])
    weakEdgesCol = np.zeros([1,row*col])
    strongIndex = 0
    weakIndex = 0
    
    weakEdges = np.zeros([1,row*col])
    for x in range(0,row):
        for y in range(0,col):
            #print("x",x)           #debugging
            #print("y",y)
            #print("strong",strongIndex)
            #print("weak",weakIndex)
            
            if (nonMaximaImage[x,y] > highThreshold):
                nonMaximaImage[x,y] = 1
                strongEdgesRow[0,strongIndex] = x
                strongEdgesCol[0,strongIndex] = y
                strongIndex = strongIndex + 1
                
            elif (nonMaximaImage[x,y] < lowThreshold):
                nonMaximaImage[x,y] = 0
            else:                        
                weakEdgesRow[0,weakIndex] = x
                weakEdgesCol[0,weakIndex] = y
                weakEdges[0,weakIndex]=nonMaximaImage[x,y]
                weakIndex = weakIndex + 1
    
    print("After double thresholding")
    plt.imshow(nonMaximaImage,cmap="gray")
    plt.show()    
    #step5  ================================================================
    hysteresisImage  = np.zeros([row,col])
    hysteresisImage=nonMaximaImage
    for i in range(row):
        for j in range(col):
            #check if current edge is weak 
            if (checkWeak(nonMaximaImage[i,j],weakEdges,weakIndex)==1):
                
                #if weak , then check if it has any strong around
                if ((nonMaximaImage[i+1, j-1] == 1) or (nonMaximaImage[i+1, j] == 1) or (nonMaximaImage[i+1, j+1] == 1) 
                or (nonMaximaImage[i, j-1] == 1) or (nonMaximaImage[i, j+1] == 1)
                or (nonMaximaImage[i-1, j-1] == 1) or (nonMaximaImage[i-1, j] == 1) 
                or (nonMaximaImage[i-1, j+1] == 1)):
                    
                    hysteresisImage[i, j] = 1
                else:
                    hysteresisImage[i, j] = 0
    
    print("After hysteresis thresholding")
    plt.imshow(hysteresisImage,cmap="gray")
    plt.show()
    
    
    #debugging purpose
    #hysteresisImage=hysteresisImage-nonMaximaImage
    #plt.imshow(hysteresisImage)
    #plt.show()
    
   
 
#driver 
canny()
