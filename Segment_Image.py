# Python program to identify
#color in images

# Importing the libraries OpenCV and numpy
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

imagePath="."
dirPictures = os.listdir(imagePath)
acceptedFileTypes = ["jpg", "png", "tif"]


if len(dirPictures) <= 0:
    print("This directory is empty")
else:    
    for i in range(len(dirPictures)):
        image_name = dirPictures[i]
        
        if image_name.split('.')[-1].lower() in acceptedFileTypes:
            #original image
            image = cv2.imread(imagePath+"/" + image_name)
            
            # Convert Image to Image HSV
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
    
            # Defining lower and upper bound HSV values
            #BLUE
            lower = np.array([45, 0, 96])
            upper = np.array([179, 255, 255])
            
            # Defining mask for detecting color
            # Defining mask for detecting color
            mask = cv2.inRange(hsv, lower, upper)
            cv2.imwrite("./STL.jpg", mask)
            
            #orange
            lower = np.array([0, 0, 88])
            upper = np.array([106, 255, 255])
            
            # Defining mask for detecting color
            mask = cv2.inRange(hsv, lower, upper)
              
            cv2.imwrite("./CAD.jpg", mask)