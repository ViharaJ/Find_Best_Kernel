"""
This script finds the optimal sigma and kernel size such that the baseline 
it creates is at least 70% the length of the original contour. 

The script uses the exact contour to create the baseline. Then it calculates
the surface roughness (or average distance) to the STL file. This roughness (or average distance)
value is what's trying to be minimized.


How to use:  
    
OPTIONAL STEP: If you dont' have the exact contour and STL file seperated into two respective 
images, use HSV_CoLour_Picker.py in this repo to create the images.

    1. Change cadImage to CAD image file
    2. Change stlImage to STL image file
    3. Modify kernel, sigma starting values to your liking
        - sigma: sigma value for kernel 
        - kernel: kernel full length (must be odd)
    4. Run
    
    
    
How it works: 
    1. getXYPoints() gets the exact contour, and recreates the contour so there 
    are no duplicate pionts
    2. We enter the loop
    3. We calculate surface roughness two ways
        a. Increase the kernel size by 2 
        b. Increase sigma by 1
    4. The change which yielded the lower surface roughness value will be used in the next iteration
    5. We exit the loop when the baseline creates a contour with less than 70% of 
    the points of the original contour returned in step 1. 
    
"""

import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from matplotlib import colors
from scipy import signal
from sklearn.neighbors import NearestNeighbors
import Module.Functions as fb
import shapely
import math
from scipy.ndimage import gaussian_filter1d

def getXYPoints(image):
    conts, hier = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    allConts = []
    
    for k in conts:
        if(len(k) > len(allConts)):
            allConts = k
        
    allConts = np.squeeze(allConts, axis=1)
    #find starting point of contour
    minIndices = np.where(allConts[:,1] == allConts[:,1].min())[0]
    minPoints = allConts[minIndices]
    minIndx = np.where(minPoints[:,0] == minPoints[:,0].min())[0][0]
    startingCord = allConts[minIndices[minIndx]]
    
    #array to store ordered points
    newOrder = [startingCord]
    
    #delete starting point from contour array (only pairs values in k)
    allConts = np.delete(allConts, minIndx, axis=0)
    
    
    #Find nearest neighbour, stop when next vertex is dist > 4 away
    while(len(allConts) > 1):
        nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(allConts)
        distance, indices = nbrs.kneighbors([newOrder[-1]])
        
        if(distance[0][0] > 4):
            break
        else:
            indices = indices[:,0]
            newOrder.append(allConts[indices[0]])
            allConts = np.delete(allConts, indices[0], axis=0)


    #get unqiue points, maintain order
    _, idx = np.unique(newOrder, axis=0,  return_index=True)
    newOrderIndx = np.sort(idx)
    
    finalOrder = []
    
    for p in newOrderIndx:
        finalOrder.append(newOrder[p])
        
    
    finalOrder = np.array(finalOrder)
    x = np.array(finalOrder[:,0])
    y = np.array(finalOrder[:,1])
    
    return x,y
    
    
def getBaseline(points, s, k):
    x = points[0]
    y = points[1]
    
    gaussFilter = gauss1D(s,k)
    
    #get baseline
    xscipy = signal.convolve(x, gaussFilter, mode='valid')
    yscipy = signal.convolve(y, gaussFilter, mode='valid')
    
    return xscipy, yscipy
    

def gauss1D(size, sigma):
    '''
    size: total length of kernel, must be odd,
    sigma: sigma of gaussian,
    returns: size length array of normalized gaussian kernel
    '''
    size = size+1 if size%2 == 0 else size
    halfLen = (size - 1)/2
    
    filter_range = np.arange(-halfLen, halfLen+1, 1)
    
    gaussFilter  = np.exp(-0.5*(filter_range/sigma)**2)
    gaussFilter = gaussFilter/np.sum(gaussFilter)
   
    return gaussFilter    

def calcSR(gOrder, points, ker, sig):
    distanceE = []
    bx, by = getBaseline(points, ker, sig)
    # bx = gaussian_filter1d(points[0], bestK)
    # by = gaussian_filter1d(points[1], bestK)
    
    dx = np.diff(bx)
    dy = np.diff(by)
 
    polyGon = shapely.geometry.LineString(gOrder)
    
    for j in range(1,len(dx)):
        xs, ys = fb.createNormalLine(bx[j], by[j], dx[j], dy[j])
        
        stack = np.stack((xs,ys), axis=-1)
        line = shapely.geometry.LineString(stack)
    
        if(polyGon.intersects(line) and j > 0):
            #intersection geometry
            interPoints = polyGon.intersection(line)
            
            #intersection point
            mx, my = fb.proccessIntersectionPoint(interPoints, bx[j], by[j])
            
            euD = fb.euclidDist(bx[j], by[j], mx, my)
            distanceE.append(euD)
    
    return np.average(distanceE)
    
cadImage = cv2.imread("CAD_1px.tif", cv2.IMREAD_GRAYSCALE)
stlImage = cv2.imread("STL_1px.tif", cv2.IMREAD_GRAYSCALE)


goal_x, goal_y = getXYPoints(cadImage)

print("Length of contour, unique points only: ", len(goal_x))
x, y = getXYPoints(stlImage)

kernel = 201
sigma = 300

bestDist = 10000000
bestK = 10
bestS = 5



goalOrder = np.stack((goal_x, goal_y), axis=-1)
basex, basey = basex, basey = getBaseline([x,y], kernel, sigma)

while len(basex) >= len(goal_x):    
    sr_sig = calcSR(goalOrder, [x,y], kernel, sigma+1)
    sr_ker = calcSR(goalOrder, [x,y], kernel+2, sigma)
    print(sr_sig, sr_ker)
    dist = min(sr_sig, bestDist)
    
    if sr_sig < sr_ker:
        sigma = sigma + 1
    else:
        kernel = kernel +2 
        
    kernel = kernel + 2 
    basex, basey = getBaseline([x,y], kernel, sigma)
    print("here")
    if dist < bestDist:
        bestK = kernel 
        bestS = sigma
        bestDist = dist
        print("New best kernel: {}  and sigma: {}".format(bestK, bestS))
        plt.title("Kernel {} and sigma {}".format(bestK, bestS))
        plt.plot(x,y, 'b.-', label="Exact contour")
        plt.plot(goal_x, goal_y, 'g.-', label="STL")
        plt.plot(basex, basey, 'r.-', label="Baseline")
        plt.legend()
        plt.show()

    





            
            