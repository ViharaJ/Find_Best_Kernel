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

def addNearestPadding(arr, k):
    #include assertion here
    paddingLen = math.floor(k/2)
    
    newArr = np.full((len(arr)+paddingLen*2),fill_value=arr[0])
    
    newArr[paddingLen:len(arr)+paddingLen] = arr
    
    newArr[len(arr) + paddingLen: -1] = arr[-1]
    
    return newArr
    
    
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


def halfCirlce(x):
    return np.sqrt(200**2- (x-270)**2)

cadImage = cv2.imread("C:/Users/v.jayaweera/Documents/Optimize_Sigma/CAD_1px.tif", cv2.IMREAD_GRAYSCALE)
stlImage = cv2.imread("C:/Users/v.jayaweera/Documents/Optimize_Sigma/STL_1px.tif", cv2.IMREAD_GRAYSCALE)


goal_x, goal_y = getXYPoints(cadImage)
x, y = getXYPoints(stlImage)

kernel = 10
sigma = 200

bestDist = 10000000
bestK = 10
bestS = 5



goalOrder = np.stack((goal_x, goal_y), axis=-1)

for i in range(200,400): #kernel sizes
    distanceE = []
    
    for s in range(300,310): # sigma values        
        bx, by = getBaseline([x,y], i, s)

        
        dx = np.diff(bx)
        dy = np.diff(by)
     
        
        polyGon = shapely.geometry.LineString(goalOrder)
        
        for j in range(1,len(dx)):
            xs, ys = fb.createNormalLine(bx[j], by[j], dx[j], dy[j])
            plt.plot(bx[j], by[j], 'r.-')
            
            stack = np.stack((xs,ys), axis=-1)
            line = shapely.geometry.LineString(stack)
        
            if(polyGon.intersects(line) and j > 0):
                #intersection geometry
                interPoints = polyGon.intersection(line)
                
                #intersection point
                mx, my = fb.proccessIntersectionPoint(interPoints, bx[j], by[j])
                
                euD = fb.euclidDist(bx[j], by[j], mx, my)
                distanceE.append(euD)
         
        
    
    
        if np.average(distanceE) < bestDist:
            bestK = i 
            bestS = s
            bestDist = np.average(distanceE)
            print("New best ", bestK, bestS)
            plt.title("Kernel {} and sigma {}".format(i,s))
            plt.plot(x,y, 'b.-')
            plt.plot(goal_x, goal_y, 'g.-')
            plt.plot(bx, by, 'r.-')
            plt.show()
            
            print(i/len(x))
    





            
            