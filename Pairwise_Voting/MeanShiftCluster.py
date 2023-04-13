import numpy as np
import numpy.matlib as matlib
import matplotlib.pyplot as plt
import random
    
def MeanShiftCluster(dataPts, bandWidth, weight):
    '''
    Perform MeanShift Clustering of data using a flat kernel
    Input:
        dataPts: input data, (numDim x numPts)
        bandWidth: bandwidth parameter (scalar)
        plotFlag: display output if 2 or 3-D (logical, True or False)
    Output:
        clustCent: locations of cluster centers (numDim x numClust)
        data2cluster: for every data point which cluster it belongs to (numPts)
        cluster2dataCell: for every cluster which points are in it (numClust)
    Bryan Feldman 02/24/06
    MeanShift first appears in K. Funkunaga and L.D. Hosteler,
    "The Estimation of the Gradient of a Density Function, with Applications in Pattern Recognition"
    '''       
    # Initialize stuff
    numDim, numPts = dataPts.shape
    numClust = 0
    bandSq = bandWidth ** 2
    initPtInds = np.arange(0, numPts)
    maxPos = np.amax(dataPts, axis=1) # biggest size in each dimension (3,)
    minPos = np.amin(dataPts, axis=1) # smallest size in each dimension
    
    boundBox = maxPos - minPos # bounding box size   
    sizeSpace = np.linalg.norm(boundBox) # indicator of size of data space
    
    stopThresh = 0.001 * bandWidth # when mean has converged
    
    # center of clust 
    clustCent = np.zeros((numDim, 0))
    
    # track if a point has been seen already
    beenVisitedFlag = np.zeros((1, numPts), 'uint8')
    
    # number of points to possibly use as initilization points
    numInitPts = numPts
    
    # resolve conflicts on cluster membership   
    clusterVotes = np.zeros((1, numPts), 'uint16')
    
    while numInitPts:
        # Pick a random seed point
        tempInd = np.ceil((numInitPts - 1e-06) * random.random())
        
        stInd = initPtInds[int(tempInd)-1] # use this point as start of mean
        myMean = dataPts[:, stInd] # intialize mean to this points location
        
        # Points that will get added to this cluster
        myMembers = []
        myMembers = np.array(myMembers)
        
        # used to resolve conflicts on cluster membership
        thisClusterVotes = np.zeros((1, numPts), 'uint16')
        
        while 1: # loop untill convergence

            # dist squared from mean to all points still active
            # (1 x numPts)
            sqDistToAll = sum((matlib.repmat(np.reshape(myMean, (-1, 1)), 1, numPts) - dataPts) ** 2)            
            # points within bandWidth
            inInds = np.reshape(np.where(sqDistToAll < bandSq), -1)
            # add a vote for all the in points belonging to this cluster
            thisClusterVotes[0, inInds] = thisClusterVotes[0, inInds] + 1
            
            # save the old mean
            myOldMean = myMean 
            # compute the new mean
            myMean = dataPts[:, inInds] @ np.reshape(weight[inInds], (-1, 1)) / sum(weight[inInds])
            
            # add any point within bandWidth to the cluster
            myMembers = np.append(myMembers, inInds)
            # mark that these points have been visited
            beenVisitedFlag[0, np.int0(myMembers)] = 1           
                    
            # if mean doesn't move much, stop this cluster
            if np.linalg.norm(myMean - myOldMean) < stopThresh:
                
                # check for merge posibilities
                mergeWith = 0
                for cN in range(numClust):
                    
                    # distance from posible new clust max to old clust max
                    a1, b1 = clustCent.shape
                    if cN+1 > b1:
                        clust = np.pad(clust, ((0, 0), (0, cN+1-b1)), 'constant', constant_values=(0, 0))
                    distToOther = np.linalg.norm(myMean - clustCent[:, cN])
                    
                    # if it's within bandwidth/2, merge new and old
                    if distToOther < bandWidth / 2:
                        mergeWith = cN
                        break
                    
                # something to merge
                if mergeWith > 0:
                    
                    # record the max as the mean of the two merged (I know biased twoards new ones)
                    a2, b2 = clustCent.shape
                    if mergeWith+1 > b2:
                        clust = np.pad(clust, ((0, 0), (0, mergeWith+1-b2)), 'constant', constant_values=(0, 0))                    
                    clustCent[:, mergeWith] = 0.5 * (myMean + clustCent[:, mergeWith])
                    
                    # add these votes to the merged cluster
                    c1, d1 = clusterVotes.shape
                    if mergeWith+1 > c1:
                        clusterVotes = np.pad(clusterVotes, ((0, mergeWith+1-c1), (0, 0)), 'constant', constant_values=(0, 0))
                    clusterVotes[mergeWith, :] = clusterVotes[mergeWith, :] + thisClusterVotes
                
                # it's a new cluster
                else:                    
                    
                    a3, b3 = clustCent.shape
                    if numClust+1 > b3:
                        clustCent = np.pad(clustCent, ((0, 0), (0, numClust+1-b3)), 'constant', constant_values=(0, 0))                                      
                    # record the mean
                    clustCent[:, numClust] = np.reshape(myMean, -1)
                    
                    c2, d2 = clusterVotes.shape
                    if numClust+1 > c2:
                        clusterVotes = np.pad(clusterVotes, ((0, numClust+1-c2), (0, 0)), 'constant', constant_values=(0, 0))            
                    clusterVotes[numClust, :] = thisClusterVotes[0, :] 
                    
                    # increment clusters
                    numClust = numClust + 1  
                break
        
        # we can initialize with any of the points not yet visited
        initPtInds = np.where(beenVisitedFlag == 0)[1]
        # number of active points in set
        numInitPts = len(initPtInds)

    # a point belongs to the cluster with the most votes
    val = np.max(clusterVotes, axis=0)
    data2cluster = np.argmax(clusterVotes, axis=0)
    
    return clustCent, data2cluster