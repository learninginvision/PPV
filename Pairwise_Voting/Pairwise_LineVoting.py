import numpy as np
from Pairwise_Voting.Intersect_3D import Intersect_3D
    
def Pairwise_linevoting(Edge_map, Arg_point):
    '''
    Doing pairwise line voting in 3-D space
    Input:
        Edge_map: the edge point of iris image
        Arg_point: the gradient direction of every edge point
    Output:
        Hough_Space: the distribution approximation of the edge appears at some location with certain shape    
    '''
    Location = []
    Location = np.array(Location)
    for y in range(480):
        for x in range(640):
            if (Edge_map[y, x] == 1):
                Location = np.append(Location, [[y, x, Arg_point[y, x]]])
                Location = np.reshape(Location, (-1, 3))
    
    L = len(Location)
    
    # initialization
    Hough_space1 = np.zeros((480, 640, 100))

    for index in range(L):
        for j in np.arange(index-1, L):
            # the gradient direction difference between two edge points
            delta_theta = np.abs(Location[index, 2] - Location[j, 2])
            
            # voting pairs whose gradient direction diferrece is in some range
            if delta_theta < (np.pi / 2 + np.pi / (5)) and delta_theta > (np.pi / 2 - np.pi / (5)):
                y1 = Location[index, 0]
                x1 = Location[index, 1]
                theta1 = Location[index, 2]
                
                y2 = Location[j, 0]
                x2 = Location[j, 1]
                theta2 = Location[j, 2]
                
                # calculate the intersection of the two edge direction line at xyr plane
                Ints1, Ints2, Mid, weight = Intersect_3D(y1, x1, theta1, y2, x2, theta2)
                
                # weighted voting
                if (np.round(Mid[1]) > 0 and np.round(Mid[1]) < 640 and np.round(Mid[0]) > 0 and np.round(Mid[0]) < 480 and np.round(Mid[2]) > 0 and np.round(Mid[2]) < 200):
                    if np.round(Mid[2]) < 100:
                        Hough_space1[int(np.round(Mid[0])), int(np.round(Mid[1])), int(np.round(Mid[2]))] += weight
    
    np.save(file='Hough_space1.npy', arr=Hough_space1)
    del Hough_space1
    
    Hough_space2 = np.zeros((480, 640, 100))
    for index in range(L):
        for j in np.arange(index-1, L):
            # the gradient direction difference between two edge points
            delta_theta = np.abs(Location[index, 2] - Location[j, 2])
            # voting pairs whose gradient direction diferrece is in some range
            if delta_theta < (np.pi / 2 + np.pi / (5)) and delta_theta > (np.pi / 2 - np.pi / (5)):
                y1 = Location[index, 0]
                x1 = Location[index, 1]
                theta1 = Location[index, 2]
                
                y2 = Location[j, 0]
                x2 = Location[j, 1]
                theta2 = Location[j, 2]
                
                # calculate the intersection of the two edge direction line at xyr plane
                Ints1, Ints2, Mid, weight = Intersect_3D(y1, x1, theta1, y2, x2, theta2)
                
                # weighted voting
                if (np.round(Mid[1]) > 0 and np.round(Mid[1]) < 640 and np.round(Mid[0]) > 0 and np.round(Mid[0]) < 480 and np.round(Mid[2]) > 0 and np.round(Mid[2]) < 200):
                    if np.round(Mid[2]) > 100:
                        Hough_space2[int(np.round(Mid[0])), int(np.round(Mid[1])), int(np.round(Mid[2])) - 100] += weight
    
    np.save(file='Hough_space2.npy', arr=Hough_space2)
    del Hough_space2