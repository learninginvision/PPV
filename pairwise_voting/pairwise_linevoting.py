import numpy as np
from pairwise_voting.intersect_3D import intersect_3D


class PairwiseLineVoting():
    
    def pairwise_linevoting_nodirection(self, edge_map, Arg_point, Rmax, tao):
        """
        Do pairwise line voting in 3-D space.
        Input:
            edge_map: the edge point of image
            Arg_point: the gradient direction of every edge point
            Rmax: the maximum radius we need to search for
            tao: the threshold for determining valid or invalid vote
        Output:
            X: the distribution approximation of the edge appears at some location with certain shape
            Weight: the weight of each distribution
        """
        # Initialize the output
        X = np.reshape(np.array([0, 0, 0]), (-1, 1))
        Weight = np.array([0])

        M, N = edge_map.shape

        Location = np.empty((0, 3))
        for y in range(M):
            for x in range(N):
                if edge_map[y, x] == 1:
                    Location = np.append(Location, [[y, x, Arg_point[y, x]]])
                    Location = np.reshape(Location, (-1, 3))

        L = len(Location)
        
        for index in range(L):
            for j in range(index-1, L):
                # the gradient direction difference between two edge points
                delta_theta = abs(Location[index, 2] - Location[j, 2])

                # voting pairs whose gradient direction difference is in some range
                if delta_theta < (np.pi - np.pi / (5)) and delta_theta > (np.pi / (5)):

                    y1 = Location[index, 0]
                    x1 = Location[index, 1]
                    theta1 = Location[index, 2]

                    y2 = Location[j, 0]
                    x2 = Location[j, 1]
                    theta2 = Location[j, 2]

                    # calculate the intersection of the two edge direction line
                    ints1, ints2, mid, weight = intersect_3D(y1, x1, theta1, y2, x2, theta2, tao)

                    # weighted voting
                    mid[2] = abs(mid[2])
                    mid = np.round(mid).astype(int)
                    if (mid[1] > 0 - 100 and mid[1] < M + 100 and mid[0] > 0 - 100
                            and mid[0] < N + 100 and mid[2] < Rmax):
                        # Xb = X[0, :] * 100000000 + X[1, :] * 10000 + X[2, :]
                        # Midb = mid[0] * 100000000 + mid[1] * 10000 + mid[2]
                        Xb = X[0, :] * 10000 + X[1, :] * 100 + X[2, :]
                        Midb = mid[0] * 10000 + mid[1] * 100 + mid[2]
                        
                        index_location = np.where(np.reshape(Xb, (-1, 1), order='F') == Midb)[0]
                        
                        if index_location:
                            length = len(Weight)
                            if index_location[-1]+1 > length:
                                zeros = np.zeros((1, index_location[-1]+1-length))
                                Weight = np.append(Weight, zeros)
                            Weight[index_location] = Weight[index_location] + weight
                        else:
                            mid = np.reshape(mid, (-1, 1))
                            X = np.hstack((X, mid))
                            Weight = np.append(Weight, weight)

        return X, Weight
    
    
    def pairwise_linevoting(self, edge_map, Arg_point):
        '''
        Doing pairwise line voting in 3-D space
        Input:
            edge_map: the edge point of iris image
            Arg_point: the gradient direction of every edge point
        Output:
            Hough_Space: the distribution approximation of the edge appears at some location with certain shape    
        '''
        Location = []
        Location = np.array(Location)
        for y in range(480):
            for x in range(640):
                if (edge_map[y, x] == 1):
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
                    ints1, ints2, mid, weight = intersect_3D(y1, x1, theta1, y2, x2, theta2)
                    
                    # weighted voting
                    if (np.round(mid[1]) > 0 and np.round(mid[1]) < 640 and np.round(mid[0]) > 0 and np.round(mid[0]) < 480 and np.round(mid[2]) > 0 and np.round(mid[2]) < 200):
                        if np.round(mid[2]) < 100:
                            Hough_space1[int(np.round(mid[0])), int(np.round(mid[1])), int(np.round(mid[2]))] += weight
        
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
                    Ints1, Ints2, mid, weight = intersect_3D(y1, x1, theta1, y2, x2, theta2)
                    
                    # weighted voting
                    if (np.round(mid[1]) > 0 and np.round(mid[1]) < 640 and np.round(mid[0]) > 0 and np.round(mid[0]) < 480 and np.round(mid[2]) > 0 and np.round(mid[2]) < 200):
                        if np.round(mid[2]) > 100:
                            Hough_space2[int(np.round(mid[0])), int(np.round(mid[1])), int(np.round(mid[2])) - 100] += weight
        
        np.save(file='Hough_space2.npy', arr=Hough_space2)
        del Hough_space2