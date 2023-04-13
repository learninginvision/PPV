import numpy as np
from Pairwise_Voting.Intersect_3D import Intersect_3D

def pairwiseLineVotingNoDirection(Edge_map, Arg_point, Rmax, tao):
    """
    Do pairwise line voting in 3-D space.
    Input:
        Edge_map: the edge point of image
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

    M, N = Edge_map.shape

    Location = np.empty((0, 3))
    for y in range(M):
        for x in range(N):
            if Edge_map[y, x] == 1:
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
                Ints1, Ints2, Mid, weight = Intersect_3D(y1, x1, theta1, y2, x2, theta2, tao)

                # weighted voting
                Mid[2] = abs(Mid[2])
                Mid = np.round(Mid).astype(int)
                if (Mid[1] > 0 - 100 and Mid[1] < M + 100 and Mid[0] > 0 - 100
                        and Mid[0] < N + 100 and Mid[2] < Rmax):
                    # Xb = X[0, :] * 100000000 + X[1, :] * 10000 + X[2, :]
                    # Midb = Mid[0] * 100000000 + Mid[1] * 10000 + Mid[2]
                    Xb = X[0, :] * 10000 + X[1, :] * 100 + X[2, :]
                    Midb = Mid[0] * 10000 + Mid[1] * 100 + Mid[2]
                    
                    index_location = np.where(np.reshape(Xb, (-1, 1), order='F') == Midb)[0]
                    
                    if index_location:
                        length = len(Weight)
                        if index_location[-1]+1 > length:
                            zeros = np.zeros((1, index_location[-1]+1-length))
                            Weight = np.append(Weight, zeros)
                        Weight[index_location] = Weight[index_location] + weight
                    else:
                        Mid = np.reshape(Mid, (-1, 1))
                        X = np.hstack((X, Mid))
                        Weight = np.append(Weight, weight)

    return X, Weight