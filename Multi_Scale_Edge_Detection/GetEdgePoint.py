import numpy as np
    
def GetEdgePoint(Edge_map):
    '''
    Extract edge points' position in image domain
    Output:
        Edge_point_location: the x and y coordinate of edge points
    '''     
    Edge_point_location = np.empty((0, 2))
    x = 0
    M, N = Edge_map.shape
    for i in range(M):
        for j in range(N):
            if (Edge_map[i, j] == 1):
                x += 1
                Edge_point_location = np.reshape(np.append(Edge_point_location, [[j, i]]), (x, 2))
    
    return Edge_point_location