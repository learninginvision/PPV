import numpy as np


def edgemap2edgepoint(edge_map):
    '''
    Extract edge points' position in image domain.
    Input:
        edge_map: The Edge map of image
    Output:
        edge_point_location: the x and y coordinate of edge points
    '''
    edge_point_location = []
    M, N = edge_map.shape
    
    for i in range(M):
        for j in range(N):
            if (edge_map[i,j] == 1):
                edge_point_location.append([j, i])
    return edge_point_location

  
def get_edgepoint(edge_map):
    '''
    Extract edge points' position in image domain
    Output:
        edge_point_location: the x and y coordinate of edge points
    '''     
    edge_point_location = np.empty((0, 2))
    x = 0
    M, N = edge_map.shape
    for i in range(M):
        for j in range(N):
            if (edge_map[i, j] == 1):
                x += 1
                edge_point_location = np.reshape(np.append(edge_point_location, [[j, i]]), (x, 2))
    
    return edge_point_location