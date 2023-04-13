def Edgemap2Edgepoint(Edge_map):
    '''
    Extract edge points' position in image domain.
    Input:
        Edge_map: The Edge map of image
    Output:
        Edge_point: the x and y coordinate of edge points
    '''
    Edge_point_location = []
    M, N = Edge_map.shape
    
    for i in range(M):
        for j in range(N):
            if (Edge_map[i,j] == 1):
                Edge_point_location.append([j, i])
    return Edge_point_location