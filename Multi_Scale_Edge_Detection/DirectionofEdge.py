import numpy as np
import math
    
def DirectionofEdge(Edge_map, I):
    '''
    Calculate the direction for every edgepoint
    Output:
        Arg: the matrix records the dominant direction for every edge point
            (unit: radian; range: from 0 to 2*pi)
    '''  
    Arg = np.zeros((480, 640))
    for i in range(480):
        for j in range(640):
            if (Edge_map[i, j] == 1):
                # the patch size is 21*21
                patch = I[(i - 11):(i + 10), (j - 11):(j + 10)]
                
                # calculate the gradient for every edge point
                # dx is the x direction (horizontal) gradient, y is the y direction (vertical) grdient
                dx, dy = np.gradient(patch)
                
                # theta is used to record the gradient direction for every point in the patch
                theta = np.zeros((21, 21))
                magnitude = np.zeros((21, 21))
                for k in range(21):
                    for l in range(21):
                        theta[k, l] = math.atan2(dy[k, l],dx[k, l])
                        magnitude[k, l] = np.sqrt(dx[k, l] ** 2 + dy[k, l] ** 2)
                
                # calculate the dominant direction
                # direction = theta[(11 - 3):(11 + 2), (11 - 3):(11 + 2)]
                dominant_direction = np.median(theta)
                Arg[i, j] = dominant_direction
    
    return Arg