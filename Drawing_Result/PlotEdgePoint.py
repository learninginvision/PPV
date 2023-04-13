import numpy as np
import matplotlib.pyplot as plt
    
def PlotEdgePoint(EdgePoint, I):
    '''
    Plot the edge point map.
    Input:
        EdgePoint: the input edge point
        I: the input image    
    '''
    M, N = I[:, :, 0].shape
    plt.imshow(255 * np.ones((M, N)))

    plt.plot(EdgePoint[:, 0], EdgePoint[:, 1], 'o', markeredgecolor='k', markerfacecolor='b', markersize=6)
        
    plt.show()