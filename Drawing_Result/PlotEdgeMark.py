import matplotlib.pyplot as plt
import numpy as np
    
def PlotEdgeMark(I, edgepoint):
    '''
    Plot the mark for every edge point.
    Input:
        I: the input image
        edgepoint: the location of every edge point    
    '''
    plt.imshow(np.uint8(I))
    
    edgepoint = np.array(edgepoint)
    
    plt.plot(edgepoint[:, 0], edgepoint[:, 1], 'o', linewidth=1, markeredgecolor='k', markerfacecolor='w', markersize=6)
    
    plt.show()