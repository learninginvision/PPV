import matplotlib.pyplot as plt
import numpy as np
    
def PlotCircle(I, xc, yc, r, color):
    '''
    Show the image and plot the circle on the image.
    Input:
        I: the image which need to be shown
        xc: the x coordinate of center position
        yc: the y coordinate of center position
        r: the radius of the circle
        color: the color we want to choose    
    '''
    plt.imshow(np.uint8(I))
    plt.plot(xc, yc, '+', color=color, linewidth=2, markersize=6)
    
    theta = np.arange(-np.pi, np.pi+0.0001, 0.0001)
    
    for i in range(len(xc)):
        plt.plot(xc[i] + r[i] * np.cos(theta), yc[i] + r[i] * np.sin(theta), '.', markersize=5, color=color)
    
    plt.show()