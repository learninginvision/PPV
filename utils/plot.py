import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import canny

from multiscale_edge_detection.edge_sample import edge_sample
from multiscale_edge_detection.edgepoint import edgemap2edgepoint


class Plot():
    
    def plot_circle(self, I, xc, yc, r, color):
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


    def plot_edgemark(self, I, edgepoint):
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


    def plot_edgepoint(self, edge_point, I):
        '''
        Plot the edge point map.
        Input:
            edgePoint: the input edge point
            I: the input image    
        '''
        M, N = I[:, :, 0].shape
        plt.imshow(255 * np.ones((M, N)))

        plt.plot(edge_point[:, 0], edge_point[:, 1], 'o', markeredgecolor='k', markerfacecolor='b', markersize=6)
            
        plt.show()


    def plot_result(self, I):
        I = I[:, :, 0]

        canny_scale = 22.0
        canny_threshold_low = 3
        canny_threshold_high = 5

        BW1 = canny(I, sigma=canny_scale, low_threshold=canny_threshold_low, high_threshold=canny_threshold_high)

        edge_map_sample = edge_sample(BW1 , 6)
        edge_point_location = edgemap2edgepoint(edge_map_sample)
        self.plot_edgemark(I, edge_point_location)