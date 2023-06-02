import numpy as np
import math
from utils.filter import Filter
from utils.gradient import Gradient

   
def edge_feature_extract(Im, sigma, edge_map):
    '''
    Extract the local feature
    (gradient direction and curvature for every edge point)
    Input:
        Im: the input image after reflections removing
        sigma: the scale value for smooth
        Edge_map: the accurate edge map
    Output:
        Cos_theta: Cos_theta(y,x) is the gradient direction at point(y, x)
        Sin_theta: Sin_theta(y,x) is the gradient direction at point(y, x)
        Curvature: Curvature(y,x) is the curvature of the edge point(y, x)
    '''
    # image smooth
    # Gaussian filter
    im_smooth = Filter().gaussian_filter(np.double(Im), sigma)
    I = im_smooth   
    # I = Im
    
    M, N = I.shape

    # calculate the gradient
    # the first order horizontal gradient
    I_x = Gradient().gradient_x(I)
    # the first order vertical gradient
    I_y = Gradient().gradient_y(I)
    # the second order horizontal gradient
    I_xx = Gradient().gradient_xx(I)
    # the second order vertical gradient
    I_yy = Gradient().gradient_yy(I)
    # the second cross gradient
    I_xy = Gradient().gradient_xy(I)
    
    # Cartesian coordinate second order derivative
    I_uu = Gradient().gradient_uu(I, I_x, I_xx, I_y, I_yy, I_xy)
    I_v = Gradient().gradient_v(I, I_x, I_y)
    
    # Calculate the curvature and direction 
    curvature = I_uu / I_v
    Cos_theta = I_x / I_v
    Sin_theta = I_y / I_v
    
    for y in range(M):
        for x in range(N):
            if edge_map[y, x] == 0:
                curvature[y, x] = 0
                Cos_theta[y, x] = 0
                Sin_theta[y, x] = 0
    
    return Cos_theta, Sin_theta, curvature


def edge_direction(edge_map, I):
    '''
    Calculate the direction for every edgepoint
    Output:
        Arg: the matrix records the dominant direction for every edge point
            (unit: radian; range: from 0 to 2*pi)
    '''  
    Arg = np.zeros((480, 640))
    for i in range(480):
        for j in range(640):
            if (edge_map[i, j] == 1):
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