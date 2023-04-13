import numpy as np
from utils.filter import GaussianFilter
from utils.gradient import gradient_x, gradient_y, gradient_xx, gradient_yy
from utils.gradient import gradient_xy, gradient_uu, gradient_v
    
def Edge_feature_Extract(Im, sigma, Edge_map): 
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
    im_smooth = GaussianFilter(np.double(Im), sigma)
    I = im_smooth   
    # I = Im
    
    M, N = I.shape

    # calculate the gradient
    # the first order horizontal gradient
    I_x = gradient_x(I)
    # the first order vertical gradient
    I_y = gradient_y(I)
    # the second order horizontal gradient
    I_xx = gradient_xx(I)
    # the second order vertical gradient
    I_yy = gradient_yy(I)
    # the second cross gradient
    I_xy = gradient_xy(I)
    
    # Cartesian coordinate second order derivative
    I_uu = gradient_uu(I, I_x, I_xx, I_y, I_yy, I_xy)
    I_v = gradient_v(I, I_x, I_y)
    
    # Calculate the curvature and direction 
    Curvature = I_uu / I_v
    Cos_theta = I_x / I_v
    Sin_theta = I_y / I_v
    
    for y in range(M):
        for x in range(N):
            if Edge_map[y, x] == 0:
                Curvature[y, x] = 0
                Cos_theta[y, x] = 0
                Sin_theta[y, x] = 0
    
    return Cos_theta, Sin_theta, Curvature