from utils.filter import GaussianFilter
from utils.gradient import gradient_x, gradient_xx, gradient_xxx
from utils.gradient import gradient_y, gradient_yy, gradient_yyy, gradient_xy
from utils.gradient import gradient_xy, gradient_xyy, gradient_xxy
from utils.gradient import gradient_vv, gradient_vvv
import numpy as np


def SubpixelEdge(I_vv, I_vvv, threshold):
    '''
    Using subpixel method to detect edge points.
    Input:
        threshold: the theshold to control extracting obvious edge, for example: -0.001
    Output:
        Edgemap: Edgemap(y,x) = 1 means pixel (y, x) is an edge point (1: positive; 0: negative)
    '''   
    M, N = I_vv.shape
    Edge_map = np.zeros((M, N))
    Edge_map_sign = np.zeros((M, N))

    for i in range(M - 1):
        for j in range(N - 1):
            vvv1 = I_vvv[i, j]
            vvv2 = I_vvv[i, j + 1]
            vvv3 = I_vvv[i + 1, j]
            vvv4 = I_vvv[i + 1, j + 1]
            # -0.001 is a threshold to extract obvious edge feature
            if max([vvv1, vvv2, vvv3, vvv4]) < threshold:
                vv1 = I_vv[i, j]
                vv2 = I_vv[i, j + 1]
                vv3 = I_vv[i + 1, j]
                vv4 = I_vv[i + 1, j + 1]
                Edge_map[i, j] = 1

                # Edge_map_sign[i, j] = (Edge_map_sign[i, j] | (vv1 > 0) << 0)
                # Edge_map_sign[i, j] = (Edge_map_sign[i, j] | (vv2 > 0) << 1)
                # Edge_map_sign[i, j] = (Edge_map_sign[i, j] | (vv3 > 0) << 2)
                # Edge_map_sign[i, j] = (Edge_map_sign[i, j] | (vv4 > 0) << 3)

                if (max([vv1, vv2, vv3, vv4]) < 0):
                    Edge_map[i, j] = 0
                if (min([vv1, vv2, vv3, vv4]) > 0):
                    Edge_map[i, j] = 0
                if (vv1 > 0 and vv4 > 0 and vv3 < 0 and vv4 < 0):
                    Edge_map[i, j] = 0
                if (vv1 < 0 and vv4 < 0 and vv3 > 0 and vv4 > 0):
                    Edge_map[i, j] = 0
    
    return Edge_map, Edge_map_sign


def Edge_Detect(I, Sigma, threshold):
    '''
    Detect the edge points of iris image at a coarse scale.
    ''' 
    im_smooth = GaussianFilter(I, Sigma)
    I = im_smooth
    
    # calculate the gradient
    I_x = gradient_x(I) # the first order horizontal gradient   
    I_y = gradient_y(I) # the first order vertical gradient
    I_xx = gradient_xx(I)
    I_yy = gradient_yy(I)
    I_xxx = gradient_xxx(I)
    I_yyy = gradient_yyy(I)    
    I_xy = gradient_xy(I) # the second cross gradient   
    I_xyy = gradient_xyy(I)  
    I_xxy = gradient_xxy(I)
    
    # Cartesian coordinate second order derivative
    I_vv = gradient_vv(I, I_x, I_xx, I_y, I_yy, I_xy)
    I_vvv = gradient_vvv(I, I_x, I_xxx, I_y, I_yyy, I_xy, I_xxy, I_xyy)
    
    # subpixel edge detection
    Edge_map_coarse, Edge_map_coarse_sign = SubpixelEdge(I_vv, I_vvv, threshold)
    
    return Edge_map_coarse, Edge_map_coarse_sign