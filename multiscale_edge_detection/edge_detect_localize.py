from utils.filter import Filter
from utils.gradient import Gradient

import numpy as np


class EdgeDetectLocalize():
    
    def subpixel_edge(self, I_vv, I_vvv, threshold):
        '''
        Using subpixel method to detect edge points.
        Input:
            threshold: the theshold to control extracting obvious edge, for example: -0.001
        Output:
            edgemap: Edgemap(y,x) = 1 means pixel (y, x) is an edge point (1: positive; 0: negative)
        '''   
        M, N = I_vv.shape
        edge_map = np.zeros((M, N))
        edge_map_sign = np.zeros((M, N))

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
                    edge_map[i, j] = 1

                    # edge_map_sign[i, j] = (edge_map_sign[i, j] | (vv1 > 0) << 0)
                    # edge_map_sign[i, j] = (edge_map_sign[i, j] | (vv2 > 0) << 1)
                    # edge_map_sign[i, j] = (edge_map_sign[i, j] | (vv3 > 0) << 2)
                    # edge_map_sign[i, j] = (edge_map_sign[i, j] | (vv4 > 0) << 3)

                    if (max([vv1, vv2, vv3, vv4]) < 0):
                        edge_map[i, j] = 0
                    if (min([vv1, vv2, vv3, vv4]) > 0):
                        edge_map[i, j] = 0
                    if (vv1 > 0 and vv4 > 0 and vv3 < 0 and vv4 < 0):
                        edge_map[i, j] = 0
                    if (vv1 < 0 and vv4 < 0 and vv3 > 0 and vv4 > 0):
                        edge_map[i, j] = 0
        
        return edge_map, edge_map_sign


    def edge_detect(self, I, sigma, threshold):
        '''
        Detect the edge points of iris image at a coarse scale.
        ''' 
        im_smooth = Filter().gaussian_filter(I, sigma)
        I = im_smooth
        
        # calculate the gradient
        I_x = Gradient().gradient_x(I) # the first order horizontal gradient   
        I_y = Gradient().gradient_y(I) # the first order vertical gradient
        I_xx = Gradient().gradient_xx(I)
        I_yy = Gradient().gradient_yy(I)
        I_xxx = Gradient().gradient_xxx(I)
        I_yyy = Gradient().gradient_yyy(I)    
        I_xy = Gradient().gradient_xy(I) # the second cross gradient   
        I_xyy = Gradient().gradient_xyy(I)  
        I_xxy = Gradient().gradient_xxy(I)
        
        # Cartesian coordinate second order derivative
        I_vv = Gradient().gradient_vv(I, I_x, I_xx, I_y, I_yy, I_xy)
        I_vvv = Gradient().gradient_vvv(I, I_x, I_xxx, I_y, I_yyy, I_xy, I_xxy, I_xyy)
        
        # subpixel edge detection
        edge_map_coarse, edge_map_coarse_sign = self.subpixel_edge(I_vv, I_vvv, threshold)
        
        return edge_map_coarse, edge_map_coarse_sign
    
    
    def edge_localize(I, sigma, edge_map_coarse, edge_map_coarse_sign, threshold):
        '''
        Localize the edge points of iris at a fine scale and constraint the
        gradient direction of the edge point at finner scale is the same as
        the gradient direction at coarse scale 
        Input:
            I: the input image data
            sigma: the scale value
            edge_map_coarse: the Edge map extracted at a coarse scale
            edge_map_coarse_sign: the sign of the edge map at a coarse scale
            threshold: the threshold controlling the edge extraction, for example: -0.005
        Output:
            edge_map: the accurate edge points of iris image
        '''
        ## Gaussian smooth
        # smooth the image at a finer scale
        im_smooth = Filter().GaussianFilter(I, sigma) # I: double
        I = im_smooth

        # calculate the gradient
        I_x = Gradient().gradient_x(I)
        I_y = Gradient().gradient_y(I)
        I_xx = Gradient().gradient_xx(I)
        I_yy = Gradient().gradient_yy(I)
        I_xxx = Gradient().gradient_xxx(I)
        I_yyy = Gradient().gradient_yyy(I)
        
        I_xy = Gradient().gradient_xy(I)
        I_xyy = Gradient().gradient_xyy(I)
        I_xxy = Gradient().gradient_xxy(I)

        I_vv = Gradient().gradient_vv(I, I_x, I_xx, I_y, I_yy, I_xy)
        I_vvv = Gradient().gradient_vvv(I, I_x, I_xxx, I_y, I_yyy, I_xy, I_xxy, I_xyy)
        
        M, N = edge_map_coarse.shape
        delta_x = np.zeros((M, N))
        edge_map = np.zeros((M, N))

        ## the searching interval
        interval = 10
        for y in range(interval, M - interval):
            for x in range(interval, N - interval):
                if( edge_map_coarse[y , x] == 1 ):
                    delta_V = np.arange(-interval , interval+1)
                    delta_X = np.round(delta_V * (I_x[y , x]/np.sqrt(I_x[y , x]**2 + I_y[y , x]**2)))
                    delta_Y = np.round(delta_V * (I_y[y , x]/np.sqrt(I_x[y , x]**2 + I_y[y , x]**2)))

                    for i in range(2 * interval + 1):
                        delta_x[i] = delta_X[i]
                        delta_y = delta_Y[i]

                        # the third gradient value on the gradient direction
                        vvv1 = I_vvv[y + delta_y , x + delta_x]
                        vvv2 = I_vvv[y + delta_y , x + delta_x + 1]
                        vvv3 = I_vvv[y + delta_y + 1 , x + delta_x]
                        vvv4 = I_vvv[y + delta_y + 1 , x + delta_x + 1]

                        # the second gradient value on the gradient direction
                        vv1 = I_vv[y + delta_y , x + delta_x]
                        vv2 = I_vv[y + delta_y , x + delta_x + 1]
                        vv3 = I_vv[y + delta_y + 1 , x + delta_x]
                        vv4 = I_vv[y + delta_y + 1 , x + delta_x + 1]

                        sign = 0
                        sign = np.bitwise_or(sign, (vv1 > 0) << 0)
                        sign = np.bitwise_or(sign, (vv2 > 0) << 1)
                        sign = np.bitwise_or(sign, (vv3 > 0) << 2)
                        sign = np.bitwise_or(sign, (vv4 > 0) << 3)

                        if( (max([vvv1, vvv2, vvv3, vvv4]) < threshold) and (np.bitwise_xor(sign, edge_map_coarse_sign[y , x]) == 0)):
                            edge_map[y + delta_y , x + delta_x] = 1

        return edge_map


    def edge_localize_noconstraint(I, sigma, edge_map_coarse, threshold):
        '''
        Localize the edge points of iris at a fine scale and there is no gradient direction constraint.
        Input:
            thershold: the thrshold controlling the obvious edge detection at a finner scale
        '''
        # Gaussian smooth
        # smooth the image at a finer scale
        im_smooth = Filter().gaussian_filter(I, sigma)
        I = im_smooth
        
        I_x = Gradient().gradient_x(I)
        I_y = Gradient().gradient_y(I)
        I_xx = Gradient().gradient_xx(I)
        I_yy = Gradient().gradient_yy(I)
        I_xxx = Gradient().gradient_xxx(I)
        I_yyy = Gradient().gradient_yyy(I)
        I_xy = Gradient().gradient_xy(I)    
        I_xyy = Gradient().gradient_xyy(I)    
        I_xxy = Gradient().gradient_xxy(I)
        I_vv = Gradient().gradient_vv(I,I_x,I_xx,I_y,I_yy,I_xy)
        I_vvv = Gradient().gradient_vvv(I,I_x,I_xxx,I_y,I_yyy,I_xy,I_xxy,I_xyy)

        M, N = edge_map_coarse.shape
        delta_x = np.zeros((M, N))
        edge_map = np.zeros((M, N))
        
        # the searching interval
        interval = 20
        for y in range(interval, M - interval):
            for x in range(interval, N - interval):
                if (edge_map_coarse[y, x] == 1):
                    # search in two directions, from the nearest neighbour point to the farthest point
                    delta_V_postive = np.arange(0, interval+1)
                    delta_V_negative = np.arange(-1, -interval-1, -1)
                    
                    delta_V = np.empty(2*interval+1, dtype=int)
                    delta_V[::2] = delta_V_postive
                    delta_V[1::2] = delta_V_negative
                    
                    delta_X = np.rint(delta_V * (I_x[y, x]/np.sqrt(I_x[y, x]**2 + I_y[y, x]**2)))           
                    delta_Y = np.rint(delta_V * (I_y[y, x]/np.sqrt(I_x[y, x]**2 + I_y[y, x]**2)))
                    
                    for i in range(2 * interval + 1):
                        delta_x = delta_X[i]
                        delta_y = delta_Y[i]

                        try:
                            # the third order gradient value on the gradient direction
                            vvv1 = I_vvv[y + delta_y, x + delta_x]
                            vvv2 = I_vvv[y + delta_y, x + delta_x + 1]
                            vvv3 = I_vvv[y + delta_y + 1, x + delta_x]
                            vvv4 = I_vvv[y + delta_y + 1, x + delta_x + 1]
                            
                            # the second order gradient value on the gradient direction
                            vv1 = I_vv[y + delta_y, x + delta_x]
                            vv2 = I_vv[y + delta_y, x + delta_x + 1]
                            vv3 = I_vv[y + delta_y + 1, x + delta_x]
                            vv4 = I_vv[y + delta_y + 1, x + delta_x + 1]

                            if max([vvv1, vvv2, vvv3, vvv4]) < threshold:
                                if max([vv1, vv2, vv3, vv4]) < 0 or min([vv1, vv2, vv3, vv4]) > 0 or (vv1>0 and vv4>0 and vv3<0 and vv4<0) or (vv1<0 and vv4<0 and vv3>0 and vv4>0):
                                    edge_map[y + delta_y, x + delta_x] = 0
                            else:
                                # if find the edge point, the search stops and iterate to another point
                                edge_map[y + delta_y, x + delta_x] = 1
                                break
                        except:
                            pass
        
        return edge_map