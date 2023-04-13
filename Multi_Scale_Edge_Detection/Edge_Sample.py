import numpy as np
from utils.gradient import gradient_x, gradient_xx, gradient_xxx
from utils.gradient import gradient_y, gradient_yy, gradient_yyy, gradient_xy
from utils.gradient import gradient_xy, gradient_xyy, gradient_xxy
from utils.gradient import gradient_vv, gradient_vvv
from utils.filter import GaussianFilter
  
def Edge_Sample(Edge_map, sample_rate):
    '''
    Down sample the edge map.
    Input:
        Edge_map: the edge point map after edge detection and edge localization
        sample_rate
    Output:
        Edge_map_sample: the sampled edge point map
    '''      
    M, N = Edge_map.shape
    
    drow = int(M / sample_rate)
    dcolumn = int(N / sample_rate)
    
    # Half of the sampling rate
    hsamplerate = np.fix(sample_rate / 2) + 1 
    
    Edge_map_sample = np.zeros((M, N))
    
    sampling_rate = int(sample_rate)
    # Sample the edge image
    for i in range(drow):
        for j in range(dcolumn):
            X = []            
            for m in range(sampling_rate):
                for n in range(sampling_rate):
                    
                    m1 = (i - 1) * sampling_rate + m
                    n1 = (j - 1) * sampling_rate + n
                    
                    if Edge_map[m1, n1] == 1:
                        X.append([m1, n1])
            
            if X != []:
                Length = len(X)
                xo = [(i - 1) * sampling_rate + hsamplerate, (j - 1) * sampling_rate + hsamplerate]
                Xo = np.ones((Length, 1)) * xo
                Diff = X - Xo
                
                # [value, index] = min(sum(Diff .^ 2, 2));
                dist = np.sum(Diff ** 2, 1)
                index = np.argmin(dist)
                
                location = X[index]
                Edge_map_sample[location[0], location[1]] = 1
    
    return Edge_map_sample


def Edge_Localize(I, Sigma, Edge_map_coarse, Edge_map_coarse_sign, threshold):
    '''
    Localize the edge points of iris at a fine scale and constraint the
    gradient direction of the edge point at finner scale is the same as
    the gradient direction at coarse scale 
    Input:
        I: the input image data
        Sigma: the scale value
        Edge_map_coarse: the Edge map extracted at a coarse scale
        Edge_map_coarse_sign: the sign of the edge map at a coarse scale
        threshold: the threshold controlling the edge extraction, for example: -0.005
    Output:
        Edge_map: the accurate edge points of iris image
    '''
    ## Gaussian smooth
    # smooth the image at a finer scale
    im_smooth = GaussianFilter(I, Sigma) # I: double
    I = im_smooth

    # calculate the gradient
    I_x = gradient_x(I)
    I_y = gradient_y(I)
    I_xx = gradient_xx(I)
    I_yy = gradient_yy(I)
    I_xxx = gradient_xxx(I)
    I_yyy = gradient_yyy(I)
    
    I_xy = gradient_xy(I)
    I_xyy = gradient_xyy(I)
    I_xxy = gradient_xxy(I)

    I_vv = gradient_vv(I, I_x, I_xx, I_y, I_yy, I_xy)
    I_vvv = gradient_vvv(I, I_x, I_xxx, I_y, I_yyy, I_xy, I_xxy, I_xyy)
    
    M, N = Edge_map_coarse.shape
    delta_x = np.zeros((M, N))
    Edge_map = np.zeros((M, N))

    ## the searching interval
    Interval = 10
    for y in range(Interval, M - Interval):
        for x in range(Interval, N - Interval):
            if( Edge_map_coarse[y , x] == 1 ):
                delta_V = np.arange(-Interval , Interval+1)
                delta_X = np.round(delta_V * (I_x[y , x]/np.sqrt(I_x[y , x]**2 + I_y[y , x]**2)))
                delta_Y = np.round(delta_V * (I_y[y , x]/np.sqrt(I_x[y , x]**2 + I_y[y , x]**2)))

                for i in range(2 * Interval + 1):
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

                    if( (max([vvv1, vvv2, vvv3, vvv4]) < threshold) and (np.bitwise_xor(sign, Edge_map_coarse_sign[y , x]) == 0)):
                        Edge_map[y + delta_y , x + delta_x] = 1

    return Edge_map


def Edge_Localize_noconstraint(I, Sigma, Edge_map_coarse, threshold):
    '''
    Localize the edge points of iris at a fine scale and there is no gradient direction constraint.
    Input:
        thershold: the thrshold controlling the obvious edge detection at a finner scale
    '''
    # Gaussian smooth
    # smooth the image at a finer scale
    im_smooth = GaussianFilter(I, Sigma)
    I = im_smooth
    
    I_x = gradient_x(I)
    I_y = gradient_y(I)
    I_xx = gradient_xx(I)
    I_yy = gradient_yy(I)
    I_xxx = gradient_xxx(I)
    I_yyy = gradient_yyy(I)
    I_xy = gradient_xy(I)    
    I_xyy = gradient_xyy(I)    
    I_xxy = gradient_xxy(I)
    I_vv = gradient_vv(I,I_x,I_xx,I_y,I_yy,I_xy)
    I_vvv = gradient_vvv(I,I_x,I_xxx,I_y,I_yyy,I_xy,I_xxy,I_xyy)

    M, N = Edge_map_coarse.shape
    delta_x = np.zeros((M, N))
    Edge_map = np.zeros((M, N))
    
    # the searching interval
    Interval = 20
    for y in range(Interval, M - Interval):
        for x in range(Interval, N - Interval):
            if (Edge_map_coarse[y, x] == 1):
                # search in two directions, from the nearest neighbour point to the farthest point
                delta_V_postive = np.arange(0, Interval+1)
                delta_V_negative = np.arange(-1, -Interval-1, -1)
                
                delta_V = np.empty(2*Interval+1, dtype=int)
                delta_V[::2] = delta_V_postive
                delta_V[1::2] = delta_V_negative
                
                delta_X = np.rint(delta_V * (I_x[y, x]/np.sqrt(I_x[y, x]**2 + I_y[y, x]**2)))           
                delta_Y = np.rint(delta_V * (I_y[y, x]/np.sqrt(I_x[y, x]**2 + I_y[y, x]**2)))
                
                for i in range(2 * Interval + 1):
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
                                Edge_map[y + delta_y, x + delta_x] = 0
                        else:
                            # if find the edge point, the search stops and iterate to another point
                            Edge_map[y + delta_y, x + delta_x] = 1
                            break
                    except:
                        pass
    
    return Edge_map