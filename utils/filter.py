import numpy as np

def dftfilt(f, H):
    '''
    Perform frequency domain filtering.
    Input:
        f: the imput image
        H: the Fourier coefficient of the filter
    Output:
        g: the output image after filtering
    '''
    # Fourier transform of the input image
    F = np.fft.fft2(f, s=(H.shape[0], H.shape[1]))
    
    # filter
    g = np.real(np.fft.ifft2(H * F))
    
    # output
    g = g[:f.shape[0], :f.shape[1]]
    
    return g


def dftuv(M, N):
    '''
    Produce the grid.
    '''
    u = np.arange(0, M)
    v = np.arange(0, N)
    
    idx = np.where(u > (M/2))
    u[idx] = u[idx] - M
    idy = np.where(v > (N/2))
    v[idy] = v[idy] - N
    
    V, U = np.meshgrid(v, u)
    
    return U, V

    
def lpfilter(type, M, N, D0, n):
    '''
    Produce the Guassian filter in Fourier spectrum.
    Input:
        type: denote the type of lpfilter
            'ideal': the standard low pass filter
            'btw': the butterworth low pass filter
            'gaussian': the gaussian low pass filter
        M, N: the height and width of the filter
        D0: the scale of the filter
        n: the order of the filter
    Output:
        H: the filter coefficients
    '''
    U, V = dftuv(M, N)
    D = np.sqrt(U ** 2 + V ** 2)
    
    if type == 'ideal':
        H = np.where(D <= D0, 1, 0)
    elif type == 'btw':
        if n is None:
            n = 1
        H = 1 / (1 + (D / D0) ** (2 * n))
    elif type == 'gaussian':
        H = np.exp(- D ** 2 / (2 * (D0 ** 2)))
    else:
        raise ValueError('Unknown filter type.')
    
    return H, D

   
def GaussianFilter(I, sigma): 
    '''
    Calculate the scale space image.
    Input:
        I: the input image
        sigma: the scale of the smoothing
    Output:
        Result: the smooth image
    '''
    M, N = I.shape
    DR = np.sqrt((M / 2) ** 2 + (N / 2) ** 2)
    D0 = DR * np.pi / (10 * sigma)
    # D0 = sigma
    
    # Gaussian low pass filter
    H, D = lpfilter('gaussian', M, N, D0, n=None)
    Result = dftfilt(I, H)
    
    return Result