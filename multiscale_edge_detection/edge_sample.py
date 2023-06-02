import numpy as np


def edge_sample(edge_map, sample_rate):
    '''
    Down sample the edge map.
    Input:
        edge_map: the edge point map after edge detection and edge localization
        sample_rate
    Output:
        edge_map_sample: the sampled edge point map
    '''      
    M, N = edge_map.shape
    
    drow = int(M / sample_rate)
    dcolumn = int(N / sample_rate)
    
    # Half of the sampling rate
    hsamplerate = np.fix(sample_rate / 2) + 1 
    
    edge_map_sample = np.zeros((M, N))
    
    sampling_rate = int(sample_rate)
    # Sample the edge image
    for i in range(drow):
        for j in range(dcolumn):
            X = []            
            for m in range(sampling_rate):
                for n in range(sampling_rate):
                    
                    m1 = (i - 1) * sampling_rate + m
                    n1 = (j - 1) * sampling_rate + n
                    
                    if edge_map[m1, n1] == 1:
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
                edge_map_sample[location[0], location[1]] = 1
    
    return edge_map_sample