import numpy as np

    
def houghspace_data(): 
    Hough_space1 = np.load(file='Hough_space1.npy')
    
    x = np.empty((3, 0))
    weight = []
    M, N, H = Hough_space1.shape
    for m in range(M):
        for n in range(N):
            for h in range(H):
                if Hough_space1[m, n, h] > 0.0001:
                    x1 = np.transpose(np.array([[m, n, h]]))
                    x = np.append(x, x1, axis=1)
                    weight = np.reshape(np.append(weight, Hough_space1[m, n, h]), (1, -1)) 
    del Hough_space1
    
    Hough_space2 = np.load(file='Hough_space2.npy')
    for m in range(M):
        for n in range(N):
            for h in range(H):
                if Hough_space2[m, n, h] > 0.0001:
                    x2 = np.transpose(np.array([[m, n, h+100]]))
                    x = np.append(x, x2, axis=1)
                    weight = np.reshape(np.append(weight, Hough_space2[m, n, h]), (1, -1)) 
    del Hough_space2
    
    return x, weight