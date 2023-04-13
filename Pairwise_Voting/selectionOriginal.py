import numpy as np

def selectionOriginal(clustCent, k, x, weight, tao):
    """
    Select the k best hypotheses.
    Input:
        clustCent: The candidate parameter for eye location and radius.
        k: The best k best hypotheses.
        x: Every vote.
        weight: The weight for every vote.
        tao: The threshold to determine the valid vote and the weight for every weight.
    Output:
        Result: The result matrix which contains the localization result.
        [xc1, yc1, rc1]
        [xc2, yc2, rc2]
    """
    Result = []
    Numclus = clustCent.shape[1]
    Numpoint = x.shape[1]
    
    # flg = np.inf
    t = 0.2 * tao
    Houghscore = np.zeros(Numclus)
    
    for i in range(Numclus):
        xc0 = clustCent[1, i]
        yc0 = clustCent[0, i]
        rc0 = clustCent[2, i]
        
        Diff = x - np.array([yc0, xc0, rc0]).reshape(-1, 1) @ np.ones((1, Numpoint))
        dis = np.sqrt(np.sum(Diff ** 2, axis=0))
        
        normdis = np.abs(dis / (x[2, :] + rc0) / 2)
        indices = np.where(normdis < tao)[0]
        Houghscore[i] = np.sum(weight[indices] * np.exp(-normdis[indices] ** 2 / t)) / rc0
    
    for i in range(k):
        h = np.max(Houghscore)
        index = np.where(Houghscore == h)[0]
        
        Result.append(clustCent[:, index])
        num = len(index)
        k = k - num + 1
        
        Houghscore = np.delete(Houghscore, index)
        clustCent = np.delete(clustCent, index, axis=1)
        
    return np.concatenate(Result, axis=1)