import numpy as np


def selection_original(clustCent, k, x, weight, tao):
    """
    Select the k best hypotheses.
    Input:
        clustCent: The candidate parameter for eye location and radius.
        k: The best k best hypotheses.
        x: Every vote.
        weight: The weight for every vote.
        tao: The threshold to determine the valid vote and the weight for every weight.
    Output:
        result: The result matrix which contains the localization result.
        [xc1, yc1, rc1]
        [xc2, yc2, rc2]
    """
    result = []
    numclus = clustCent.shape[1]
    numpoint = x.shape[1]
    
    # flg = np.inf
    t = 0.2 * tao
    Houghscore = np.zeros(numclus)
    
    for i in range(numclus):
        xc0 = clustCent[1, i]
        yc0 = clustCent[0, i]
        rc0 = clustCent[2, i]
        
        Diff = x - np.array([yc0, xc0, rc0]).reshape(-1, 1) @ np.ones((1, numpoint))
        dis = np.sqrt(np.sum(Diff ** 2, axis=0))
        
        normdis = np.abs(dis / (x[2, :] + rc0) / 2)
        indices = np.where(normdis < tao)[0]
        Houghscore[i] = np.sum(weight[indices] * np.exp(-normdis[indices] ** 2 / t)) / rc0
    
    for i in range(k):
        h = np.max(Houghscore)
        index = np.where(Houghscore == h)[0]
        
        result.append(clustCent[:, index])
        num = len(index)
        k = k - num + 1
        
        Houghscore = np.delete(Houghscore, index)
        clustCent = np.delete(clustCent, index, axis=1)
        
    return np.concatenate(result, axis=1)


def vote_selection(x, weight, threshold):
    """
    Select the vote which has some obvious contribution to the hypothesis selection.
    Input:
        x (numpy.ndarray): Every vote.
        weight (numpy.ndarray): The weight for every vote.
        threshold (float): The threshold for the vote selection.
    Output:
            x_selected (numpy.ndarray): The selected vote.
            weight_selected (numpy.ndarray): The weight for selected vote.
    """
    x_selected = x[:, weight > threshold]
    weight_selected = weight[weight > threshold]

    return x_selected, weight_selected