import numpy as np

def voteSelection(x, weight, threshold):
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