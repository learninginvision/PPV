import numpy as np
import cv2
from skimage.feature import canny
import yaml
import matplotlib.pyplot as plt

from multiscale_edge_detection.edge_sample import edge_sample
from multiscale_edge_detection.edge_feature import edge_feature_extract

from pairwise_voting.pairwise_linevoting import PairwiseLineVoting
from pairwise_voting.selection import vote_selection
from pairwise_voting.meanshift_cluster import meanshift_cluster
from pairwise_voting.selection import selection_original

from utils.plot import Plot


def main():
    ## Input Image
    Imagename = 'test_images/multiple.png'

    im = cv2.imread(Imagename)
    image = im[:, :, 0]


    ## Parameters
    param = yaml.load(open('./param.yml'), Loader=yaml.FullLoader)
    Rmax = param['Rmax']
    bandwidth = param['bandwidth']
    tao = param['tao']
    k = param['k']


    ## Canny Edge Detection
    canny_scale = param['canny_scale']
    canny_threshold_low = param['canny_threshold_low']
    canny_threshold_high = param['canny_threshold_high']
    


    BW1 = canny(image, sigma=canny_scale, low_threshold=canny_threshold_low, high_threshold=canny_threshold_high)


    ## Sample Edge Map
    M, N = image.shape
    sampling_rate = np.round(np.sqrt((M * N)/(0.2 * 10**4)))

    BW = edge_sample(BW1, sampling_rate)


    ## Extract Edge Features
    sigma = param['sigma']
    Cos_theta, Sin_theta, curvature = edge_feature_extract(image, sigma, BW)
    Arg_point = np.arctan2(Sin_theta, Cos_theta)


    ## Line Clustering
    x, weight = PairwiseLineVoting().pairwise_linevoting_nodirection(BW, Arg_point, Rmax, tao)


    ## Vote Selection
    x_selected, weight_selected = vote_selection(x, weight, threshold=0.01)


    ## Mode Finding and Mean Shift Clustering
    clustCent, point2cluster = meanshift_cluster(x_selected, bandwidth, weight_selected)
    #, clustMembsCell


    ## Select the k Best Hypotheses
    result = selection_original(clustCent, k, x_selected, weight_selected, tao)


    ## Plot the Result
    # Plot().plot_result(im)
    color = param['color']
    Plot().plot_circle(im, result[0,:], result[1,:], result[2,:], color)


if __name__ == "__main__":
    main()