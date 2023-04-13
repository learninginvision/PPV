import numpy as np
import cv2
from skimage.feature import canny

from Multi_Scale_Edge_Detection.Edge_Sample import Edge_Sample
from Multi_Scale_Edge_Detection.Edge_feature_Extract import Edge_feature_Extract

from Pairwise_Voting.Pairwise_LineVoting_NoDirection import pairwiseLineVotingNoDirection
from Pairwise_Voting.voteSelection import voteSelection
from Pairwise_Voting.MeanShiftCluster import MeanShiftCluster
from Pairwise_Voting.selectionOriginal import selectionOriginal

from Drawing_Result.PlotCircle import PlotCircle


## Input Image
Imagename = 'test_images/multiple.png'

im = cv2.imread(Imagename)
image = im[:, :, 0]


## Args
Rmax = 100
bandwidth = 25
tao = 0.2
k = 10


## Canny Edge Detection
canny_scale = 20
canny_threshold_low = 3
canny_threshold_high = 5

BW1 = canny(image, sigma=canny_scale, low_threshold=canny_threshold_low, high_threshold=canny_threshold_high)


## Sample Edge Map
M, N = image.shape
sampling_rate = np.round(np.sqrt((M * N)/(0.2 * 10**4)))

BW = Edge_Sample(BW1, sampling_rate)


## Extract Edge Features
sigma = 5
Cos_theta, Sin_theta, Curvature = Edge_feature_Extract(image, sigma, BW)
Arg_point = np.arctan2(Sin_theta, Cos_theta)


## Line Clustering
x, weight = pairwiseLineVotingNoDirection(BW, Arg_point, Rmax, tao)


## Vote Selection
x_selected, weight_selected = voteSelection(x, weight, threshold=0.01)


## Mode Finding and Mean Shift Clustering
clustCent, point2cluster = MeanShiftCluster(x_selected, bandwidth, weight_selected)
#, clustMembsCell


## Select the k Best Hypotheses
Result = selectionOriginal(clustCent, k, x_selected, weight_selected, tao)


## Plot the Result
color = 'green'
PlotCircle(im, Result[0,:], Result[1,:], Result[2,:], color)