import cv2
from skimage.feature import canny
from Multi_Scale_Edge_Detection.Edge_Sample import Edge_Sample
from Edgemap2Edgepoint import Edgemap2Edgepoint
from PlotEdgeMark import PlotEdgeMark

I = cv2.imread('test_images/occlusion.jpg')
I = I[:, :, 0]

canny_scale = 22.0
canny_threshold_low = 3
canny_threshold_high = 5

BW1 = canny(I, sigma=canny_scale, low_threshold=canny_threshold_low, high_threshold=canny_threshold_high)

Edge_map_sample = Edge_Sample(BW1 , 6)
Edge_point_location = Edgemap2Edgepoint(Edge_map_sample)
PlotEdgeMark(I, Edge_point_location)