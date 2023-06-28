# Fast and Robust Circular Object Detection with Probabilistic Pairwise Voting

<div align=center>
<img src="https://github.com/learninginvision/PPV/blob/main/pic/proposed_method.png" alt="image-20230411104631752" width="950" />
</div>
 
These are official codes for the paper [*Fast and Robust Circular Object Detection With Probabilistic Pairwise Voting*](https://ieeexplore.ieee.org/document/6008626). In this paper, we propose a new detection method for circular objects - Probabilistic Pairwise Voting (PPV). Based on an extension of the Hough transform, this method utilizes the gradient information to produce projection lines in the *x-y-r* space. If the projection lines of two edge pixels intersect at one point, they belong to the same circle. Otherwise, they produce a hypothesis of a circle. The likelihood of each such hypothesis is modeled as a Gaussian distribution dependent on the distance between two projection lines, equal to the length of a line perpendicular to both projection lines. The score of a hypothesis is calculated by marginalization over all edge point pairs that contribute to the hypothesis.

It has many advantages: robust against occlusions, noise, and moderate shape deformations. Also, this method succeeded in multiple circular object detection and human iris detection.



## Tasks and Performances

### Detecting Circular Objects in Natural Images

Our method was tested on four representative natural images gathered from Google Image. Circular detection results on natural scenes. **(a)** Occlusion, **(b)** background clutter, **(c)** shape deformation, and **(d)** multiple circular objects.

<div align=center>
<img src="https://github.com/learninginvision/PPV/blob/main/pic/natural_results.png" alt="image-20230411105942277"  width="800" /> </div>

Compared with three other circle detection algorithms CHT, RCD, and AMLE, PPV shows its superiority in terms of accuracy and the short time required.

<div align=center>
<img src="https://github.com/learninginvision/PPV/blob/main/pic/natural_table.png" alt="image-20230411111428191"  width="500" /> </div>



### Localizing Iris in Face Images

This method was further tested on the challenging task of localizing the irides in face images from the CMU Multi-PIE face database. Our algorithm was robust to reflections from eyeglasses, and occlusion from eyelashes, eyelids, and hair.

<div align=center>
<img src="https://github.com/learninginvision/PPV/blob/main/pic/iris_results.png" alt="image-20230411112358346"  width="800" /> </div>

Also, the effect of the parameter $\tau$ on the localization performance and the eye localization accuracy were evaluated respectively. The best performance was obtained for $\tau=0.4$. Our method has the most accurate results among all the methods due to the largest area under the curve.

<div align=center>
<img src="https://github.com/learninginvision/PPV/blob/main/pic/iris_table.png" alt="image-20230411113425165"  width="600" /> </div>



## Code Descriptions

Here are the descriptions of the main codes.

```
main.py:                    use PPV for circular objects detection
utils:                      include the required filters and functions to calculate the gradient
Drawing_Result:             include functions about how to draw the edge results
Multi_Scale_Edge_Detection: include edge detection-related functions
Pairwise_Voting:            include functions about Probabilistic Hough Transform
```



## Supplementary Results

Some more test results are provided here. The input images, the edge maps, and the detection results of our method are presented simultaneously.

Simple examples of detecting circular objects:

<div align=center><img src="https://github.com/learninginvision/PPV/blob/main/pic/supplementary1.PNG" alt="supplementary1"  width="600" /> </div>

Results of detecting partially occluded circular objects:

<div align=center><img src="https://github.com/learninginvision/PPV/blob/main/pic/supplementary2.PNG" alt="supplementary2"  width="600" /> </div>

 Results of detecting various multiple circular objects:

<div align=center><img src="https://github.com/learninginvision/PPV/blob/main/pic/supplementary3.PNG" alt="supplementary3"  width="600" /> </div>
