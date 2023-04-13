import numpy as np
from numpy.linalg import norm
    
def Intersect_3D(y1, x1, theta1, y3, x3, theta3, tao):
    '''
    Calculate the middle point of 2 lines in x-y-r plane
    Input:
        y1: the y coordinate of a point on line 1
        x1: the x coordinate of a point on line 1
        theta1: the gradient diection of edge point (x1, y1)
        y3: the y coordinate of a points on line 2
        x3: the x coordinate of a points on line 2
        theta3: the gradient diection of edge point (x2, y2)
    Output:
        y: the y coordinate of the middle point in x-y-r plane
        x: the x coordinate of the middle point in x-y-r plane
        r: the r coordinate of the middle point in x-y-r plane
    '''
    r1 = 0
    r3 = 0
    
    # another point on line 1
    y2 = y1 - 5 * np.sin(theta1)
    x2 = x1 - 5 * np.transpose(np.cos(theta1))
    r2 = 5
    
    # another point on line 2
    y4 = y3 - 5 * np.sin(theta3)
    x4 = x3 - 5 * np.cos(theta3)
    r4 = 5
    
    x13 = x1 - x3
    y13 = y1 - y3
    r13 = r1 - r3
    x43 = x4 - x3
    y43 = y4 - y3
    r43 = r4 - r3
    x21 = x2 - x1
    y21 = y2 - y1
    r21 = r2 - r1
    
    d1343 = x13 * x43 + y13 * y43 + r13 * r43
    d4321 = x43 * x21 + y43 * y21 + r43 * r21
    d1321 = x13 * x21 + y13 * y21 + r13 * r21
    d4343 = x43 * x43 + y43 * y43 + r43 * r43
    d2121 = x21 * x21 + y21 * y21 + r21 * r21
    
    denom = d2121 * d4343 - d4321 * d4321
    numer = d1343 * d4321 - d1321 * d4343
    
    mua = numer / denom
    mub = (d1343 + d4321 * mua) / d4343
    
    Ints1 = []
    Ints2 = []
    Mid = []
    
    Ints1.append(x1 + mua * x21) # Ints1[0]
    Ints1.append(y1 + mua * y21) # Ints1[1]
    Ints1.append(r1 + mua * r21) # Ints1[2]
    
    # coordinates of the second intersection on line 2
    Ints2.append(x3 + mub * x43) # Ints2[0]
    Ints2.append(y3 + mub * y43) # Ints2[1]
    Ints2.append(r3 + mub * r43) # Ints2[2]
    
    Mid.append((Ints1[0] + Ints2[0]) / 2) # Mid[0]
    Mid.append((Ints1[1] + Ints2[1]) / 2) # Mid[1]
    Mid.append((Ints1[2] + Ints2[2]) / 2) # Mid[2]
    
    # dis = np.linalg.norm((np.array(Ints1)-np.array(Ints2)), 2)
    # sigma = 2.5
    # weight = np.exp(- dis ** 2 / (2 * (sigma * (Ints1[2] + Ints2[2]) / 2)))
    
    Ints1 = np.array(Ints1)
    Ints2 = np.array(Ints2)
    Mid = np.array(Mid)
    normdis = norm((Ints1-Ints2), 2) / np.abs(Mid[2])
    
    t = tao * 0.2
    if normdis < tao:
        weight = np.exp(- normdis ** 2 / t)
    else:
        weight = 0
        
    return Ints1, Ints2, Mid, weight