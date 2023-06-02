import numpy as np


class Gradient():
    
    def gradient_x(self, I):
        '''
        Calculate the first order horizontal gradient.
        Output:
            I_x: the first order x direction gradient
        '''  
        I_x = 1/2 * (np.roll(I, shift=-1, axis=1) - np.roll(I, shift=1, axis=1))   
        return I_x

        
    def gradient_xx(self, I):
        I_xx = np.roll(I, shift=-1, axis=1) - 2 * I + np.roll(I, shift=1, axis=1)
        return I_xx


    def gradient_xxx(self, I):
        I_xxx = 1 / 2 * (np.roll(I, shift=-2, axis=1) - 2 * np.roll(I, shift=-1, axis=1) + 2 * np.roll(I, shift=1, axis=1) - np.roll(I, shift=2, axis=1))
        return I_xxx


    def gradient_y(self, I):
        '''
        Calculate the first order vertical gradient.
        Output:
            I_x: the first order y direction gradient
        '''
        I_y = 1/2 * (np.roll(I, shift=-1, axis=0) - np.roll(I, shift=1, axis=0))
        return I_y


    def gradient_yy(self, I):
        I_yy = np.roll(I, shift=-1, axis=0) - 2 * I + np.roll(I, shift=1, axis=0)
        return I_yy


    def gradient_yyy(self, I):
        I_yyy = 1 / 2 * (np.roll(I, shift=-2, axis=0) - 2 * np.roll(I, shift=-1, axis=0) + 2 * np.roll(I, shift=1, axis=0) - np.roll(I, shift=2, axis=0))
        return I_yyy


    def gradient_xy(self, I):
        '''
        Calculate the second order cross-gradient.
        Output:
            I_x: the second order xy direction cross-gradient
        '''       
        I_xy = 1/4 * (np.roll(np.roll(I, shift=-1, axis=0), shift=-1, axis=1) - np.roll(np.roll(I, shift=1, axis=0), shift=-1, axis=1)
            - np.roll(np.roll(I, shift=-1, axis=0), shift=1, axis=1) + np.roll(np.roll(I, shift=1, axis=0), shift=1, axis=1))
        return I_xy


    def gradient_xxy(self, I):
        I_xxy = 1/2 * (np.roll(np.roll(I, shift=-1, axis=0), shift=-1, axis=1) - 2 * np.roll(I, shift=-1, axis=0)
                - np.roll(np.roll(I, shift=1, axis=0), shift=-1, axis=1) + np.roll(np.roll(I, shift=-1, axis=0), shift=1, axis=1)
                + 2 * np.roll(I, shift=1, axis=0) - np.roll(np.roll(I, shift=1, axis=0), shift=1, axis=1))
        return I_xxy


    def gradient_xyy(self, I):
        I_xyy = 1/2 * (np.roll(np.roll(I, shift=-1, axis=0), shift=-1, axis=1) - 2 * np.roll(I, shift=-1, axis=1)
                + np.roll(np.roll(I, shift=1, axis=0), shift=-1, axis=1) - np.roll(np.roll(I, shift=-1, axis=0), shift=1, axis=1)
                + 2 * np.roll(I, shift=1, axis=1) - np.roll(np.roll(I, shift=1, axis=0), shift=1, axis=1))
        return I_xyy


    def gradient_v(self, I, I_x, I_y):
        '''
        Output:
            I_v: the Cartesian coordinate first order derivative
        '''
        I_v = np.sqrt(I_x**2 + I_y**2)
        return I_v


    def gradient_uu(self, I, I_x, I_xx, I_y, I_yy, I_xy): 
        '''
        Input:
            I: the input image
            I_x, I_xx, I_y, I_yy, I_xy: the image derivative
        Output:
            I_uu: the Cartesian coordinate second order dirivative
        '''   
        I_uu = (I_x**2 * I_yy - 2 * I_x * I_y * I_xy + I_y**2 * I_xx) / (I_x**2 + I_y**2)
        return I_uu


    def gradient_vv(self, I, I_x, I_xx, I_y, I_yy, I_xy):
        '''
        Output:
            I_vv: the Cartesian coordinate second order derivative
        '''
        I_vv = (I_x**2 * I_xx + 2 * I_x * I_y * I_xy + I_y**2 * I_yy) / (I_x**2 + I_y**2)
        return I_vv


    def gradient_vvv(self, I, I_x, I_xxx, I_y, I_yyy, I_xy, I_xxy, I_xyy):
        '''
        Output:
            I_vvv: the Cartesian coordinate third order derivative
        '''
        I_vvv = (I_x ** 3 * I_xxx + 3 * I_x ** 2 * I_y * I_xxy + 3 * I_x * I_y ** 2 * I_xyy + I_y ** 3 * I_yyy) / (I_x ** 2 + I_y ** 2)
        return I_vvv