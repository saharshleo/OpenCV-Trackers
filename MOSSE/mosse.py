import numpy as np
import cv2
import random
import sys
import matplotlib.pyplot as plt


class MOSSE:
    '''
    TUNING PARAMS:
    learning_rate
    train_num
    sigma
    rotation angle in affine transformation
    '''

    def __init__(self, frame, roi, learning_rate, train_num, sigma):
        self.frame = frame
        self.roi = roi                      # contains (top_left_x, top_left_y, width, height)
        self.learning_rate = learning_rate  # running average parameter 
        self.train_num = train_num          # number of training examples to be trained on
        self.sigma = sigma                  # parameter for calculating gaussian map 


    def get_gaussian_map(self):
        '''
        returns the gaussian map response
        '''

        x, y, w, h = self.roi
        center_x = x + w/2
        center_y = y + h/2
        
        # create a rectangular grid out of two given one-dimensional arrays
        xx, yy = np.meshgrid(np.arange(x, x+w), np.arange(y, y+h))

        # calculating distance of each pixel from roi center
        dist = (np.square(xx - center_x) + np.square(yy - center_y)) / (2*self.sigma)
        
        response = np.exp(-dist)
        response = (response - response.min()) / (response.max() - response.min())
        
        return response


    def get_G(self):
        '''
        returns the fft2 (2D Discrete Fast Fourier Transform) of the gaussian response map
        '''

        x, y, w, h = self.roi
        g = self.get_gaussian_map()
        self.G = np.fft.fft2(g)


    def preprocessing(self,f):
        '''
        preprocessing as mentioned in MOSSE paper
        '''

        h, w = f.shape
        
        # transformation using a log function which helps with low contrast lighting situations.
        img = np.log(f+1)
        
        # pixel values are normalized to have a mean value of 0.0 and a norm of 1.0.
        img = (img - np.mean(img)) / (np.std(img))
        
        # image is multiplied by a cosine window which gradually reduces the pixel values near the edge to zero.
        window_col = np.hanning(w)  # returns a hanning window....Hanning window is a taper formed by using a weighted cosine
        window_row = np.hanning(h)
        col_mask, row_mask = np.meshgrid(window_col, window_row)
        window = col_mask * row_mask
        img = img * window
        
        return f


    def get_rand_affine(self):
        '''
        returns a random affine transformation
        '''

        # center pixel coordinates
        c_x = self.frame.shape[0]//2
        c_y = self.frame.shape[1]//2

        min_angle = 10
        angle = np.random.uniform(-min_angle, min_angle)

        rotate_mat = cv2.getRotationMatrix2D((c_x, c_y), angle, 1)
        rot_img = cv2.warpAffine(self.frame, rotate_mat, (self.frame.shape[1], self.frame.shape[0]))

        x, y, w, h = self.roi
        
        return rot_img[y:y+h, x:x+w]
        

    def pre_training(self):
        '''
        training the filter over initial frame
        G = F . H*
        Ai = ΣGi.Fi*
        Bi = ΣFi.Fi*
        Hi* = Ai / Bi
        '''

        x, y, w, h = self.roi
        
        self.get_G()    # initializes centered 2D gaussian map
        
        f = self.preprocessing(self.frame[y:y+h, x:x+w])
        F = np.fft.fft2(f)
        
        self.Ai = self.G * np.conjugate(F)
        self.Bi = F * np.conjugate(F)
        
        for i in range(self.train_num):
            f = self.preprocessing(self.get_rand_affine())
            F = np.fft.fft2(f)
            
            self.Ai += self.G * np.conjugate(F)
            self.Bi += F * np.conjugate(F)


    def update_frame(self, frame):
        self.frame = frame
    
    
    def get_new_roi(self):
        '''
        Get new position of roi by applying filter
        G = F . H*
        '''

        x, y, w, h = self.roi
        
        self.Hi = self.Ai / self.Bi
        
        f = self.preprocessing(self.frame[y:y+h, x:x+w])
        F = np.fft.fft2(f)
        
        self.G = self.Hi * F
        g = np.fft.ifft2(self.G)
        
        # mapping gaussian map to 0-1
        g = (g - g.min()) / (g.max() - g.min())
        print('g',g)
        max_value = np.max(g)
        max_pos = np.where(g == max_value)
        dy = int(np.mean(max_pos[0]) - g.shape[0] / 2)
        dx = int(np.mean(max_pos[1]) - g.shape[1] / 2)

        psr = self.psr(g)
        if psr <= 6:
            dx=0
            dy=0
            print('OBJECT OCCLUDED!!')
        print('psr:',psr)
        return (x+dx, y+dy, w, h) , psr


    def update_roi(self, roi):
        self.roi = roi

    
    def update(self):
        '''
        Update the filter using current frame features
        Ai = η Gi.Fi* + (1 − η) Ai−1
        Bi = η Fi.Fi* + (1 − η) Bi−1
        Hi* = Ai / Bi
        '''

        x, y, w, h = self.roi
        
        self.get_G()    # initializes centered 2D gaussian map

        f = self.preprocessing(self.frame[y:y+h, x:x+w])
        F = np.fft.fft2(f)
        
        # filters need to quickly adapt in order to follow objects. Running average is used for this purpose.
        self.Ai = self.learning_rate * self.G * np.conjugate(F) + (1 - self.learning_rate) * self.Ai
        self.Bi = self.learning_rate * F * np.conjugate(F) + (1 - self.learning_rate) * self.Bi
    
    def psr(self,g):
        g_max = np.max(g)
        x, y, w, h = self.roi
        center_x = x + w//2
        center_y = y + h//2
        mask = np.ones(g.shape,dtype=np.bool)
        mask[center_x-5:center_x+6,center_y-5:center_y+6] = False
        g = g.flatten()
        mask = mask.flatten()
        sidelobe = g[mask]
        mn = np.mean(sidelobe)
        sd = np.std(sidelobe)
        psr = (g_max-mn)/sd
        return psr