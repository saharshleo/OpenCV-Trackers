import numpy as np
import cv2
import random
import sys
import matplotlib.pyplot as plt

class MOSSE():

    def __init__(self,frame,roi,learning_rate,train_num,sigma):
        self.frame = frame
        self.roi = roi # contains (top_left_x,top_left_y,width,height)
        self.learning_rate = learning_rate 
        self.train_num = train_num # number of training examples to be trained on
        self.sigma = sigma

    def update_roi(self,roi):
        self.roi = roi

    def update_frame(self,frame):
        self.frame = frame

    # returns a random affine transforamtion
    def get_rand_affine(self):
        # center pixel coordinates
        c_x = self.frame.shape[0]//2
        c_y = self.frame.shape[1]//2

        min_angle = 10
        angle = np.random.uniform(-min_angle,min_angle)

        rotate_mat = cv2.getRotationMatrix2D((c_x,c_y),angle,1)
        rot_img = cv2.warpAffine(self.frame,rotate_mat,(self.frame.shape[1],self.frame.shape[0]))

        x,y,w,h = self.roi
        return rot_img[y:y+h,x:x+w]
        
    # returns the fft2 of the gaussian response map
    def get_G(self):
        x,y,w,h = self.roi
        g = self.get_gaussian_map()
        self.G = np.fft.fft2(g)
        
    # training the filter over initial frame
    def pre_training(self):
        x,y,w,h = self.roi
        self.get_G()
        f = self.preprocessing(self.frame[y:y+h,x:x+w])
        F = np.fft.fft2(f)
        self.Ai = self.G * np.conjugate(F)
        self.Bi = F * np.conjugate(F)
        for i in range(self.train_num):
            f = self.preprocessing(self.get_rand_affine())
            F = np.fft.fft2(f)
            self.Ai = self.Ai + self.G * np.conjugate(F)
            self.Bi = self.Bi + F * np.conjugate(F)

    # Get new position of roi    
    def get_new_roi(self):
        x,y,w,h = self.roi
        self.Hi=self.Ai/self.Bi
        f=self.preprocessing(self.frame[y:y+h,x:x+w])
        F=np.fft.fft2(f)
        self.G=self.Hi*F
        g=np.fft.ifft2(self.G)
        # mapping gaussian map to 0-1
        g=(g - g.min()) / (g.max() - g.min())
        max_value = np.max(g)
        max_pos = np.where(g == max_value)
        dy = int(np.mean(max_pos[0]) - g.shape[0] / 2)
        dx = int(np.mean(max_pos[1]) - g.shape[1] / 2)
        return (x+dx,y+dy,w,h)
        
    # Update the filter using current frame features
    def update(self):
        x,y,w,h = self.roi
        f = self.preprocessing(self.frame[y:y+h,x:x+w])
        F = np.fft.fft2(f)
        self.get_G()
        # filters need to quickly adapt in order to follow objects. Running average is used for this purpose.
        self.Ai = self.learning_rate*self.G*np.conjugate(F) + (1-self.learning_rate)*self.Ai
        self.Bi = self.learning_rate*F*np.conjugate(F) + (1-self.learning_rate)*self.Bi
        
    # returns the gaussian map response
    def get_gaussian_map(self):
        x,y,w,h = self.roi
        center_x = x+w/2
        center_y = y+h/2
        xx, yy = np.meshgrid(np.arange(x,x+w), np.arange(y,y+h))
        dist = (np.square(xx - center_x) + np.square(yy - center_y))/(2*self.sigma)
        response = np.exp(-dist)
        response=(response - response.min()) / (response.max() - response.min())
        return response

    def preprocessing(self,f):
        h,w = f.shape
        # transformation using a log function which helps with low contrast lighting situations.
        img = np.log(f+1)
        # pixel values are normalized to have a mean value of 0.0 and a norm of 1.0.
        img = (img - np.mean(img))/(np.std(img))
        # image is multiplied by a cosine window which gradually reduces the pixel values near the edge to zero.
        window_col = np.hanning(w)
        window_row = np.hanning(h)
        col_mask,row_mask = np.meshgrid(window_col,window_row)
        window = col_mask * row_mask
        img=img*window
        return f
