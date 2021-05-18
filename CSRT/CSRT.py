import cv2
from skimage.feature import hog
from skimage.transform import resize
import HOG_features as hogfeat
import numpy as np

def get_gaussian_map(roi, sigma):
    '''
    returns the gaussian map response
    '''

    x, y, w, h = roi
    center_x = x + w/2
    center_y = y + h/2
        
    # create a rectangular grid out of two given one-dimensional arrays
    xx, yy = np.meshgrid(np.arange(x, x+w), np.arange(y, y+h))

    # calculating distance of each pixel from roi center
    dist = (np.square(xx - center_x) + np.square(yy - center_y)) / (2 * sigma)
        
    response = np.exp(-dist)
    response = (response - response.min()) / (response.max() - response.min())
        
    return response

def cap_func(element):
    return np.fft.fft2(element)

class CSRT():

    def __init__(self,frame,roi):
        self.frame = frame
        self.roi = roi
        self.sigma = 100
        self.g = get_gaussian_map(self.roi, self.sigma)
        # print(self.g)
        self.features = []

    def set_roiImage(self):
        x,y,w,h = self.roi
        self.roi_img = self.frame[y:y+h,x:x+w]
        self.h = np.zeros((self.roi_img.shape[0], self.roi_img.shape[1]))
        self.I = np.zeros_like(self.h)


    def generate_features(self, des_orientations, des_pixels_per_cell):
        self.features.append(hogfeat.get_hog_features(self.roi_img, des_orientations,
            des_pixels_per_cell)) 

    def get_spatial_reliability_map(self):
        self.mask = np.zeros((self.frame.shape[0],self.frame.shape[1]))
        self.fg_Model = np.zeros((1,65), dtype="float")
        self.bg_Model = np.zeros((1,65), dtype="float")

        (self.mask,self.fg_Model, self.bg_Model) = cv2.grabCut(self.frame, self.mask, self.roi, 
            self.bg_Model, self.fg_Model, iterCount = 5, mode = cv2.GC_INIT_WITH_RECT)
        outputmask = np.where((self.mask == cv2.GC_BGD)| (self.mask == cv2.GC_PR_BGD),0,1)
        outputmask = (outputmask).astype("uint8")*255
        # print(type(outputmask))
        output = cv2.bitwise_and(self.frame,self.frame,mask=outputmask)
        valuemask = (self.mask == cv2.GC_PR_BGD).astype("uint8")*255
        cv2.imshow("valuemask",valuemask)
        cv2.imshow("mask",outputmask)
        cv2.imshow("grabcut",output)
        x, y, w, h = self.roi
        self.m = valuemask[y : y + h, x : x + w] 
        # print(type(self.m))
        indice_one = np.where(self.m == 255)
        indice_zero = np.where(self.m == 0)
        self.m[indice_one] = 0
        self.m[indice_zero] = 255
        # print(self.m)
        cv2.imshow("Spatial_map", self.m)

    def preprocessing():
        pass

    def update_H(self):
        h_prev = self.h
        I_prev = self.I
        self.hm = self.h * self.m 
        self.hc = np.zeros_like(self.hm)
        self.mu = 5
        self.beta, self.lambad = 3, 0.01
        self.D = self.h.shape[0] * self.h.shape[1]
        i = 0
        while (i < 4):
            i = 0
            print(type(self.features[0]))
            print(self.features[0].shape)
            np.reshape(self.features[0], (self.features[0].shape[0], 1))
            print(self.features[0].shape)
            f_hat = cap_func(self.features[0])
            g_hat = cap_func(self.g)
            hm_hat = cap_func(self.hm)
            I_hat = cap_func(self.I)
            self.hc = (f_hat * np.conjugate(g_hat) + (self.mu * hm_hat - I_hat)) / (f_hat * np.conjugate(f_hat) + self.mu)
            self.h = self.m * np.fft.ifft((I_hat + self.mu * self.hc) / (self.lambad / 2 * self.D + self.mu))
            I_hat = I_hat + self.mu * (cap_func(self.hc - self.h))
            self.mu *= self.beta
            i += 1


        

    def channel_reliability():
        pass
