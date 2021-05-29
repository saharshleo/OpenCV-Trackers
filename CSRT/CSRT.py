import cv2
from skimage import feature
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

    def __init__(self, frame, roi, debug=True):
        self.debug = None

        self.frame = frame
        self.roi = roi
        self.sigma = 100
        self.mu = 5
        self.beta, self.λ = 3, 0.01
        self.g = get_gaussian_map(self.roi, self.sigma)
        
        if self.debug:
            cv2.imshow("Gaussian", self.g)
        
        self.features = []
        self.h_cap = [1, 1] # Will contain h_cap values for all channels in sequential order
        self.p = [] # This variable will store position of object in each frame.
        self.channel_weights = [] # Will store channel weights for all channels.


    def set_roiImage(self):
        x,y,w,h = self.roi
        self.roi_img = self.frame[y:y+h, x:x+w]
        self.h = np.zeros((self.roi_img.shape[0], self.roi_img.shape[1]))
        self.I = np.zeros_like(self.h)


    def generate_features(self, des_orientations, des_pixels_per_cell):
        # self.features.append(hogfeat.get_hog_features(self.roi_img, des_orientations,
        #     des_pixels_per_cell)) 
        self.f = hogfeat.get_hog_features(self.roi_img, des_orientations, des_pixels_per_cell)
        self.features.append(self.f)
      
        if self.debug:
            cv2.imshow('hog_image', self.features[0])


    def get_spatial_reliability_map(self):
        self.mask = np.zeros((self.frame.shape[0], self.frame.shape[1]))
        self.fg_Model = np.zeros((1,65), dtype="float")
        self.bg_Model = np.zeros((1,65), dtype="float")

        (self.mask,self.fg_Model, self.bg_Model) = cv2.grabCut(self.frame, self.mask, self.roi, 
            self.bg_Model, self.fg_Model, iterCount = 5, mode = cv2.GC_INIT_WITH_RECT)
        
        outputmask = np.where((self.mask == cv2.GC_BGD) | (self.mask == cv2.GC_PR_BGD),0,1)
        outputmask = (outputmask).astype("uint8") * 255        
        output = cv2.bitwise_and(self.frame, self.frame, mask=outputmask)
        valuemask = (self.mask == cv2.GC_PR_BGD).astype("uint8") * 255
        # cv2.imshow("valuemask",valuemask)
        # cv2.imshow("mask",outputmask)
        # cv2.imshow("grabcut",output)
        
        x, y, w, h = self.roi
        self.m = valuemask[y:y+h, x:x+w] 
        if self.debug:
            cv2.imshow("Spatial reliability map", self.m) 
        
        indice_one = np.where(self.m == 255)
        indice_zero = np.where(self.m == 0)
        self.m[indice_one] = 0
        self.m[indice_zero] = 1


    def initialize_fgh(self):
        f_hat = cap_func(self.features[0])
        g_hat = cap_func(self.g)
        # self.h = (f_hat * np.conjugate(g_hat)) / (f_hat * np.conjugate(f_hat) + self.λ)
        self.h = self.g
        

    def update_H(self):
        self.initialize_fgh()
        
        for channel_index in range(len(self.features)):
            f_hat = cap_func(self.features[channel_index])
            f_hat_conjugate = np.conjugate(f_hat)
        
            g_hat = cap_func(self.g)
            g_hat_conjugate = np.conjugate(g_hat)

            self.D = self.h.shape[0] * self.h.shape[1]

            self.mu_i = self.mu
            I_hat = cap_func(self.I)

            for i in range(4):
                self.hm = self.h * self.m
                hm_hat = cap_func(self.hm)

                # Eqn 12
                self.hc = (f_hat * g_hat_conjugate + (self.mu * hm_hat - I_hat)) / (f_hat_conjugate * f_hat + self.mu_i)
            
                # Eqn 13
                self.h = (self.m * np.fft.ifft((I_hat + self.mu * self.hc))) / (self.λ / (2 * self.D) + self.mu_i)
            
                hc_cap = cap_func(self.hc)
                h_cap = cap_func(self.h)
                # Eqn 11
                I_hat = I_hat + self.mu * (hc_cap - h_cap)
            
                self.mu_i *= self.beta
            self.h_cap[channel_index] = h_cap


    def get_new_roi(self, get_max_G_value = None):
        x, y, w, h = self.roi

        f_hat = cap_func(self.features[0])
        res = f_hat * self.h
        G = np.fft.ifft2(res)
        G = (G - G.min()) / (G.max() - G.min())
        
        _G = np.absolute(G)
        _G = np.array(_G * 255, dtype=np.dtype('uint8'))
        if self.debug:
            cv2.imshow("Output", _G)

        max_value = np.max(G)
        max_pos = np.where(G == max_value)
        if (get_max_G_value):
            return max_value, np.max(np.delete(G, max_pos))
        dy = int(np.mean(max_pos[0]) - G.shape[0] / 2)
        dx = int(np.mean(max_pos[1]) - G.shape[1] / 2)

        if self.debug:
            print("roi", self.roi)
            print('G shape', self.features[0].shape)
            print('dx, dy',(dx, dy)) 
        return (x+dx, y+dy, w, h)

    def channel_reliability(self):
        w_learn, pd_max_2 = self.get_new_roi(True)
        w_deter = max(0.5, 1 - pd_max_2 / w_learn)
        print(w_learn, pd_max_2)
        return w_learn * w_deter

    def apply_csrt(self):
        self.p = self.get_new_roi()
        # print(self.p)
        self.channel_reliability()
