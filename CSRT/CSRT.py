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

    def __init__(self, frame, roi, num_features, debug = False):
        self.debug = None

        self.frame = frame
        self.roi = roi
        self.sigma = 100
        self.mu = 5
        self.beta, self.λ, self.n = 3, 0.01, 0.02
        self.g = get_gaussian_map(self.roi, self.sigma)
        
        if self.debug:
            cv2.imshow("Gaussian", self.g)
        
        self.features = [0] * num_features
        self.h_cap = [0] * num_features # Will contain h_cap values for all channels in sequential order
        self.p = [] # This variable will store position of object in each frame.
        self.channel_weights = [0] * num_features # Will store channel weights for all channels.
        self.G_cap = [0] * num_features # Will store individual G_cap/g_tilda values for channels.
        self.G_res = 0 # Will store resultant G_cap after using channel_reliability.


    def set_roiImage(self):
        x,y,w,h = self.roi
        self.roi_img = self.frame[y:y+h, x:x+w]
        self.h = np.zeros((self.roi_img.shape[0], self.roi_img.shape[1]))
        self.I = np.zeros_like(self.h)


    def generate_features(self, des_orientations, des_pixels_per_cell):
        for i in range(len(self.features)):
            self.f = hogfeat.get_hog_features(self.roi_img, des_orientations + i, des_pixels_per_cell + i)
            self.features[i] = self.f
      
        if self.debug:
            cv2.imshow('hog_image', self.features)


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
        # self.initialize_fgh()
        
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
        print("h_cap for feature channels ", self.h_cap)
        print("length h_cap", len(self.h_cap))
        print("Lengths of h_cap", len(h_cap[0]), len(h_cap[1]))


    def calculate_g_cap_and_channel_weights(self):
        for channel_index in range(len(self.features)):
            f_hat = cap_func(self.features[channel_index])
            res = f_hat * cap_func(self.h_cap[channel_index])
            G = np.fft.ifft2(res)
            G = (G - G.min()) / (G.max() - G.min())
            self.G_cap[channel_index] = G

            max_value = np.max(G)
            G_nms = []
            for i in range(1, G.shape[0] - 1):
                for j in range(1, G.shape[1] - 1):
                    G_nms.append(np.max(G[i - 1 : i + 2, j - 1 : j + 2]))
            pd_max1_index = G_nms.index(max(G_nms))
            self.channel_weights[channel_index] = G_nms[pd_max1_index]
            G_nms.pop(pd_max1_index)
            self.channel_weights[channel_index] /= max(G_nms)
            self.channel_weights[channel_index] = 1 - self.channel_weights[channel_index]
            self.channel_weights[channel_index] *= max_value
            self.G_cap += max_value * self.channel_weights[channel_index]
        print("channel_weights", self.channel_weights)
        print("G_cap", self.G_cap)

        if self.debug:
            print("roi", self.roi)
            print('G shape', self.features[0].shape)

    def channel_reliability(self):
        w_learn, pd_max_2 = self.get_new_roi(True)
        w_deter = max(0.5, 1 - pd_max_2 / w_learn)
        print(w_learn, pd_max_2)
        return w_learn * w_deter

def calculate_final_g_cap(self):
    for channel_index in range(len(self.features)):
        self.G_res += self.G_cap[channel_index] * self.channel_weights[channel_index]
    return self.G_res

    def apply_csrt(self):
        self.p = self.get_new_roi()
        # print(self.p)
        self.channel_reliability()
