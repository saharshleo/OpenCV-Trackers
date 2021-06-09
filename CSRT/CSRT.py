from types import DynamicClassAttribute
import cv2
import HOG_features as hogfeat
import numpy as np
from colorname import *
from skimage.feature.peak import peak_local_max


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
    labels = (response - response.min()) / (response.max() - response.min())

    # labels = np.roll(labels, -int(np.floor(w / 2)), axis=1)
    # labels = np.roll(labels,-int(np.floor(h/2)),axis=0)
        
    return labels


def cap_func(element):
    return np.fft.fft2(element)


class CSRT():

    def __init__(self, frame, roi, num_features, debug = False):
        self.debug = debug

        self.frame = frame
        self.roi = roi
        self.beta, self.λ, self.n, self.sigma, self.mu = 3, 0.01, 0.02, 100, 5
        self.g = get_gaussian_map(self.roi, self.sigma)
        
        if self.debug:
            cv2.imshow("Gaussian", self.g)
        
        # self.features = [0] * num_features
        # self.h_cap = [0] * num_features # Will contain h_cap values for all channels in sequential order
        # self.p = [] # This variable will store position of object in each frame.
        # self.channel_weights = [0] * num_features # Will store channel weights for all channels.
        # self.G_cap = [0] * num_features # Will store individual G_cap/g_tilda values for channels.
        # self.G_res = 0 # Will store resultant G_cap after using channel_reliability.

    def cos_window(self,sz):
        """
        width, height = sz
        j = np.arange(0, width)
        i = np.arange(0, height)
        J, I = np.meshgrid(j, i)
        cos_window = np.sin(np.pi * J / width) * np.sin(np.pi * I / height)
        """

        cos_window = np.hanning(int(sz[1]))[:, np.newaxis].dot(np.hanning(int(sz[0]))[np.newaxis, :])
        cos_window = cos_window.T
        cv2.imshow('cosine wndow',cos_window)
        return cos_window


    def init(self):
        self.set_roiImage()
        coswindow = self.cos_window(self.roi_img.shape)
        print('roi image',self.roi_img.shape)
        features = self.generate_features(8,4)
        features = features*coswindow[:,:,None]
        self.get_spatial_reliability_map()
        self.h = self.get_csrt_filter(features)
        response = np.fft.fft2(features)* (np.fft.fft2(self.h))
        G = np.real(np.fft.ifft2(response))

        
        channel_weights = np.max(G.reshape(G.shape[0]*G.shape[1],-1),axis=0)
        self.channel_weights = channel_weights/np.sum(channel_weights)

        G = np.sum(G * self.channel_weights[None,None,:],axis = 2)
        
        G = (G - G.min()) / (G.max() - G.min())
        cv2.imshow('G',G)
        max_value = np.max(G)
        max_pos = np.where(G == max_value)
        dy = int(np.mean(max_pos[0]) - G.shape[0] / 2)
        dx = int(np.mean(max_pos[1]) - G.shape[1] / 2)
        print(dx,dy)

    def get_new_roi(self,frame):
        self.frame = frame
        self.set_roiImage()
        coswindow = self.cos_window(self.roi_img.shape)
        features = self.generate_features(8,4)
        features = features*coswindow[:,:,None]

        response = np.fft.fft2(features)* (np.fft.fft2(self.h))
        response = np.real(np.fft.ifft2(response))
        G = np.sum(response * self.channel_weights[None,None,:],axis = 2)
        G = (G - G.min()) / (G.max() - G.min())
        self.g = G
        max_value = np.max(G)
        max_pos = np.where(G == max_value)
        dy = int(np.mean(max_pos[0]) - G.shape[0] / 2)
        dx = int(np.mean(max_pos[1]) - G.shape[1] / 2)
        print(dx,dy)

        self.channel_det = np.ones(response.shape[2])

        for i in range(response.shape[2]):
            norm_img = response[:,:,i] - response[:,:,i].min() / response[:,:,i].max() - response[:,:,i].min()
            peak_locs = peak_local_max(norm_img,min_distance=5)
            if len(peak_locs)<2:
                    continue
            vals=reversed(sorted(norm_img[peak_locs[:,0],peak_locs[:,1]]))
            second_max_val=None
            max_val=None
            for index,val in enumerate(vals):
                if index==0:
                    max_val=val
                elif index==1:
                    second_max_val=val
                else:
                    break
            self.channel_det[i] = max(1- (second_max_val / (max_val+ 1e-10)), 0.5)
        x,y,w,h = self.roi
        self.roi = (x+dx, y+dy, w, h)
        return self.roi

    def update(self):
        coswindow = self.cos_window(self.roi_img.shape)
        features = self.generate_features(8,4)
        features = features*coswindow[:,:,None]
        self.get_spatial_reliability_map()
        h = self.get_csrt_filter(features)


        response = np.fft.fft2(features)* (np.fft.fft2(self.h))
        G = np.real(np.fft.ifft2(response))

        channel_weights = np.max(G.reshape(G.shape[0]*G.shape[1],-1),axis=0)
        channel_weights = channel_weights*self.channel_det
        channel_weights = channel_weights/np.sum(channel_weights)

        self.h = (1-self.n)*self.h + self.n*h
        self.channel_weights = (1-self.n)*self.channel_weights + self.n*channel_weights
        self.channel_weights = self.channel_weights/np.sum(self.channel_weights)



    def set_roiImage(self):
        self.x, self.y, self.width, self.height = self.roi
        self.roi_img = self.frame[self.y : self.y + self.height, self.x : self.x + self.width]


    def generate_features(self, des_orientations, des_pixels_per_cell):
        # for i in range(len(self.features)):
        #     self.f = hogfeat.get_hog_features(self.roi_img, des_orientations + i, des_pixels_per_cell + i)
        #     self.features[i] = self.f

        # if self.debug:
        #     cv2.imshow('hog_image', self.features[0])
        
        features = extract_cn_feature_byw2c(self.roi_img)
        return features


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
        

    def get_csrt_filter(self,features):
        
        f_hat = cap_func(features)
        f_hat_conjugate = np.conjugate(f_hat)
        g_hat = cap_func(self.g)
        g_hat_conjugate = np.conjugate(g_hat)

        h = (f_hat * np.conjugate(g_hat)[:,:,None]) / (f_hat * np.conjugate(f_hat) + self.λ)

        self.D = h.shape[0] * h.shape[1]

        self.mu_i = self.mu
        I_hat = np.zeros_like(h)

        for i in range(4):
            hm = h * self.m[:,:,None]
            hm_hat = cap_func(hm)

            # Eqn 12
            hc = (f_hat * g_hat_conjugate[:,:,None] + (self.mu * hm_hat - I_hat)) / (f_hat_conjugate * f_hat + self.mu_i)
        
            # Eqn 13
            h = (self.m[:,:,None] * np.fft.ifft((I_hat + self.mu * hc))) / (self.λ / (2 * self.D) + self.mu_i)
        
            hc_cap = cap_func(hc)
            h_cap = cap_func(h)
            # Eqn 11
            I_hat = I_hat + self.mu * (hc_cap - h_cap)
        
            self.mu_i *= self.beta
        
        return h



