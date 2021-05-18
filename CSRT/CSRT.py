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

class CSRT():

    def __init__(self,frame,roi):
        self.frame = frame
        self.roi = roi
        self.sigma = 100
        self.g = get_gaussian_map(self.roi, self.sigma)
        print(self.g)

    def set_roiImage(self):
        x,y,w,h = self.roi
        self.roi_img = self.frame[y:y+h,x:x+w]

    def generate_features(self, des_orientations, des_pixels_per_cell):
        self.features = []
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
        output = cv2.bitwise_and(self.frame,self.frame,mask=outputmask)
        valuemask = (self.mask == cv2.GC_PR_BGD).astype("uint8")*255
        print(self.roi_img.shape)
        cv2.imshow("valuemask",valuemask)
        cv2.imshow("mask",outputmask)
        cv2.imshow("grabcut",output)

    def preprocessing():
        pass

    def update_H():
        pass

    def channel_reliability():
        pass
