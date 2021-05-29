import h5py
import numpy as np
import scipy.io as sio
import cv2

from im2c import* 

w2c = sio.loadmat('Add the path of w2c.mat ',struct_as_record = False)
w2c = w2c['w2c']

img = cv2.imread('Add the path')

out = im2c(img,w2c,-1)