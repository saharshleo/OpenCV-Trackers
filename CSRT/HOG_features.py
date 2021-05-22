import cv2
from skimage.feature import hog
# from skimage.transform import resize

def get_hog_features(image, des_orientations, des_pixels_per_cell):
    fd, hog_image = hog(image, orientations = des_orientations,
        pixels_per_cell = (des_pixels_per_cell, des_pixels_per_cell), cells_per_block=(2, 2),
        visualize=True,multichannel=True)
    return fd
