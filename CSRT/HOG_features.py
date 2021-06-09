import cv2
from skimage.feature import hog
import numpy as np

def get_hog_features(img, des_orientations, des_pixels_per_cell):

    fd, hog_image = hog(img, orientations = des_orientations, pixels_per_cell = (des_pixels_per_cell,des_pixels_per_cell), cells_per_block=(2, 2), visualize=True,multichannel=True)
    # np.resize(fd, hog_image.shape)
    # print("fd shape", fd.shape)
    # print("hog shape", hog_image.shape)
    np.set_printoptions(threshold=np.inf)
    print("Hog image", hog_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return hog_image


