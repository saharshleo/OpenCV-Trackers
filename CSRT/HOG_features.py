import cv2
from skimage.feature import hog
import numpy as np

def get_hog_features(image, des_orientations, des_pixels_per_cell):
<<<<<<< HEAD
    fd, hog_image = hog(image, orientations = des_orientations,
        pixels_per_cell = (des_pixels_per_cell, des_pixels_per_cell), cells_per_block=(2, 2),
        visualize=True,multichannel=True)

    print('feature dim, hog_img dim, img_dim',fd.shape,hog_image.shape,image.shape)
=======
    fd, hog_image = hog(image, orientations = des_orientations, pixels_per_cell = (image.shape[0] / 100, image.shape[1] / 100), cells_per_block=(2, 2), visualize=True,multichannel=True)
    # np.resize(fd, hog_image.shape)
    # print("fd shape", fd.shape)
    # print("hog shape", hog_image.shape)
    # cv2.imshow("Hog image", hog_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
>>>>>>> cf1c2cddbf5965d8da79a627c15f46704184b354
    return hog_image



