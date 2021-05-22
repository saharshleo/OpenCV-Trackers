import cv2
from skimage.feature import hog
from skimage.transform import resize

def get_hog_features(image):
    resized_img = resize(image,(128,64))
    fd, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(8, 8), 
                    cells_per_block=(2, 2), visualize=True,multichannel=True)
    return fd