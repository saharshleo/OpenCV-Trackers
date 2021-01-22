import numpy as np
import cv2
from PIL import Image
from skimage.transform import integral_image
from skimage.feature import haar_like_feature
from skimage.feature import haar_like_feature_coord
from skimage.feature import draw_haar_like_feature


img = np.ones((6, 6), dtype=np.uint8)
print('IMAGE:')
print(img)



print('IMAGE DIMENSIONS')
print('{} * {}'.format(img.shape[0],img.shape[1]))

img_ii = integral_image(img)
print(img_ii)

feature_coord, feature_type = haar_like_feature_coord(width=img.shape[1],height=img.shape[0],feature_type="type-3-x")
print('COORDINATES OF EACH FEATURE')
print((feature_coord))
print(feature_type)
print(len(feature_type))
print('VALUE OF IMAGE AFTER APPLYING THE FEATURES')
feature = haar_like_feature(img_ii, 0, 0, img.shape[1],img.shape[0] , "type-3-x")
print(feature)
print(len(feature))


