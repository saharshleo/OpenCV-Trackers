import cv2
from skimage.transform import resize
from CSRT import *


img = cv2.imread('/home/prathamesh/OpenCV-Trackers/assets/dog_test.png')
roi = cv2.selectROI('select roi',img)
# print(roi)
# print(type(roi))
# print(type(img))
# print(img.shape)

tracker = CSRT(img, roi, 2)
tracker.set_roiImage()
tracker.get_spatial_reliability_map()
tracker.generate_features(8, 2)
tracker.update_H()
tracker.calculate_g_cap_and_channel_weights()
# tracker.apply_csrt()

# x, y, w, h = tracker.get_new_roi()
# cv2.rectangle(img, (x, y),(x + w, y + h), (0,255,0), 2)
# cv2.imshow('csrt',img)
# cv2.waitKey(0)
