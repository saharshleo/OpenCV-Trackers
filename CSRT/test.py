import cv2
from skimage.transform import resize
from CSRT import *


img = cv2.imread('/home/prathamesh/OpenCV-Trackers/assets/dog_test.png')
roi = cv2.selectROI('select roi',img)
# print(roi)
# print(type(roi))
# print(type(img))
# print(img.shape)

tracker = CSRT(img,roi)
tracker.set_roiImage()
tracker.get_spatial_reliability_map()
tracker.generate_features(9, 2)
tracker.generate_features(10, 2)
tracker.update_H()
tracker.apply_csrt()

x, y, w, h = tracker.get_new_roi()
cv2.rectangle(img, (x, y),(x + w, y + h), (0,255,0), 2)
cv2.imshow('csrt',img)
cv2.waitKey(0)
