import cv2
from skimage.transform import resize
from CSRT import*


img = cv2.imread('/home/prathamesh/OpenCV-Trackers/assets/dog_test.png')
roi = cv2.selectROI('select roi',img)
# print(roi)
# print(type(roi))
# print(type(img))
# print(img.shape)
tracker = CSRT(img,roi)
tracker.set_roiImage()
tracker.get_spatial_reliability_map()
tracker.generate_features(9, 4)
tracker.update_H()
cv2.waitKey(0)
