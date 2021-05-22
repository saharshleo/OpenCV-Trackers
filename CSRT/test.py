import cv2
from skimage.transform import resize
from CSRT import*

video = cv2.VideoCapture(0)

_,frame = video.read()

img = cv2.imread('OpenCV-Trackers/assets/dog_test.png')
img = frame
roi = cv2.selectROI('select roi',img)
print(roi)
print(img.shape)
tracker = CSRT(img,roi)
tracker.set_roiImage()
tracker.get_spatial_reliability_map()
cv2.waitKey(0)