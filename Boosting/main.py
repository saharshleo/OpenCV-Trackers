import numpy as np
from skimage.transform import integral_image
import cv2
from FeatureHaar import*
from Boosting import*

video = cv2.VideoCapture('OpenCV-Trackers/assets/chaplin.mp4')

ret, frame = video.read()
roi = cv2.selectROI(frame)
frame = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
roi_ls = [roi[0],roi[1],roi[0]+roi[2],roi[1]+roi[3]]
tracker = Boosting(frame,roi_ls,150,12500,50,2)
tracker.get_search_region()
tracker.set_ii_searchregion()
tracker.build_features()
tracker.init_selector_pool()
tracker.train_weak_classifier()
tracker.get_strong_classifier()

while(video.isOpened()):
    ret, frame = video.read()

    frame = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
    tracker.update_frame(frame)
    tracker.get_confidence_map()
    new_roi = tracker.get_meanshift_bbox()
    tracker.update_roi(new_roi)
    tracker.get_search_region()
    cv2.rectangle(frame,(new_roi[0],new_roi[1]),(new_roi[2],new_roi[3]),(0,255,0),2)
    cv2.imshow('frame',frame)

    k = cv2.waitKey(1) & 0xff
    if k == 27 : 
        break

video.release()
cv2.destroyAllWindows()

