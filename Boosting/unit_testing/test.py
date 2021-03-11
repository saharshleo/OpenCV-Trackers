import numpy as np
from skimage.transform import integral_image
import cv2
from FeatureHaar import*
from Boosting import*


# img = np.array([[ 1  ,2  ,3  ,4  ,5  ,6],
#                 [ 2  ,4  ,6  ,8 ,10 ,12],
#                 [ 3  ,6  ,9 ,12 ,15 ,18],
#                 [ 4  ,8 ,12 ,16 ,20 ,24],
#                 [ 5 ,10 ,15 ,20 ,25 ,30],
#                 [ 6 ,12 ,18 ,24 ,30 ,36]])
# img = np.array([[ 1  ,2  ,3  ,4  ,15 ,6],
#                 [ 2  ,4  ,6  ,8 ,1 ,2],
#                 [ 3  ,6  ,19 ,2 ,5 ,1],
#                 [ 6  ,3 ,13 ,6 ,0 ,4],
#                 [ 0 ,11 ,15 ,21 ,2 ,3],
#                 [ 21 ,1 ,18 ,14 ,3 ,16]])
# print('IMAGE:')
# print(img)

# print('IMAGE DIMENSIONS')
# print('{} * {}'.format(img.shape[0],img.shape[1]))

# img_ii = integral_image(img)
# print(img_ii)
# roi = [1,1,4,4]
# bo = Boosting(img,roi,10,30,6,2)
# bo.get_search_region()
# bo.set_ii_searchregion()
# bo.build_features()
# bo.init_selector_pool()
# bo.train_weak_classifier()

img_org = cv2.imread("./assets/edge-detection.png")
roi = cv2.selectROI(img_org)
print(roi)
img = cv2.cvtColor(img_org,cv2.COLOR_RGB2GRAY)
roi_image = img[roi[1]:roi[1]+roi[3],roi[0]:roi[0]+roi[2]]
cv2.imshow("roiImage",roi_image)
roi_ls = [roi[0],roi[1],roi[0]+roi[2],roi[1]+roi[3]]
bo = Boosting(img,roi_ls,150,12500,50,2)
bo.get_search_region()
bo.set_ii_searchregion()
bo.build_features()
bo.init_selector_pool()
bo.train_weak_classifier()
bo.get_strong_classifier()
bo.get_confidence_map()
new_roi = bo.get_meanshift_bbox()
#new_roi = bo.get_bbox()
cv2.rectangle(img_org,(new_roi[0],new_roi[1]),(new_roi[2],new_roi[3]),(0,255,0),2)
cv2.imshow('newroi',img_org)
cv2.waitKey(0)