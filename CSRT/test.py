import cv2
from CSRT import *


# img = cv2.imread('/home/aman/Desktop/SMORT/OpenCV-Trackers/assets/dog_test.png')
# roi = cv2.selectROI('select roi',img)
# print(roi)
# print(type(roi))
# print(type(img))
# print(img.shape)
# print(roi)

# tracker = CSRT(img, roi, 10,True)
# tracker.set_roiImage()
# tracker.get_spatial_reliability_map()
# tracker.generate_features(8, 4)
# tracker.update_H()
# tracker.calculate_g_cap_and_channel_weights()
# next_roi = tuple(tracker.draw_bbox())
# print(next_roi)
# cv2.waitKey(0)

video = cv2.VideoCapture(0)

_, frame1 = video.read()
print('frame',frame1.shape)
roi = cv2.selectROI('Select Roi', frame1)
tracker = CSRT(frame1, roi, 1,True)
tracker.init()
# tracker = CSRT(frame1, roi, 10)
while(video.isOpened()):
    _, frame = video.read()
    new_roi = tracker.get_new_roi(frame)
    tracker.update()
    x,y,w,h = new_roi
    cv2.rectangle(frame, (x, y),(x + w, y + h), (0,255,0), 2)
    cv2.imshow('tracker',frame)
    if cv2.waitKey(10) == ord('x'):
        cv2.destroyAllWindows()
        video.release()

# tracker.set_roiImage()
# tracker.get_spatial_reliability_map()
# tracker.generate_features(8, 2)
# tracker.update_H()
# tracker.calculate_g_cap_and_channel_weights()
# roi = tuple(tracker.draw_bbox())

