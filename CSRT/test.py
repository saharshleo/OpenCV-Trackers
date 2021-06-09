import cv2
from CSRT import *


# img = cv2.imread('/home/prathamesh/OpenCV-Trackers/assets/dog_test.png')
# roi = cv2.selectROI('select roi',img)
# # print(roi)
# # print(type(roi))
# # print(type(img))
# # print(img.shape)
# print(roi)

# tracker = CSRT(img, roi, 40)
# tracker.set_roiImage()
# tracker.get_spatial_reliability_map()
# tracker.generate_features(8, 2)
# tracker.update_H()
# tracker.calculate_g_cap_and_channel_weights()
# next_roi = tuple(tracker.draw_bbox())
# print(next_roi)

video = cv2.VideoCapture("/home/prathamesh/OpenCV-Trackers/assets/chaplin.mp4")

_, frame1 = video.read()
roi = cv2.selectROI('Select Roi', frame1)
# tracker = CSRT(frame1, roi, 10)
while(video.isOpened()):
    _, frame = video.read()
    print("roi ", roi)
    tracker = CSRT(frame, roi, 1)
    tracker.set_roiImage()
    tracker.get_spatial_reliability_map()
    tracker.generate_features(8, 2)
    tracker.update_H()
    tracker.calculate_g_cap_and_channel_weights()
    print("Here")
    roi = tuple(tracker.draw_bbox())
    print("Here too")
    print("roi", roi)


# tracker.set_roiImage()
# tracker.get_spatial_reliability_map()
# tracker.generate_features(8, 2)
# tracker.update_H()
# tracker.calculate_g_cap_and_channel_weights()
# roi = tuple(tracker.draw_bbox())

