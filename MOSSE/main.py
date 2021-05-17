import cv2
from mosse import *

video = cv2.VideoCapture(0)

ret,frame = video.read()
roi = cv2.selectROI(frame)

img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

tracker = MOSSE(img_gray, roi, learning_rate=0.125, train_num=8, sigma=100)

tracker.pre_training()  # get initial correlation filter

# Tracking Loop
while(video.isOpened()):

    ret,frame = video.read()

    frame_h, frame_w ,_=frame.shape
    frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    tracker.update_frame(frame_gray)
    new_roi,psr = tracker.get_new_roi()     # apply H and get new bounding box on object

    x,y,w,h = new_roi
    
    # limiting the coordinates in case the tracked roi is going outside the frame
    x = max(0, x)
    y = max(0, y)
    
    if x + w >= frame_w:
        x = frame_w-w
    if y + h >= frame_h:
        y = frame_h-h

    new_roi = (x, y, w, h)
    tracker.update_roi(new_roi)

    if psr > 8:
        tracker.update()
    
    cv2.rectangle(frame, (x, y),(x + w, y + h), (0,255,0), 2)
    cv2.imshow('frame',frame)

    k = cv2.waitKey(1) & 0xff
    if k == 27 : 
        break

video.release()
cv2.destroyAllWindows()