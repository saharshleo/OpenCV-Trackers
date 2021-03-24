import cv2
from mosse import*

video = cv2.VideoCapture(0)

ret,frame = video.read()
roi = cv2.selectROI(frame)

img_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

tracker = MOSSE(img_gray,roi,0.125,8,100)

tracker.pre_training()

while(video.isOpened()):

    ret,frame = video.read()

    frame_h, frame_w ,_=frame.shape
    frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    tracker.update_frame(frame_gray)
    new_roi = tracker.get_new_roi()

    x,y,w,h = new_roi
    if x < 0:
        x = 0
    if y < 0:
        y = 0
    if x+w >= frame_w:
        x = frame_w-w
    if y+h >= frame_h:
        y = frame_h-h

    new_roi = (x,y,w,h)

    tracker.update_roi(new_roi)
    tracker.update()
    
    cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
    cv2.imshow('frame',frame)

    k = cv2.waitKey(1) & 0xff
    if k == 27 : 
        break

video.release()
cv2.destroyAllWindows()