import cv2
import time

# Helper class
from boosting import Boosting

if __name__ == '__main__':
    print("[INFO] Starting...")

    video_path = 'input/highway.mp4'

    # Initialize cam
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), 'Cannot capture source'

    frame_width = int(cap.get(3)) 
    frame_height = int(cap.get(4)) 
    size = (frame_width, frame_height) 
    result = cv2.VideoWriter('output/out.avi',  
                         cv2.VideoWriter_fourcc(*'MJPG'), 
                         30, size)

    # Draw roi
    ret, frame = cap.read()
    # roi --> topLeftX, topLeftY, Width, Height 
    object_roi = cv2.selectROI("frame", frame, showCrosshair=False, fromCenter=False)
    cv2.destroyAllWindows()
    
    # Tracker object
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bo = Boosting(frame_gray, object_roi, num_rows=None, num_features=12500, num_selectors=50, num_to_replace=1)

    # Build features
    bo.build_features()     # feature_list gets initialized with random Haar features

    # Make weak classifiers from features
    bo.get_weak_classifiers()   # compute theta, polarity of above generated features

    # Initialize Selector pool
    bo.init_selector_pool()     # distribute features among selectors

    # Get strong classifier
    bo.get_strong_classifier()  # get strong classifier for each selector

    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            bo.frame = frame_gray

            # Get search region
            # right now, using blue rect

            # Get confidence map
            s = time.time()
            bo.get_confidence_map()
            print("[DEBUG] Time for confidence map: {}".format(time.time() - s))

            # Get bbox
            bo.get_bbox()

            # Draw on frame
            x, y, w, h = bo.object_roi
            frame = cv2.rectangle(frame, (x,y), (x+w,y+h), (255, 0, 0), 2)

            cv2.imshow("frame", frame)
            # result.write(frame)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break

            # Update strong classifier
            bo.update_strong_classifier()
            # break

        else:   # if frame not read
            break     

    cap.release()
    result.release()
    cv2.destroyAllWindows()