import cv2
import numpy as np
from sort import *
# Load Yolo
print("LOADING YOLO")
net = cv2.dnn.readNet("./SORT/yolo/yolov3.weights", "./SORT/yolo/yolov3.cfg")
#save all the names in file o the list classes
with open("./SORT/yolo/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
#get layers of the network
layer_names = net.getLayerNames()
#Determine the output layer names from the YOLO model 
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
print("YOLO LOADED")

tracker = sort(max_age=1, min_hits=3,iou_threshold=0.3)
video = cv2.VideoCapture("./assets/chaplin.mp4")
while(video.isOpened()):
    ret,img = video.read()
    # Capture frame-by-frame
    # img = cv2.imread("test_img.jpg")
    # img = cv2.resize(img, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape

    # USing blob function of opencv to preprocess image
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416),
    swapRB=True, crop=False)
    #Detecting objects
    net.setInput(blob)
    outs = net.forward(output_layers)
    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id==0:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([int(x), int(y), int(x+w), int(y+h), confidence])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    #We use NMS function in opencv to perform Non-maximum Suppression
    #we give it score threshold and nms threshold as arguments.
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    boxes= np.array(boxes)
    boxes = tracker.update(boxes)
    # print(boxes)
    for i in range(len(boxes)):
        if i in indexes:
            x1, y1, x2, y2, objId = boxes[i]
            # print("{} {} {} {}".format(x1,y1,x2,y2))
            # label = str(classes[class_ids[i]]) + str(objId)
            label = str(objId)
            color = colors[class_ids[i]]
            cv2.rectangle(img, (int(x1),int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(img, label, (int(x1), int(y1 -5)),cv2.FONT_HERSHEY_SIMPLEX,1/2, color, 2)

    cv2.imshow("Image",img)
    k = cv2.waitKey(1) & 0xff
    if k == 27 : 
        break
video.release()
cv2.destroyAllWindows()