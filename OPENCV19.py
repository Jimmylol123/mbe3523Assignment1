# Save this file to Github as OpenCV-19-YOLO-part1.py

import cv2
import numpy as np

cap = cv2.VideoCapture(1)

# Create an empty list - classes[] and point the classesFile to 'coco80.names'
classesFile = 'coco.names'
classes = []
# Load all classes in coco80.names into classes[]
with open(classesFile, 'r') as f:
    classes = f.read().splitlines()
    print(classes)
    print(len(classes))

# Load the configuration and weights file
net = cv2.dnn.readNetFromDarknet('yolov3.cfg','yolov3.weights')
# Use OpenCV as backend and use CPU
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
bboxes = []
confs = []
classids = []

while True:
    success , img = cap.read()
    width,height,channel = img.shape
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (320, 320), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)

    layerNames = net.getLayerNames()
    print(layerNames)

    outputlayers = net.getUnconnectedOutLayersNames()
    print(outputlayers)

    OutputsLayer = net.forward(outputlayers)
    print(OutputsLayer[0].shape)
    #print(OutputsLayer[1].shape)
    #print(OutputsLayer[2].shape)
    #print(OutputsLayer[0][0])
    for output in OutputsLayer:
        for detection in output:
            scores = detection[5:]
            classid = np.argmax(scores)
            confidence = scores[classid]
            if confidence>0.4:
                cx = int(detection[0]*width)
                cy = int(detection[1]*height)
                w = int(detection [2]*width)
                h = int(detection[3]*height)
                x = int (cx - w/2)
                y = int (cy - h/2)
                bboxes.append([x,y,w,h])
                confs.append(float(confidence))
                classids.append(classid)
    indices = cv2.dnn.NMSBoxes(bboxes,confs,0.4,0.4)
    print(indices)
    font = cv2.FONT_HERSHEY_SIMPLEX
    if len(indices) >0:
        for i in indices.flatten():
            x,y,w,h = bboxes[i]
            label = str(classes[classids[i]])
            confidence= str(round(confs[i],2))
            cv2.rectangle(img, (x, y), (x + w, y + h), (255,0,0), 2)
            cv2.putText(img, label + " " + confidence, (x, y + 20), font, 2, (255, 255, 255), 2)

    cv2.imshow('Image', img)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()