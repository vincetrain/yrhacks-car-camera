'''
TODO:
X Read data from camera / video 
X Computer vision to detect cars 
    - Calculate speed of each car and compare to local road's speed limit
    - Read license plate of each car and log the license,time,location
        - Only log if same data doesnt already exist
            - Log time data to nearest half hour?
    - Compare license to a database of stolen licenses
        - Alert a user? / log to new database if stolen license matches.
'''

import time
import numpy as np
import cv2
import licenseocr

CONFIDENCE_THRESHOLD = 0.8
NMS_THRESHOLD = 0.6

VEHICLES = ['car', 'truck', 'motorbike', 'bus']

## initializes model and network

class_file = open('./data/coco.names')
class_names = class_file.read().splitlines()

COLORS = np.random.uniform(0, 255, size=(len(class_names), 3))

model_config = 'cfg/yolov4.cfg'
model_weights = 'yolov4.weights'

net = cv2.dnn.readNetFromDarknet(model_config, model_weights)

## tells net to PREFER USING MY GPU but cv2 is not compiled to work with gpu... and im too lazy to do that...
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

## main loop

cv2.namedWindow('Camera Footage', cv2.WINDOW_NORMAL)

while(True):
    cam = cv2.VideoCapture('example-footage.mp4')

    while(cam != None):
        ret, frame = cam.read()
        cam_width, cam_height, _ = frame.shape
        if not ret:
            break
        
        start = time.time()
        
        classes, scores, boxes = model.detect(frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

        end = time.time()
        
        fps = 1 / (end-start)
        
        for (class_id, score, box) in zip(classes, scores, boxes):
            color = COLORS[int(class_id) % len(COLORS)]
            class_name = class_names[class_id]
            if (class_name in VEHICLES):
                plate = licenseocr.get_plate(frame[box[1]:box[1]+box[3], box[0]:box[0]+box[2]])
                if (plate != None or plate != ''):
                    print(plate)
                label = "%s : %f" % (class_name, score)
                cv2.rectangle(frame, box, color, 2)
                cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        
        fps_label = f'FPS: {fps:.2f}'
        cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        cv2.imshow('Camera Footage', frame)
        
        key = cv2.waitKey(1)
        
        if key == ord('q'):
            break
        
        if key == ord(' '):
            while (cv2.waitKey(-1) != ord(' ')):
                print('Video paused. Press [SPACE] to unpause.')
        
    cam.release()

    cv2.destroyAllWindows()

