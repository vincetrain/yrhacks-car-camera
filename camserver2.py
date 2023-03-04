import time
import numpy as np
import cv2
import licenseocr
import pytesseract
import math

# set the path to the Tesseract executable file
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

key = cv2.waitKey(1)

CONFIDENCE_THRESHOLD = 0.8
NMS_THRESHOLD = 0.6

VEHICLES = ['car', 'truck', 'motorbike', 'bus']

## initializes model and network

class_file = open('yrhacks-car-camera\data\coco.names')
class_names = class_file.read().splitlines()

COLORS = np.random.uniform(0, 255, size=(len(class_names), 3))

model_config = 'yrhacks-car-camera\cfg\yolov4.cfg'
model_weights = 'yrhacks-car-camera\cfg\yolov4.weights'

net = cv2.dnn.readNetFromDarknet(model_config, model_weights)

## tells net to PREFER USING MY GPU but cv2 is not compiled to work with gpu... and im too lazy to do that...
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

## main loop

cv2.namedWindow('Camera Footage', cv2.WINDOW_NORMAL)

# resize window to fit screen
cv2.resizeWindow("Camera Footage", 1280, 720)

car_tracker = {}
car_speeds = {}

while(True):
    cam = cv2.VideoCapture('yrhacks-car-camera\example-footage.mp4')

    while(cam != None):
        ret, frame = cam.read()
        cam_width, cam_height, _ = frame.shape
        if not ret:
            break
        
        start = time.time()
        
        classes, scores, boxes = model.detect(frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

        end = time.time()
        
        fps = 1 / (end-start)
        
        def get_points(frame):
            # detect the car in the frame
            car_cascade = cv2.CascadeClassifier('cars.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cars = car_cascade.detectMultiScale(gray, 1.1, 1)

            # get the center point of the car
            if len(cars) > 0:
                (x, y, w, h) = cars[0]
                return (x + w//2, y + h//2)
            else:
                return None

        def calculate_speed(old_pos, new_pos, fps):
            # Calculate the distance between the two positions
            distance = math.sqrt((new_pos[0] - old_pos[0])**2 + (new_pos[1] - old_pos[1])**2)
            
            # Calculate the speed by dividing the distance by the time between frames
            speed = distance * fps
            
            return speed
        


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
                
                # check if a car is already being tracked
                if class_id in car_tracker:
                    # calculate speed
                    new_x, new_y, new_w, new_h = box
                    old_x, old_y, old_w, old_h, _ = car_tracker[class_id]
                    speed = calculate_speed((old_x, old_y), (new_x, new_y), 2)
                    car_speeds[class_id] = speed
                    # update tracker
                    car_tracker[class_id] = (new_x, new_y, new_w, new_h, time.time())
                else:
                    # add car to tracker
                    car_tracker[class_id] = (box[0], box[1], box[2], box[3], time.time())
                    car_speeds[class_id] = 0

                    
        # display speed of each tracked car
        for car_id, speed in car_speeds.items():
            if speed is not None:
                label = "Speed: {} km/h".format(round(speed, 2))
                x, y, _, _ , _ = car_tracker[car_id]
                cv2.putText(frame, label, (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        cv2.putText(frame, "FPS: %.2f" % fps, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imshow('Camera Footage', frame)
        
       
        key = cv2.waitKey(1)
        if key == ord('q'):
            should_exit = True

    cam.release()
    cv2.destroyAllWindows()