import re
import pytesseract as pt
import cv2
import numpy as np

# Load the Haar Cascade classifier for license plates
plate_cascade = cv2.CascadeClassifier('cascades/haarcascade_russian_plate_number.xml')

def get_plate(frame):


    # Convert the image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect license plates in the image
    plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=2)

    plate_text = ''
    # Loop over each license plate and extract the text using OCR
    for (x, y, w, h) in plates:
        plate_img = frame[y:y+h, x:x+w]
        plate_gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        plate_gray = cv2.resize(plate_gray, None, fx=5, fy=5, interpolation=cv2.INTER_NEAREST)
        dilated_img = cv2.dilate(plate_gray, np.ones((7, 7), np.uint8))


        # Threshold using Otsu's
        
        cv2.imwrite('plate.png', dilated_img)
            
        plate_text = pt.image_to_data(dilated_img, config="-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ", output_type=pt.Output.DICT)

    
    
    return []
        
if __name__ == '__main__':
    print('this does nothing when you run it. please don\'t run it.')
    exit(0)

__all__ = [get_plate]