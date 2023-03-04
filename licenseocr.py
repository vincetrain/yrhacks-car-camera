import pytesseract
import cv2
import numpy as np
import imutils

# def get_plate(frame):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
#     _, contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
#     if contours:
#         areas = [cv2.contourArea(c) for c in contours]
        
#         max_index = np.argmax(areas)
        
#         max_cnt = contours[max_index]
#         max_cnt_area = areas[max_index]
        
#         x, y, w, h = cv2.boundingRect(max_cnt)
        
#         if not self.ratioCheck(max_cnt_area, frame.shape[1]. frame.shape[0]):
#             return frame, False, None
#         return frame, True, [x, y, w, h]
#     else:
#         return frame, False, None

def get_plate(frame):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.bilateralFilter(frame_gray, 11, 17, 17)
    frame_edged = cv2.Canny(frame_gray, 30, 200)
    cnts,new = cv2.findContours(frame_edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    frame_copy=frame.copy()
    cv2.drawContours(frame_copy,cnts,-1,(0,255,0),3)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    print(pytesseract.image_to_string(frame_rgb))
        
if __name__ == '__main__':
    print('this does nothing when you run it. please don\'t run it.')
    exit(0)

__all__ = []