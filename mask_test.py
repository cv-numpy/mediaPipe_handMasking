import cv2 as cv
import numpy as np

import ai_detect
import drawIndex_numpy

# size of cam frame
width = 1280
height = 720 

r = int( (width - height) / 2 )
cap = cv.VideoCapture(0)

cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)

mask_R = drawIndex_numpy.Mask()
mask_L = drawIndex_numpy.Mask()

import time

init_time = time.time()

while True:
    ret, frame = cap.read()

    frame = frame[:, r:r+height]

    # Run MediaPipe 
    # The Camera is like a mirror, flipped left and right!
    number, left_right, keyPoints_list_normalized = ai_detect.it(frame)
    
    if number > 0:

        # Create an empty mask to start with
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)

        for index, element in enumerate(left_right):
            if element == 'Left':
                # get pixel level sized keypoints
                keyPoints_list_float = [ [point[0] * height, point[1] * height] for point in keyPoints_list_normalized[index] ]

                mask_r = mask_R.create_mask(frame, keyPoints_list_float)

                mask = cv.bitwise_or(mask, mask_r)

            if element == 'Right':
                # get pixel level sized keypoints
                keyPoints_list_float = [ [point[0] * height, point[1] * height] for point in keyPoints_list_normalized[index] ]

                mask_l = mask_L.create_mask(frame, keyPoints_list_float)

                mask = cv.bitwise_or(mask, mask_l)

        # Apply the combined mask to the frame
        frame = cv.bitwise_and(frame, frame, mask=mask)

    cv.imshow('Demo', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

# test result: 
# time to run mediapipe:  0.02s
# time to run mask:  0.002s