import numpy as np
import cv2

import ai_detect
import handDay

indexFinger =   handDay.Finger([ 5,  6,  7,  8])
middleFinger =  handDay.Finger([ 9, 10, 11, 12])
ringFinger =    handDay.Finger([13, 14, 15, 16])
littleFinger =  handDay.Finger([17, 18, 19, 20])
fingers = [indexFinger, middleFinger, ringFinger, littleFinger]

def distance(p1, p2):
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5


# testing

image = cv2.imread('hand_image.png')
image = image[:, 80:560]

image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# edges = cv2.Canny(image_gray, 100, 200)

number, left_right, keyPoints_list_normalized = ai_detect.it(image)

# get pixel level sized keypoints
keyPoints_list_float = [ [point[0] * 480, point[1] * 480] for point in keyPoints_list_normalized[0] ]
keyPoints_list_int = [ [int(point[0]), int(point[1])] for point in keyPoints_list_float ]


for finger in fingers:
    finger.update(keyPoints_list_int)
    finger.update_length()
    finger.update_angle()
    finger.get_perpendicular_linesX()

# Get the perpendicular lines
lineIndexes = [
    indexFinger.distal['lineIndex'], middleFinger.distal['lineIndex'], ringFinger.distal['lineIndex'], littleFinger.distal['lineIndex'],

    indexFinger.mediate['lineIndex'], middleFinger.mediate['lineIndex'], ringFinger.mediate['lineIndex'], littleFinger.mediate['lineIndex'],

    indexFinger.proximal['lineIndex'], middleFinger.proximal['lineIndex'], ringFinger.proximal['lineIndex'], littleFinger.proximal['lineIndex'],
]

# draw circle at the line indexes
for lineIndex in lineIndexes:
    for point in lineIndex:
        cv2.circle(image, point, 1, (0, 0, 255), -1)


# Convolution
        
def calculate_width(line_pixels, lineIndexes):
    # Apply 1D convolution to calculate the gradient
    gradient = np.convolve(line_pixels, [1, -1], mode='valid')

    # Find the indices of the maximum gradient in both directions
    max_gradient_index1 = np.argmax(np.abs(gradient[:len(gradient)//2]))
    max_gradient_index2 = np.argmax(np.abs(gradient[len(gradient)//2:])) + len(gradient)//2
    
    return max_gradient_index1, max_gradient_index2

# Extract the pixels along the line
line_pixels = [[image_gray[point[1], point[0]] for point in lineIndex] for lineIndex in lineIndexes]


# Calculate and print the width for each line
for i, line in enumerate(line_pixels):
    max_gradient_index1, max_gradient_index2 = calculate_width(line, lineIndexes[i])
    
    # Calculate the width as the distance between the two points with maximum gradients
    width = distance(lineIndexes[i][max_gradient_index1], lineIndexes[i][max_gradient_index2])
    print(width)

    # # draw the edge points
    cv2.circle(image, lineIndexes[i][max_gradient_index1], 2, (0, 255, 0), -1)
    cv2.circle(image, lineIndexes[i][max_gradient_index2], 2, (0, 255, 0), -1)


cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()