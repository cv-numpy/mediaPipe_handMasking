# Import the MediaPipe solutions module
from mediapipe import solutions

# Initialize the MediaPipe Hands solution
mp_hands = solutions.hands
# Create a Hands object with specific parameters
mp_solution = mp_hands.Hands(
        model_complexity = 0,  # Use the simplest model for faster processing
        min_detection_confidence = 0.5,  # Minimum confidence for a hand to be detected
        min_tracking_confidence = 0.5  # Minimum confidence for the hand landmarks to be tracked
        )

# Define the total number of keypoints that the model will output
points_num = 21

# Define the main function that will process the frame
def it(frame): 
    # Initialize variables
    number_detected = 0
    left_right = None
    output_xy = None

    # Process the frame with the MediaPipe Hands model
    mp_output = mp_solution.process(frame)
    # Get the handness (left or right) of the detected hands
    output_left_right = mp_output.multi_handedness

    # Count the number of hands detected
    number_detected = len(output_left_right) if output_left_right is not None else 0

    # If no hands are detected, return the initial values
    if number_detected == 0:
        return number_detected, left_right, output_xy
    
    else:
        # If hands are detected, process each hand
        left_right = []
        for i in range(number_detected):
            # Append the handness of each hand to the list
            left_right.append(output_left_right[i].classification[0].label)

        # Initialize the list of keypoints
        output_xy = []
        for _ in range(number_detected):
            # Get the landmarks of the current hand
            xy_mpDict = mp_output.multi_hand_landmarks[_]

            # Initialize the list of keypoints for the current hand
            measure_detect = []
            for i in range(points_num):
                # Append each keypoint to the list
                measure_detect.append(
                    [xy_mpDict.landmark[i].x, 
                     xy_mpDict.landmark[i].y]
                     )
                
            # Append the keypoints of the current hand to the main list
            output_xy.append(measure_detect)

        # Return the number of hands, the handness of each hand, and the keypoints of each hand
        return number_detected, left_right, output_xy