import numpy as np
from cv2 import line

import queue
class Mask:
    def __init__(self):
        self.palmX = [0, 1, 5, 9, 13, 17]

        self.phalangesX = [
            [ 1,  2], [ 2,  3], [ 3,  4],
            [ 5,  6], [ 6,  7], [ 7,  8],
            [ 9, 10], [10, 11], [11, 12],
            [13, 14], [14, 15], [15, 16],
            [17, 18], [18, 19], [19, 20],
        ]

        self.phalanges_queue = queue.Queue()

    def update_points(self, points):
        self.points = np.array(points, dtype=np.int32)

        self.points_distance()

        self.get_phalanges()

    def get_phalanges(self):
        for i in self.phalangesX:
            self.phalanges_queue.put([
                self.points[i[0]],
                self.points[i[1]],
            ])

        length = len(self.palmX)
        for index in range(length):
            index1 = self.palmX[index % length]
            index2 = self.palmX[(index + 1) % length]

            self.phalanges_queue.put([
                self.points[index1],
                self.points[index2],
            ])


    def points_distance(self):
        # Define the points
        middleF_points = [
            self.points[ 9], # 4*2+1
            self.points[10], # 4*2+2
            self.points[11], # 4*2+3
            self.points[12], # 4*2+4
        ]
        palmPoints = [self.points[0], self.points[5], self.points[17]]

        # Calculate the distances
        middleF_distances = [np.linalg.norm(middleF_points[i+1] - middleF_points[i]) for i in range(len(middleF_points) - 1)]
        wrist_distances = [np.linalg.norm(palmPoints[i+1] - palmPoints[i]) for i in range(len(palmPoints) - 1)]
        wrist_distances.append(np.linalg.norm(palmPoints[-1] - palmPoints[0]))

        # Find the maximum distance
        self.distance = max(middleF_distances + wrist_distances)

    def create_mask(self, image, points):
        # Update the points
        self.update_points(points)

        # pre calculate the distance
        distance = int(self.distance / 4)

        # Create the mask
        mask = np.zeros(image.shape[:2], dtype=np.uint8)

        # Draw the phalanges
        while not self.phalanges_queue.empty():
            phalange = self.phalanges_queue.get()
            x1, y1 = phalange[0]
            x2, y2 = phalange[1]
            line(mask, (x1, y1), (x2, y2), 255, distance, -1)

        return mask