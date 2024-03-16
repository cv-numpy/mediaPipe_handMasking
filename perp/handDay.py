import numpy as np

def create_box():
    return {'head': None, 'middle': None, 'tail': None}

def mean(points):
    return [sum([p[0] for p in points])/len(points), sum([p[1] for p in points])/len(points)]

def distance(p1, p2):
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5


length_scale_factor = [0.35, 0.55, 0.65] # proximal mediate distal
class Finger:
    def __init__(self, fingerKeyPointMap):
        self.proximal = create_box()
        self.mediate = create_box()
        self.distal = create_box()

        # assert the length of fingerKeyPointMap is 4
        assert len(fingerKeyPointMap) == 4
        self.fingerKeyPointMap = fingerKeyPointMap


    def update_proximal(self, point):
        self.proximal['tail'] = point[0]
        self.proximal['head'] = point[1]
        self.proximal['middle'] = mean([point[0], point[1]])

    def update_mediate(self, point):
        self.mediate['tail'] = point[0]
        self.mediate['head'] = point[1]
        self.mediate['middle'] = mean([point[0], point[1]])

    def update_distal(self, point):
        self.distal['tail'] = point[0]
        self.distal['head'] = point[1]
        self.distal['middle'] = mean([point[0], point[1]])

    def update(self, handKeyPoints):
        points = [handKeyPoints[i] for i in self.fingerKeyPointMap]

        self.update_proximal(  points[0:2])
        self.update_mediate(   points[1:3])
        self.update_distal(    points[2:4])

    def update_length(self):
        self.proximal['length'] = distance(self.proximal['head'], self.proximal['tail'])
        self.mediate['length'] = distance(self.mediate['head'], self.mediate['tail'])
        self.distal['length'] = distance(self.distal['head'], self.distal['tail'])

    def update_angle(self):
        self.proximal['angle'] = np.arctan2(self.proximal['tail'][1] - self.proximal['head'][1], self.proximal['tail'][0] - self.proximal['head'][0])
        self.mediate['angle'] = np.arctan2(self.mediate['tail'][1] - self.mediate['head'][1], self.mediate['tail'][0] - self.mediate['head'][0])
        self.distal['angle'] = np.arctan2(self.distal['tail'][1] - self.distal['head'][1], self.distal['tail'][0] - self.distal['head'][0])

        self.perpAngle_right = [self.proximal['angle'] + np.pi / 2, self.mediate['angle'] + np.pi / 2, self.distal['angle'] + np.pi / 2]
        self.perpAngle_left = [self.proximal['angle'] - np.pi / 2, self.mediate['angle'] - np.pi / 2,  self.distal['angle'] - np.pi / 2]

    def get_perpendicular_linesX(self):
        for i, attr_name in enumerate(['proximal', 'mediate', 'distal']):
            phalange = getattr(self, attr_name)

            # Initialize the list of points
            line_points_right = []
            # Calculate the exact point coordinates at each step
            for step in range(int(phalange['length']*length_scale_factor[i])):
                exact_point = (phalange['middle'][0] + step * np.cos(self.perpAngle_right[i]), phalange['middle'][1] + step * np.sin(self.perpAngle_right[i]))
                # Round the coordinates and add the point to the list
                line_points_right.append((int(round(exact_point[0])), int(round(exact_point[1]))))

            # Initialize the list of points
            line_points_left = []
            # Calculate the exact point coordinates at each step
            for step in range(int(phalange['length']*length_scale_factor[i])):
                exact_point = (phalange['middle'][0] + step * np.cos(self.perpAngle_left[i]), phalange['middle'][1] + step * np.sin(self.perpAngle_left[i]))
                # Round the coordinates and add the point to the list
                line_points_left.append((int(round(exact_point[0])), int(round(exact_point[1]))))

            # Combine the points from left to right
            lineIndex = line_points_left[::-1] + line_points_right
            phalange['lineIndex'] = lineIndex