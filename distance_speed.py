class DistanceCalculator:
    def __init__(self, camera_matrix, square_size=1.0):
        self.focal_length = camera_matrix[0, 0]  # Assuming fx = fy
        self.square_size = square_size

    def calculate_distance(self, bounding_box):
        x1, y1, x2, y2 = bounding_box
        w = x2 - x1
        h = y2 - y1
        diagonal = (w**2 + h**2)**0.5

        # Distance calculation
        distance = (self.square_size * self.focal_length) / diagonal
        return distance