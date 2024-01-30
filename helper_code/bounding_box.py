import numpy as np
import cv2

def distance(point1, point2):
    """Calculate the Euclidean distance between two points."""
    return np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

def extend_line(line, img_width, img_height, is_vertical=True):
    """Extend the line to span the entire width or height of the image."""
    x1, y1, x2, y2 = line
    if is_vertical:
        return (x1, 0, x2, img_height)
    else:
        return (0, y1, img_width, y2)

def find_intersection(line1, line2):
    """Find intersection point of two lines."""
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denominator == 0:
        return None  # Lines are parallel

    intersect_x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denominator
    intersect_y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denominator
    return (int(intersect_x), int(intersect_y))

def calculate_angle(x1, y1, x2, y2):
    return np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi

def line_length(line):
    x1, y1, x2, y2 = line
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def combine_lines(lines, is_vertical=True, combine_threshold=30):
    combined_lines = []
    used = set()

    for i, line1 in enumerate(lines):
        if i in used:
            continue

        x11, y11, x12, y12 = line1
        for j, line2 in enumerate(lines):
            if j in used or i == j:
                continue

            x21, y21, x22, y22 = line2
            # Check if lines are close enough
            if (is_vertical and abs(x11 - x21) < combine_threshold) or \
               (not is_vertical and abs(y11 - y21) < combine_threshold):
                # Combine the lines
                combined_line = (min(x11, x21), min(y11, y21), max(x12, x22), max(y12, y22))
                combined_lines.append(combined_line)
                used.update([i, j])
                break

        if i not in used:
            combined_lines.append(line1)

    return combined_lines

def get_intersection_points(image_num):
    image_name = '../plot_images/' + str(image_num) + '.png'
    image = cv2.imread(image_name)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=150, maxLineGap=10)

    angle_threshold = 5
    vertical_lines = []
    horizontal_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = calculate_angle(x1, y1, x2, y2)

        if abs(angle) < angle_threshold or abs(angle - 180) < angle_threshold:
            horizontal_lines.append((x1, y1, x2, y2))

        if (90 - angle_threshold) <= abs(angle) <= (90 + angle_threshold):
            vertical_lines.append((x1, y1, x2, y2))

    # Combine nearby line segments
    combined_horizontal_lines = combine_lines(horizontal_lines, is_vertical=False)
    combined_vertical_lines = combine_lines(vertical_lines, is_vertical=True)

    # Sort by length and select the two longest lines
    combined_horizontal_lines.sort(key=line_length, reverse=True)
    combined_vertical_lines.sort(key=line_length, reverse=True)
    selected_horizontal = combined_horizontal_lines[:2]
    selected_vertical = combined_vertical_lines[:2]

    # Extend vertical and horizontal lines
    extended_vertical_lines = [extend_line(line, image.shape[1], image.shape[0], is_vertical=True) for line in selected_vertical]
    extended_horizontal_lines = [extend_line(line, image.shape[1], image.shape[0], is_vertical=False) for line in selected_horizontal]

    # Draw extended lines
    for line in extended_horizontal_lines + extended_vertical_lines:
        x1, y1, x2, y2 = line
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    intersection_points = []
    # Calculate and mark intersection points
    for v_line in extended_vertical_lines:
        for h_line in extended_horizontal_lines:
            intersection = find_intersection(v_line, h_line)
            if intersection:
                intersection_points.append(intersection)
                cv2.circle(image, intersection, 5, (255, 0, 0), -1)
    return image_name, intersection_points