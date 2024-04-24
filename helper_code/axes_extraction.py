import numpy as np
import boto3
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def is_number(s):
    if s == '0':
        return True
    try:
        float(s)
        return True
    
    except ValueError:
        return False

def find_nearest_y_candidate(ocr_results, point):
    nearest = None
    min_distance = float('inf')

    for text, coords in ocr_results:
        x, y = get_mean_coordinate(coords)
        if x < point[0] and is_number(text):
            distance = abs(y - point[1])
            if distance < min_distance:
                min_distance = distance
                if text == 'o':
                    text = '0'
                nearest = {'text': text, 'coords': coords}

    return nearest

def find_nearest_x_candidate(ocr_results, point):
    nearest = None
    min_distance = float('inf')

    for text, coords in ocr_results:
        x, y = get_mean_coordinate(coords)
        if y >= point[1] and is_number(text):
            distance = abs(x - point[0])
            if distance < min_distance:
                min_distance = distance
                if text == 'o':
                    text = '0'
                nearest = {'text': text, 'coords': coords}

    return nearest

def get_mean_coordinate(coords):
    x, y = 0, 0
    for coord in coords:
        x += coord[0]
        y += coord[1]
    x /= len(coords)
    y /= len(coords)
    return (x, y)

def calculate_y_axis(ocr_results, bounding_box, chat_gpt_info):
    top_left = [bounding_box['topLeft'][0], bounding_box['topLeft'][1]]
    bottom_left = [bounding_box['topLeft'][0], bounding_box['bottomRight'][1]]

    rmax = find_nearest_y_candidate(ocr_results, top_left)
    rmin = find_nearest_y_candidate(ocr_results, bottom_left)

    if rmin and rmax:
        print("Y MIN", rmin["text"])
        print("Y MAX", rmax["text"])

        try:
            y_sb = float(rmin["text"])
        except ValueError:
            print("error with value on y min")
            y_sb = float(chat_gpt_info[0])

        try:
            y_sa = float(rmax["text"])
        except ValueError:
            print("error with value on y max")
            y_sa = float(chat_gpt_info[1])
        print("SEMANTIC Y MIN", y_sb)
        print("SEMANTIC Y MAX", y_sa)

        y_b_min = get_mean_coordinate(rmin['coords'])[1]
        y_a_max = get_mean_coordinate(rmax['coords'])[1]

        print("Y")
        print("OCR DIFF", y_a_max - y_b_min)
        print("SEM DIFF", y_sa - y_sb)
        
        ratio = (y_b_min - y_a_max)/(y_sb - y_sa)
        print("OCR RATIO", ratio)

        bounding_box_pixel_difference = bounding_box['topLeft'][1] - bounding_box['bottomRight'][1]
        real_difference = bounding_box_pixel_difference/ratio

        pixel_diff_from_base_point = (y_b_min - bounding_box['bottomRight'][1])
        print("PIXEL DIFF", pixel_diff_from_base_point)

        real_y_diff_from_base_point = pixel_diff_from_base_point/ratio 
        print("REAL DIFF FROM BASE POINT", real_y_diff_from_base_point)
        
        y_min = y_sb - real_y_diff_from_base_point
        y_max = y_min + real_difference

        print("Y_MIN", y_min)
        print("Y_MAX", y_max)

        return y_min, y_max, ratio
    else:
        return None, None, None

def calculate_x_axis(ocr_results, bounding_box, chat_gpt_info):
    bottom_left = [bounding_box['topLeft'][0], bounding_box['bottomRight'][1]]
    bottom_right = [bounding_box['bottomRight'][0], bounding_box['bottomRight'][1]]

    rmin = find_nearest_x_candidate(ocr_results, bottom_left)
    rmax = find_nearest_x_candidate(ocr_results, bottom_right)

    if rmin and rmax:
        try:
            x_sb = float(rmin["text"])
        except ValueError:
            print("error with value on x min")
            x_sb = float(chat_gpt_info[0])

        try:
            x_sa = float(rmax["text"])
        except ValueError:
            print("error with value on x max")
            x_sa = float(chat_gpt_info[1])

        print("SEMANTIC X MIN", x_sb)
        print("SEMANTIC X MAX", x_sa)

        x_b_min = get_mean_coordinate(rmin['coords'])[0]
        x_a_max = get_mean_coordinate(rmax['coords'])[0]

        print("X")
        print("OCR DIFF", x_a_max - x_b_min)
        print("SEM DIFF", x_sa - x_sb)

        ratio = (x_sa - x_sb)/(x_a_max - x_b_min)
        print("OCR RATIO", ratio)

        chart_width = bottom_right[0] - bottom_left[0]
        real_difference = chart_width * ratio

        pixel_diff_from_base_point = (x_b_min - bounding_box['topLeft'][0])
        print("PIXEL DIFF", pixel_diff_from_base_point)
        
        real_x_diff_from_base_point = pixel_diff_from_base_point * ratio 
        print("REAL DIFF FROM BASE POINT", real_x_diff_from_base_point)
        
        x_min = x_sb - real_x_diff_from_base_point
        x_max = x_min + real_difference

        print("X_MIN", x_min)
        print("X_MAX", x_max)
        
        return x_min, x_max, ratio
    else:
        return None, None, None
    
def get_true_range(image_name, boundingBox, chat_gpt_info):
    # Initialize the Textract client
    textract = boto3.client('textract')

    # Load the image file
    with open(image_name, 'rb') as document:
        img_test = bytearray(document.read())

    # Call Amazon Textract
    response = textract.detect_document_text(Document={'Bytes': img_test})

    # Process Textract response to match desired output format
    ocr_results = []
    for item in response['Blocks']:
        if item['BlockType'] == 'LINE' and 'Confidence' in item and item['Confidence'] > 50:  # Adjust confidence as needed
            text = item['Text']
            # Extract bounding box coordinates scaled to image dimensions
            width, height = Image.open(image_name).size
            box = item['Geometry']['BoundingBox']
            x = box['Left'] * width
            y = box['Top'] * height
            w = box['Width'] * width
            h = box['Height'] * height
            box = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
            ocr_results.append((text, box))

    # Visualization (assuming you want to visualize the results similarly)
    img = cv2.imread(image_name)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img)

    for text, box in ocr_results:
        # Extract the bounding box coordinates
        start_point = box[0]
        end_point = box[2]
        box_width = end_point[0] - start_point[0]
        box_height = end_point[1] - start_point[1]
        
        # Create a Rectangle patch
        rect = patches.Rectangle(start_point, box_width, box_height, linewidth=1, edgecolor='r', facecolor='none')
        
        # Add the rectangle to the Axes
        ax.add_patch(rect)
        
        # Annotate the image with the OCR'ed text
        ax.text(start_point[0], start_point[1] - 10, text, bbox=dict(fill=False, edgecolor='red', linewidth=2), color='red')

    plt.axis('off')  # Turn off axis numbers and ticks
    plt.show()

    try:
        ymin, ymax, y_scale = calculate_y_axis(ocr_results, boundingBox, chat_gpt_info['y'])
    except:
        ymin, ymax = None, None

    try:
        xmin, xmax, x_scale = calculate_x_axis(ocr_results, boundingBox, chat_gpt_info['x'])
    except:
        xmin, xmax = None, None

    x_range = []
    for x in [xmin, xmax]:
        if x is None or np.isnan(x) or not np.isfinite(x):
            x_range = chat_gpt_info['x']
            print("x is in fault condition")
            break
        
        x_range.append(x)
    
    y_range = []
    for y in [ymin, ymax]:
        if y is None or np.isnan(y) or not np.isfinite(y):
            y_range = chat_gpt_info['y']
            print("y is in fault condition")
            break

        y_range.append(y)

    if x_range is None or x_range[0] is None or type(x_range[0]) is str or x_range[1] is None or type(x_range[1]) is str or x_range[1] == x_range[0]:
        x_range = [0, 100]

    if y_range is None or y_range[0] is None or type(y_range[0]) is str or y_range[1] is None or type(y_range[1]) is str or y_range[1] == y_range[0]:
        y_range = [0, 100]

    return x_range, y_range

def find_nearest_second_y_candidate(ocr_results, point):
    nearest = None
    min_distance = float('inf')

    for text, coords in ocr_results:
        x, y = get_mean_coordinate(coords)
        if x > point[0] and is_number(text):
            distance = abs(y - point[1])
            if distance < min_distance:
                min_distance = distance
                if text == 'o':
                    text = '0'
                nearest = {'text': text, 'coords': coords}

    return nearest

def get_second_y_range(image_name, boundingBox, second_range):
    # Initialize the Textract client
    textract = boto3.client('textract')

    # Load the image file
    with open(image_name, 'rb') as document:
        img_test = bytearray(document.read())

    # Call Amazon Textract
    response = textract.detect_document_text(Document={'Bytes': img_test})

    # Process Textract response to match desired output format
    ocr_results = []
    for item in response['Blocks']:
        if item['BlockType'] == 'LINE' and 'Confidence' in item and item['Confidence'] > 50:  # Adjust confidence as needed
            text = item['Text']
            # Extract bounding box coordinates scaled to image dimensions
            width, height = Image.open(image_name).size
            box = item['Geometry']['BoundingBox']
            x = box['Left'] * width
            y = box['Top'] * height
            w = box['Width'] * width
            h = box['Height'] * height
            box = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
            ocr_results.append((text, box))

    top_right = [boundingBox['bottomRight'][0], boundingBox['topLeft'][1]]
    bottom_right = [boundingBox['bottomRight'][0], boundingBox['bottomRight'][1]]

    rmax = find_nearest_second_y_candidate(ocr_results, top_right)
    rmin = find_nearest_second_y_candidate(ocr_results, bottom_right)

    print(rmin)
    print(rmax)

    if rmin and rmax:
        print("SECOND Y MIN", rmin["text"])
        print("SECOND Y MAX", rmax["text"])

        try:
            y_sb = float(rmin["text"])
        except ValueError:
            print("error with value on second y min")
            y_sb = float(second_range[0])

        try:
            y_sa = float(rmax["text"])
        except ValueError:
            print("error with value on second y max")
            y_sa = float(second_range[1])
        print("SEMANTIC SECOND Y MIN", y_sb)
        print("SEMANTIC SECOND Y MAX", y_sa)

        y_b_min = get_mean_coordinate(rmin['coords'])[1]
        y_a_max = get_mean_coordinate(rmax['coords'])[1]

        print("SECOND Y")
        print("OCR DIFF", y_a_max - y_b_min)
        print("SEM DIFF", y_sa - y_sb)
        
        ratio = (y_b_min - y_a_max)/(y_sb - y_sa)
        print("OCR RATIO", ratio)

        bounding_box_pixel_difference = boundingBox['topLeft'][1] - boundingBox['bottomRight'][1]
        real_difference = bounding_box_pixel_difference/ratio

        pixel_diff_from_base_point = (y_b_min - boundingBox['bottomRight'][1])
        print("PIXEL DIFF", pixel_diff_from_base_point)

        real_y_diff_from_base_point = pixel_diff_from_base_point/ratio 
        print("REAL DIFF FROM BASE POINT", real_y_diff_from_base_point)
        
        y_min = y_sb - real_y_diff_from_base_point
        y_max = y_min + real_difference

        print("SECOND Y_MIN", y_min)
        print("SECOND Y_MAX", y_max)

        y_range = []
        for y in [y_min, y_max]:
            if y is None or np.isnan(y) or not np.isfinite(y):
                y_range = y_sb
                print("second y is in fault condition")
                break

            y_range.append(y)


        return y_range
    else: 
        print("ERROR SECOND RANGE")
        return second_range