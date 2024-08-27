import numpy as np

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
    
def get_true_range(ocr_results, boundingBox, chat_gpt_info):
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

def get_second_y_range(ocr_results, boundingBox, second_range):

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
    
def axis_recalculation_module(prompt, ocr_results, boundingBox):
    x_axis = prompt["x-axis"]["range"]
    y_axis = prompt["y-axis"]["range"]
    x_axis_title = prompt["x-axis"]["title"]
    y_axis_title = prompt["y-axis"]["title"]
    second_y_axis_title = prompt["second-y-axis"]["title"]

    x_range, y_range = get_true_range(ocr_results, boundingBox, {'x': x_axis, 'y': y_axis})

    second_y_range = None
    if "second-y-axis" in prompt and prompt["second-y-axis"]["title"] is not None:
        print("SECOND RANGE")
        second_y_range = get_second_y_range(ocr_results, boundingBox, prompt["second-y-axis"]["range"])

    return x_axis_title, y_axis_title, second_y_axis_title, x_range, y_range, second_y_range