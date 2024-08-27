from collections import Counter

def get_legend_item_pairs(legend_box, ocr_results, legend_items):

    x_min = int(legend_box['x'] - legend_box['width']/2)
    x_max = int(legend_box['x'] + legend_box['width']/2) + 1
    y_min = int(legend_box['y'] - legend_box['height']/2)
    y_max = int(legend_box['y'] + legend_box['height']/2) + 1
    
    for text, box in ocr_results:

        start_point = box[0]
        end_point = box[2]

        if x_min <= start_point[0] and x_max >= end_point[0] and y_min <= start_point[1] and y_max >= end_point[1] and text != "-":
        
            print("LEGEND ITEM", text)

            legend_items.append({
                "text": text,
                "text_location": box,
                "marker": None,
            })
    
    return x_min

def get_background_color(image):
    width, height, _ = image.shape
    memo = {}
    for y in range(0, width):
        for x in range(0, height):
            r,g,b = list(image[y, x])

            if (r,g,b) not in memo:
                memo[(r,g,b)] = 0

            memo[(r,g,b)] += 1

    maximum = max(memo.values())
    key = None
    for k, v in memo.items():
        if v == maximum:
            key = k
            break

    return key

def get_legend_markers(image, legend_items, x_min, key, num_most_common=5):
    for legend_item in legend_items:
        box = legend_item["text_location"]
        start_point = box[0]
        end_point = box[2]

        # List to store RGB values
        rgb_values = []

        # Loop through the specified area
        for x in range(x_min, start_point[0]):
            for y in range(start_point[1], end_point[1]):
                pixel = tuple(list(image[y, x]))

                if key != pixel:

                    rgb_values.append(pixel)

            # Create a Counter object
        counter = Counter(rgb_values)

        # Find the most common element
        most_common_elements = counter.most_common(num_most_common)

        most_common = []

        for element, count in most_common_elements:

            most_common.append(element)

        legend_item["marker"] = most_common

def legend_extraction_module(image, legend_box, ocr_results, num_most_common=5):
    legend_items = []

    x_min = get_legend_item_pairs(legend_box, ocr_results, legend_items)

    key = get_background_color(image)

    finished = get_legend_markers(image, legend_items, x_min, key, num_most_common)

    return legend_items