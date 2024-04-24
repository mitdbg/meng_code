import cv2
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from helper_code.graph_reconstruction import convertPoint, get_filtered_answer, get_middle_coordinate
from helper_code.legend_extraction import extracted_mask
from helper_code.mask_detection import clean_mask_edges_and_convert_back, get_bounding_box, get_main_mask, show_mask
from helper_code.color_detection import alter_image, get_color_masks
from helper_code.axes_extraction import get_true_range, get_second_y_range
from helper_code.metadata_extraction import ask_gpt
from helper_code.bounding_box import distance, get_intersection_points
import json
from PIL import Image

def is_valid_color(color):
    try:
        mcolors.to_rgba(color)
        return True
    except ValueError:
        return False
    
def get_points_new(color_masks, width, height, boundingBox, color, wBox, hBox, x_axis, y_axis, extra_info):

    graph = []
    mask = color_masks[color]

    for x0 in range(boundingBox["topLeft"][0], boundingBox["bottomRight"][0], wBox):
        for y0 in range(boundingBox["topLeft"][1], boundingBox["bottomRight"][1], hBox):

            if extra_info is not None and extra_info[y0, x0] == 1:
                continue
            else:
                x, y = get_middle_coordinate(x0, y0, wBox, hBox)
                value = get_filtered_answer(mask, x0, y0, wBox, hBox, boundingBox["bottomRight"][0], boundingBox["bottomRight"][1])

                if value: 
                    graph.append({
                    "topLeft": (x0, y0), 
                    "middle": convertPoint((x,y), boundingBox, width, height, x_axis, y_axis), 
                    })
    
    return graph

def do_analysis(image_number, predictor):
    image_name = '../plot_images/'+str(image_number)+'.png'
    image = cv2.imread(image_name)

    height, width = image.shape[:2]
    input_point = np.array([[width // 2 - 50, height // 2 - 50], [width // 2 - 50, height // 2 + 50], [width // 2 + 50, height // 2 - 50], [width // 2 + 50, height // 2 + 50]])
    input_label = np.array([1, 1, 1, 1])

    predictor.set_image(image)

    masks, scores, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
    
    predictor.reset_image()
    
    main_mask, score = get_main_mask(masks, scores, image)
    main_mask_cleaned = clean_mask_edges_and_convert_back(main_mask)
    n_image = image.copy()
    print(n_image)
    _, boundingBox = get_bounding_box(main_mask_cleaned, n_image)
    
    return image_name, boundingBox

def get_fast_bounding_box(image_num, predictor):
    image_name, intersection_points = get_intersection_points(image_num)
    # Assuming we have four intersection points
    if len(intersection_points) == 4:
        # Sort points for consistency: [top-left, top-right, bottom-left, bottom-right]
        intersection_points.sort(key=lambda x: (x[1], x[0]))

        width = distance(intersection_points[0], intersection_points[1])
        height = distance(intersection_points[0], intersection_points[2])
        area = width * height

        image = cv2.imread(image_name)
        if area > .5 * image.shape[0] * image.shape[1]:
            topLeft = min(intersection_points, key=lambda p: (p[0], p[1]))
            bottomRight = max(intersection_points, key=lambda p: (p[0], p[1]))

            bounding_box = {'topLeft': topLeft, 'bottomRight': bottomRight}
            return image_name, bounding_box
    
    image_name, bounding_box = do_analysis(image_num, predictor)
    return image_name, bounding_box

def do_complete_analysis(wBox, hBox, metadata, image_name, boundingBox, extra_info = None, image_alter = True, margin = 10):

    x_axis = metadata["x-axis"]["range"]
    y_axis = metadata["y-axis"]["range"]
    x_axis_title = metadata["x-axis"]["title"]
    y_axis_title = metadata["y-axis"]["title"]
    second_y_axis_title = metadata["second-y-axis"]["title"]

    true_x_range, true_y_range = get_true_range(image_name, boundingBox, {'x': x_axis, 'y': y_axis})
    
    print("X")
    print("Predicted", true_x_range)
    print("Chat GPT", x_axis)

    print("Y")
    print("Predicted", true_y_range)
    print("Chat GPT", y_axis)

    axis_labels = []
    rgb_colors = []
    coordinates = []
    for label, color in metadata["types"]:
        axis_labels.append(label)
        rgb_colors.append(color)
    
    image = None
    if image_alter:
        image = alter_image(image_name, "Contrast")
    else:
        image = alter_image(image_name, "")
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    
    color_masks, width, height, memo = get_color_masks(image, rgb_colors, boundingBox, margin)

    for color in rgb_colors:
        coordinates.append(get_points_new(color_masks, width, height, boundingBox, color, wBox, hBox, true_x_range, true_y_range, extra_info))

    return coordinates, x_axis_title, y_axis_title, second_y_axis_title, axis_labels, rgb_colors, true_x_range, true_y_range, memo

def get_reconstructed_plot(image_num, sam_predictor, yolo_model, image_alter, margin):
    prompt = {}
    
    image_name, boundingBox = get_fast_bounding_box(image_num, sam_predictor)
    
    try: 
        response = ask_gpt(image_name)
        answer = response["choices"][0]["message"]["content"]
        print(answer)
        cleaned = answer.replace('```json\n', '').replace('\n```', '')
        prompt = json.loads(cleaned)
    except:
        prompt = {
            "x-axis": {
                "title": "X Axis Title",
                "range": [0, 100],
            },
            "y-axis": {
                "title": "Y Axis Title",
                "range": [0, 100],
            },
            "second-y-axis": {
                "title": None,
                "range": None,
            },
            'types': [
                ['Type 1', 'red'],
                ['Type 2', 'blue'],
                ['Type 3', 'green']
            ]
        }
    
    print(prompt)
    with open("../results/" + str(image_num) + "/metadata.json", 'w') as json_file:
        json.dump(prompt, json_file, indent=4)
    
    second_y_range = None
    if "second-y-axis" in prompt and prompt["second-y-axis"]["title"] is not None:
        print("SECOND RANGE")
        second_y_range = get_second_y_range(image_name, boundingBox, prompt["second-y-axis"]["range"])

        
    print(boundingBox)
    with open("../results/" + str(image_num) + "/bounding_box.json", 'w') as json_file:
        json.dump(boundingBox, json_file, indent=4)

    image = cv2.imread(image_name)
    cv2.rectangle(image, boundingBox['topLeft'], boundingBox['bottomRight'], (0, 0, 255), 2)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(f'Bounding Box in Image {image_num}')
    
    plt.savefig("../results/"+str(image_num)+"/bounding_box.png")
    plt.show()
    
    extra_info = extracted_mask(image_num, image_name, yolo_model)
    coordinates, x_axis_title, y_axis_title, second_y_axis_title, axis_labels, rgb_colors, x_range, y_range, memo = do_complete_analysis(1, 1, prompt, image_name, boundingBox, extra_info, image_alter, margin)

    return coordinates, x_axis_title, y_axis_title, second_y_axis_title, axis_labels, rgb_colors, x_range, y_range, memo, second_y_range