import cv2
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from helper_code.graph_reconstruction import convertPoint, get_filtered_answer, get_middle_coordinate
from helper_code.legend_extraction import get_legend_extraction
from helper_code.mask_detection import get_bounding_box, get_main_mask
from helper_code.color_detection import alter_image, get_color_masks
import json

def is_valid_color(color):
    try:
        mcolors.to_rgba(color)
        return True
    except ValueError:
        return False
    
def get_points_new(color_masks, width, height, boundingBox, color, wBox, hBox, x_axis, y_axis, legend):

    graph = []
    mask = color_masks[color]

    for x0 in range(boundingBox["topLeft"][0], boundingBox["bottomRight"][0], wBox):
        for y0 in range(boundingBox["topLeft"][1], boundingBox["bottomRight"][1], hBox):

            if legend and legend["top_x"] - legend["width"] / 2 <= x0 <= legend["top_x"] + legend["width"] / 2 and legend["top_y"] - legend["height"] / 2 <= y0 <= legend["top_y"] + legend["height"] / 2:
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
    main_mask, _ = get_main_mask(masks, scores)

    n_image = image.copy()
    _, boundingBox = get_bounding_box(main_mask, n_image)
    
    return image_name, boundingBox

def do_complete_analysis(wBox, hBox, metadata, image_name, boundingBox, legend = None):

    x_axis = metadata["x-axis"]["range"]
    y_axis = metadata["y-axis"]["range"]
    x_axis_title = metadata["x-axis"]["title"]
    y_axis_title = metadata["y-axis"]["title"]

    axis_labels = []
    rgb_colors = []
    coordinates = []
    for label, color in metadata["types"]:
        axis_labels.append(label)
        rgb_colors.append(color)
    
    image = alter_image(image_name, "Contrast")
    color_masks, width, height, memo = get_color_masks(image, rgb_colors)

    print("This image has the following colors", memo)

    for color in rgb_colors:
        coordinates.append(get_points_new(color_masks, width, height, boundingBox, color, wBox, hBox, x_axis, y_axis, legend))

    return coordinates, x_axis_title, y_axis_title, axis_labels, rgb_colors, x_axis, y_axis, memo

def get_reconstructed_plot(image_num, sam_predictor, yolo_model):
    prompt = {}
    with open('../plot_json/'+str(image_num)+'.json', 'r') as file:
        prompt = json.load(file)
    
    image_name, boundingBox = do_analysis(image_num, sam_predictor)
    legend = get_legend_extraction(image_name, yolo_model)
    coordinates, x_axis_title, y_axis_title, axis_labels, rgb_colors, x_range, y_range, memo = do_complete_analysis(1, 1, prompt, image_name, boundingBox, legend)

    return coordinates, x_axis_title, y_axis_title, axis_labels, rgb_colors, x_range, y_range, memo