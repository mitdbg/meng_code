import cv2
import matplotlib.colors as mcolors
from helper_code.annotation_extraction import annotation_extraction_module
from helper_code.color_detection import color_extraction_module
from helper_code.axes_extraction import axis_recalculation_module
from helper_code.metadata_extraction import metadata_extraction_module
from helper_code.bounding_box import bounding_box_module
from helper_code.ocr_extraction import ocr_extraction_module
from helper_code.legend_extraction import legend_extraction_module

def is_valid_color(color):
    try:
        mcolors.to_rgba(color)
        return True
    except ValueError:
        return False

def get_reconstructed_plot(image_num, sam_predictor, yolo_model, image_alter, margin):

    image_name = '../plot_images/' + str(image_num) + '.png'
    image_bgr = cv2.imread(image_name)
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    prompt = {}
    
    ## BOUNDING BOX
    boundingBox = bounding_box_module(image_num, image, sam_predictor)

    # METADATA EXTRACTION
    prompt = metadata_extraction_module(image_name, image_num)

    print(prompt)

    # OCR EXTRACTION
    ocr_results, ocr_mask = ocr_extraction_module(image, image_name)

    # ANNOTATION EXTRACTION MODULE
    legend_box, ocr_mask = annotation_extraction_module(ocr_mask, image_num, image_name, image, yolo_model)

    # LEGEND EXTRACTION MODULE
    num_most_common = 5
    legend_items = legend_extraction_module(image, legend_box, ocr_results, num_most_common)
    print(legend_items)

    # AXIS RECALCULATION
    x_axis_title, y_axis_title, second_y_axis_title, x_range, y_range, second_y_range = axis_recalculation_module(prompt, ocr_results, boundingBox)

    # COLOR EXTRACTION MODULE
    ## USE LEGEND ITEMS IN THE COLOR EXTRACTION
    coordinates, axis_labels, rgb_colors, memo = color_extraction_module(image_name, prompt, boundingBox, x_range, y_range, ocr_mask)

    return coordinates, x_axis_title, y_axis_title, second_y_axis_title, axis_labels, rgb_colors, x_range, y_range, memo, second_y_range