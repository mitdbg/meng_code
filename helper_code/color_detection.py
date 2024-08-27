import numpy as np
from PIL import Image, ImageEnhance
from helper_code.graph_reconstruction import get_points_new
from scipy.spatial import KDTree
import webcolors

def alter_image(image_name, filter_type):
    image = Image.open(image_name)
    image = image.convert('RGB')
    if filter_type == "Contrast":
        filter = ImageEnhance.Contrast(image)
        image = filter.enhance(2)
    elif filter_type == "Brightness":
        filter = ImageEnhance.Brightness(image)
        image = filter.enhance(2)
    elif filter_type == "Sharpness":
        filter = ImageEnhance.Sharpness(image)
        image = filter.enhance(2)
    return image

def get_all_color_names():
    color_names = []
    color_rgb_values = []

    color_names = list(webcolors.CSS3_NAMES_TO_HEX.keys())
    color_rgb_values = [webcolors.hex_to_rgb(hex_value) for hex_value in webcolors.CSS3_NAMES_TO_HEX.values()]
    return color_names, color_rgb_values

    for color_name in webcolors.names("css3"):
        color_names.append(color_name)
        color_rgb_values.append(webcolors.name_to_rgb(color_name))
    return color_names, color_rgb_values

COLOR_NAMES, COLOR_RGB_VALUES = get_all_color_names()
KDT_DB = KDTree(COLOR_RGB_VALUES)

def convert_rgb_to_names(rgb_tuple, memo):
    _, index = KDT_DB.query(rgb_tuple)
    complex_color = COLOR_NAMES[index]
    simple_color = complex_to_simple_color[COLOR_NAMES[index]]
    memo[complex_color] = memo.get(complex_color, 0) + 1
    return simple_color

def get_color_masks(image, rgb_colors, boundingBox, margin = 10):

    width, height = image.size
    masks = {}
    for c in rgb_colors:
        masks[c] = np.zeros((width, height))
    
    memo = {}

    x_left, y_top = boundingBox["topLeft"]
    x_right, y_bottom = boundingBox["bottomRight"]

    for y in range(0, height):
        for x in range(0, width):

            r, g, b = image.getpixel((x, y))

            new_color = convert_rgb_to_names([r,g,b], memo)

            if new_color in rgb_colors:
                if new_color in ["gray", "grey", "white", "black"]:
                    if (x >= x_left + margin) and (x <= x_right - margin) and (y >= y_top + margin) and (y <= y_bottom - margin):
                        masks[new_color][x, y] = 1
                elif max(abs(r - g), abs(g - b), abs(r - b)) > margin:
                    masks[new_color][x, y] = 1
                
                
    return masks, width, height, memo

def color_extraction_module(image_name, prompt, boundingBox, x_range, y_range, extra_info):
    image = alter_image(image_name, "Contrast")
    axis_labels = []
    rgb_colors = []
    coordinates = []
    for label, color in prompt["types"]:
        axis_labels.append(label)
        rgb_colors.append(color)

    color_masks, width, height, memo = get_color_masks(image, rgb_colors, boundingBox, margin=10)

    print(memo)

    for color in rgb_colors:
        coordinates.append(get_points_new(color_masks, width, height, boundingBox, color, 1, 1, x_range, y_range, extra_info))

    return coordinates, axis_labels, rgb_colors, memo

complex_to_simple_color = {
    'aliceblue': 'white',
    'antiquewhite': 'white',
    'cyan': 'blue',
    'aquamarine': 'green',
    'azure': 'white',
    'beige': 'white',
    'bisque': 'white',
    'black': 'black',
    'blanchedalmond': 'white',
    'blue': 'blue',
    'blueviolet': 'purple',
    'brown': 'red',
    'burlywood': 'orange',
    'cadetblue': 'blue',
    'chartreuse': 'green',
    'chocolate': 'orange',
    'coral': 'orange',
    'cornflowerblue': 'blue',
    'cornsilk': 'white',
    'crimson': 'red',
    'darkblue': 'blue',
    'darkcyan': 'blue',
    'darkgoldenrod': 'yellow',
    'darkgray': 'grey',
    'darkgrey': 'grey',
    'darkgreen': 'green',
    'darkkhaki': 'yellow',
    'darkmagenta': 'purple',
    'darkolivegreen': 'green',
    'darkorange': 'orange',
    'darkorchid': 'purple',
    'darkred': 'red',
    'darksalmon': 'orange',
    'darkseagreen': 'green',
    'darkslateblue': 'purple',
    'darkslategray': 'blue',
    'darkslategrey': 'blue',
    'darkturquoise': 'blue',
    'darkviolet': 'purple',
    'deeppink': 'pink',
    'deepskyblue': 'blue',
    'dimgray': 'grey',
    'dimgrey': 'grey',
    'dodgerblue': 'blue',
    'firebrick': 'red',
    'floralwhite': 'white',
    'forestgreen': 'green',
    'magenta': 'pink',
    'gainsboro': 'white',
    'ghostwhite': 'white',
    'gold': 'yellow',
    'goldenrod': 'yellow',
    'gray': 'grey',
    'grey': 'grey',
    'green': 'green',
    'greenyellow': 'green',
    'honeydew': 'white',
    'hotpink': 'pink',
    'indianred': 'pink',
    'indigo': 'purple',
    'ivory': 'white',
    'khaki': 'yellow',
    'lavender': 'white',
    'lavenderblush': 'white',
    'lawngreen': 'green',
    'lemonchiffon': 'white',
    'lightblue': 'blue',
    'lightcoral': 'pink',
    'lightcyan': 'white',
    'lightgoldenrodyellow': 'white',
    'lightgray': 'grey',
    'lightgrey': 'grey',
    'lightgreen': 'green',
    'lightpink': 'pink',
    'lightsalmon': 'orange',
    'lightseagreen': 'blue',
    'lightskyblue': 'blue',
    'lightslategray': 'grey',
    'lightslategrey': 'grey',
    'lightsteelblue': 'blue',
    'lightyellow': 'white',
    'lime': 'green',
    'limegreen': 'green',
    'linen': 'white',
    'maroon': 'red',
    'mediumaquamarine': 'green',
    'mediumblue': 'blue',
    'mediumorchid': 'purple',
    'mediumpurple': 'purple',
    'mediumseagreen': 'green',
    'mediumslateblue': 'purple',
    'mediumspringgreen': 'green',
    'mediumturquoise': 'blue',
    'mediumvioletred': 'pink',
    'midnightblue': 'blue',
    'mintcream': 'white',
    'mistyrose': 'white',
    'moccasin': 'yellow',
    'navajowhite': 'yellow',
    'navy': 'blue',
    'oldlace': 'white',
    'olive': 'green',
    'olivedrab': 'green',
    'orange': 'orange',
    'orangered': 'red',
    'orchid': 'purple',
    'palegoldenrod': 'yellow',
    'palegreen': 'green',
    'paleturquoise': 'blue',
    'palevioletred': 'pink',
    'papayawhip': 'white',
    'peachpuff':'orange',
    'peru': 'orange',
    'pink': 'pink',
    'plum': 'purple',
    'powderblue': 'blue',
    'purple': 'purple',
    'red': 'red',
    'rosybrown': 'pink',
    'royalblue': 'blue',
    'saddlebrown': 'orange',
    'salmon': 'pink',
    'sandybrown': 'orange',
    'seagreen': 'green',
    'seashell': 'white',
    'sienna': 'orange',
    'silver': 'grey',
    'skyblue': 'blue',
    'slateblue': 'purple',
    'slategray': 'gray',
    'slategrey': 'grey',
    'snow': 'white',
    'springgreen': 'green',
    'steelblue': 'blue',
    'tan': 'orange',
    'teal': 'blue',
    'thistle': 'pink',
    'tomato': 'orange',
    'turquoise': 'blue',
    'violet': 'pink',
    'wheat': 'yellow',
    'white': 'white',
    'whitesmoke': 'white',
    'yellow': 'yellow',
    'yellowgreen': 'green'
}