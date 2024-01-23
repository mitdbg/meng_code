import numpy as np
from PIL import Image, ImageEnhance
from scipy.spatial import KDTree
from webcolors import (
    CSS3_HEX_TO_NAMES,
    hex_to_rgb,
)

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
    css3_db = CSS3_HEX_TO_NAMES
    color_names = []
    color_rgb_values = []
    for color_hex, color_name in css3_db.items():
        color_names.append(color_name)
        color_rgb_values.append(hex_to_rgb(color_hex))
    return color_names, color_rgb_values

COLOR_NAMES, COLOR_RGB_VALUES = get_all_color_names()
KDT_DB = KDTree(COLOR_RGB_VALUES)

def convert_rgb_to_names(rgb_tuple, memo):
    _, index = KDT_DB.query(rgb_tuple)
    simple_color = complex_to_simple_color[COLOR_NAMES[index]]
    memo[simple_color] = memo.get(simple_color, 0) + 1
    return simple_color

def get_color_masks(image, rgb_colors):

    width, height = image.size
    masks = {}
    for c in rgb_colors:
        masks[c] = np.zeros((width, height))
    
    memo = {}
    for y in range(0, height):
        for x in range(0, width):

            r, g, b = image.getpixel((x, y))

            new_color = convert_rgb_to_names([r,g,b], memo)

            if new_color == "blue":
                print([r, g, b])
                print(new_color)

            if new_color in rgb_colors:
                masks[new_color][x, y] = 1
                
    return masks, width, height, memo

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
    'darkturquoise': 'blue',
    'darkviolet': 'purple',
    'deeppink': 'pink',
    'deepskyblue': 'blue',
    'dimgray': 'grey',
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
    'lightgreen': 'green',
    'lightpink': 'pink',
    'lightsalmon': 'orange',
    'lightseagreen': 'blue',
    'lightskyblue': 'blue',
    'lightslategray': 'grey',
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