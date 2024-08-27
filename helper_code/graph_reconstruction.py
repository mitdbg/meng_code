## GRAPH RECONSTRUCTION

def get_middle_coordinate(top_left_x, top_left_y, wBox, hbox):
    '''
    Get the center of the box
    '''
    return top_left_x + wBox // 2, top_left_y + hbox // 2

def convertPoint(point, boundingBox, width, height, x_axis, y_axis):

    b_width = boundingBox["bottomRight"][0] - boundingBox["topLeft"][0]
    b_height = boundingBox["bottomRight"][1] - boundingBox["topLeft"][1]
    point_x, point_y = point

    x = (point_x - boundingBox["topLeft"][0]) / b_width
    y = (boundingBox["bottomRight"][1] - point_y) / b_height

    x *= (x_axis[1] - x_axis[0])
    x += x_axis[0]

    y *= (y_axis[1] - y_axis[0])
    y += y_axis[0]
    
    return [x, y]

def get_filtered_answer(mask, x0, y0, wBox, hBox, max_x, max_y):
    for x in range(x0, x0 + wBox + 1):
        for y in range(y0, y0 + hBox + 1):
            if x <= max_x and y <= max_y and mask[x, y] == 1: 
                return True
    return False

def get_points_new(color_masks, width, height, boundingBox, color, wBox, hBox, x_axis, y_axis, ocr_mask):

    graph = []
    mask = color_masks[color]

    for x0 in range(boundingBox["topLeft"][0], boundingBox["bottomRight"][0], wBox):
        for y0 in range(boundingBox["topLeft"][1], boundingBox["bottomRight"][1], hBox):

            if ocr_mask is not None and ocr_mask[y0, x0] == 1:
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