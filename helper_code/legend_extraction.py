import numpy as np
import cv2

# LEGEND EXTRACTION
def get_legend_extraction(image_name, model):

    # visualize your prediction
    result = model.predict(image_name, confidence=40, overlap=30)
    result.plot()

    final = result.json()
    if final:
        if "predictions" in final.keys():
            final = final["predictions"]
            if len(final) > 0:
                legend_classification = final[0]
                return {
                    'top_x': legend_classification['x'], 
                    'top_y': legend_classification['y'], 
                    'width': legend_classification['width'], 
                    'height': legend_classification['height']
                }
    return None

def extracted_mask(image_num, image_name, model):
    # visualize your prediction
    result = model.predict(image_name, confidence=40, overlap=30)
    result.plot()

    result.save(f'../results/{image_num}/annotation.png')

    final = result.json()

    image = cv2.imread(image_name)
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)

    if final:
        if "predictions" in final.keys():
            final = final["predictions"]
            for classification in final:
                x_min = int(classification['x'] - classification['width']/2)
                x_max = int(classification['x'] + classification['width']/2) + 1
                y_min = int(classification['y'] - classification['height']/2)
                y_max = int(classification['y'] + classification['height']/2) + 1

                mask[y_min:y_max, x_min:x_max] = 1
            
            return mask
            
    return None