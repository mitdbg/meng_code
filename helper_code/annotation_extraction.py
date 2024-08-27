import cv2
import matplotlib.pyplot as plt

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


def annotation_extraction_module(ocr_mask, image_num, image_name, image, model):
    result = model.predict(image_name, confidence=40, overlap=30)
    result.plot()

    result.save(f'../results/{image_num}/annotation.png')

    final = result.json()

    if final:
        if "predictions" in final.keys():
            final = final["predictions"]
            legend_box = None

            for classification in final:
                print(classification)
                if classification["class"] == 'legend':
                    legend_box = classification
                x_min = int(classification['x'] - classification['width']/2)
                x_max = int(classification['x'] + classification['width']/2) + 1
                y_min = int(classification['y'] - classification['height']/2)
                y_max = int(classification['y'] + classification['height']/2) + 1

                ocr_mask[y_min:y_max, x_min:x_max] = 1
            
            # Visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            ax1.set_title('Original Image')
            ax1.axis('off')

            ax2.imshow(ocr_mask, cmap='gray')
            ax2.set_title('Annotation Mask')
            ax2.axis('off')

            plt.savefig(f'../results/{image_num}/annotation.png')
            plt.show()
            
            return legend_box, ocr_mask
            
    return None