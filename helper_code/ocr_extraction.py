import boto3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize_ocr(image, ocr_results):
    # Visualization (assuming you want to visualize the results similarly)
    plt.figure()
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)

    for text, box in ocr_results:
            # Extract the bounding box coordinates
            start_point = box[0]
            end_point = box[2]
            box_width = end_point[0] - start_point[0]
            box_height = end_point[1] - start_point[1]
            
            # Create a Rectangle patch
            rect = patches.Rectangle(start_point, box_width, box_height, linewidth=1, edgecolor='r', facecolor='none')
            
            # Add the rectangle to the Axes
            ax.add_patch(rect)
            
            # Annotate the image with the OCR'ed text
            ax.text(start_point[0], start_point[1] - 10, text, bbox=dict(fill=False, edgecolor='red', linewidth=2), color='red')

    plt.axis('off')  # Turn off axis numbers and ticks
    plt.show()

def get_ocr_mask_and_results(image, image_name):
    # Initialize the Textract client
    textract = boto3.client('textract')

    height, width, _ = image.shape
    mask = np.zeros((height, width), dtype=np.uint8)

    # Load the image file
    with open(image_name, 'rb') as document:
        img_test = bytearray(document.read())

    # Call Amazon Textract
    ocr_response = textract.detect_document_text(Document={'Bytes': img_test})

    # Process Textract response to match desired output format
    ocr_results = []
    for item in ocr_response['Blocks']:
        if item['BlockType'] == 'LINE' and 'Confidence' in item and item['Confidence'] > 50:  # Adjust confidence as needed
            text = item['Text']
            # Extract bounding box coordinates scaled to image dimensions
            height, width, _ = image.shape
            box = item['Geometry']['BoundingBox']
            x = int(box['Left'] * width)
            y = int(box['Top'] * height)
            w = int(box['Width'] * width)
            h = int(box['Height'] * height)
            box = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
            ocr_results.append((text, box))

            # Mark the region in the mask as 1
            mask[y:y+h, x:x+w] = 1

    return ocr_results, mask

def ocr_extraction_module(image, image_name):
    ocr_results, mask = get_ocr_mask_and_results(image, image_name)
    visualize_ocr(image, ocr_results)
    return ocr_results, mask

