import numpy as np
import matplotlib.pyplot as plt
import cv2

def keep_largest_component(mask):
    """
    Keep only the largest connected component in the mask.

    Parameters:
    mask (numpy.ndarray): The binary mask.

    Returns:
    numpy.ndarray: The mask with only the largest connected component.
    """
    # Find connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    # If there's no component (other than background), return the same mask
    if num_labels <= 1:
        return mask

    # Find the largest component, ignoring the background (label 0)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    
    # Create a mask of the largest component
    largest_component = (labels == largest_label).astype(np.uint8)

    return largest_component

def clean_mask_edges_and_convert_back(mask, kernel_size=3):
    # Convert the boolean mask to uint8 format
    mask_uint8 = (mask * 255).astype(np.uint8)

    # Create the kernel for morphological operations
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Perform opening to remove noise and then closing to close small holes
    mask_cleaned = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
    mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel)

    # Keep only the largest connected component
    largest_component = keep_largest_component(mask_cleaned)

    return largest_component

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    
    
def get_main_mask(masks, scores, image=None):
    sorted_masks = sorted(zip(masks, scores), key=lambda x: x[1], reverse=True)
    main_mask = sorted_masks[0][0]
    main_mask_score = sorted_masks[0][1]
    if image is not None:
        plt.figure(figsize=(10,10))
        plt.imshow(image)
        show_mask(main_mask, plt.gca())
    return main_mask, main_mask_score 

def get_bounding_box(main_mask, new_image):

    boundingBox = {
        "topLeft": None,
        "bottomRight": [0,0],
    }

    topLeft = False

    for i in range(len(new_image)):
        for j in range(len(new_image[0])):
            if main_mask[i][j]:
                if not topLeft:
                    boundingBox["topLeft"] = [j, i]
                    topLeft = True
                boundingBox["topLeft"] = [min(boundingBox["topLeft"][0], j), min(boundingBox["topLeft"][1], i)]
                boundingBox["bottomRight"] = [max(boundingBox["bottomRight"][0], j), max(boundingBox["bottomRight"][1], i)]
    
    return new_image, boundingBox