from core import Diagram
from core.annotate import *
from core.draw import *
from core.interface import *
from core.parse import *
import cv2
import numpy as np
import os
import pandas as pd

def annotate(ann_path, img_path, out_path):
    if not os.path.exists(img_path) or not os.path.exists(ann_path):
        print("[ERROR] {} or {} is missing. Skipping diagram.".format(
            img_path, ann_path
        ))
        return
    
    # Initialise diagram
    diagram = Diagram(ann_path, img_path)

    # Create a diagram with original annotation
    diagram.graph = create_graph(diagram.annotation,
                                 edges=True, arrowheads=True)

    # Draw layout segmentation
    layout = draw_layout(img_path, diagram.annotation,
                         height=720, dpi=200, arrowheads=True)

    cv2.imwrite(out_path, layout)

    return "HI"

if __name__ == "__main__":
    ann_path = "../ai2d/annotations/0.png.json"
    img_path = "../ai2d/images/0.png"
    out_path = "../ai2d/annotated_images/0.png"
    annotate(ann_path, img_path, out_path)