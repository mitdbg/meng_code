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