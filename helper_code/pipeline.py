import os
import matplotlib.pyplot as plt
from helper_code.full_extraction_pipeline import is_valid_color, get_reconstructed_plot
from helper_code.dual_axis import second_axis_clustering_elbow_method

def run_pipeline(image_num, min_samples, predictor, model):
    directory_path = f"../results/{image_num}"

    # Check if the directory exists, and create it if it doesn't
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' was created.")
    else:
        print(f"Directory '{directory_path}' already exists.")

    coordinates, x_axis_title, y_axis_title, second_y_axis_title, axis_labels, rgb_colors, x_range, y_range, memo, second_y_range = get_reconstructed_plot(image_num, predictor, model, True, 30)

    plt.figure()
    for index, value in enumerate(coordinates):
        x = []
        y = []

        for i in range(len(value)):
            x.append(value[i]["middle"][0])
            y.append(value[i]["middle"][1])

        color_normalized = rgb_colors[index]

        if (is_valid_color(color_normalized)): plt.scatter(x, y, label=axis_labels[index], color=color_normalized, s=1)

    plt.xlim(x_range[0], x_range[1])
    plt.ylim(y_range[0], y_range[1])
    plt.xlabel(x_axis_title)
    plt.ylabel(y_axis_title)
    plt.legend()
    plt.savefig("../results/"+str(image_num)+"/final_one_axis.png")
    plt.show()

    if second_y_axis_title:
        try:
            all_downsampled_data = second_axis_clustering_elbow_method(image_num, rgb_colors, coordinates, axis_labels, x_axis_title, y_axis_title, second_y_axis_title, min_samples, y_range, second_y_range)
            all_downsampled_data.to_csv('../results/'+str(image_num)+'/points.csv', index=False)
        except:
            print("ERROR CLUSTERING IMAGE", image_num)