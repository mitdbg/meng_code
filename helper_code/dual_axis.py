import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np
import colorsys
from matplotlib.colors import to_rgb
import json
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from collections import Counter
from sklearn.metrics import silhouette_score
import seaborn as sns
from matplotlib.lines import Line2D

from helper_code.metadata_extraction import ask_gpt_clusters

def create_df(rgb_colors, coordinates):
    temp = []
    for index, value in enumerate(coordinates):
        
        for i in range(len(value)):
            temp.append([rgb_colors[index], value[i]["middle"][0], value[i]["middle"][1]])
    
    df = pd.DataFrame(temp, columns=["Color", "X", "Y"])
    return df

def get_optimal_eps_elbow(std_df):
    """
    Based on methods: https://www.kaggle.com/code/tanmaymane18/nearestneighbors-to-find-optimal-eps-in-dbscan

    Args:
        std_df (_type_): _description_

    Returns:
        _type_: _description_
    """    
    # Fit Nearest Neighbors
    nn = NearestNeighbors(n_neighbors=200).fit(std_df)
    distances, indices = nn.kneighbors(std_df)

    # Sort distances
    distances = np.sort(distances, axis=0)
    distances = distances[:, 199]  # The distance to the 200th nearest neighbor

    # Normalize the distances to [0, 1] for both x and y
    norm_distances = distances / distances.max()
    x = np.linspace(0, 1, len(distances))

    # Coordinates of the line
    line_start = np.array([x[0], norm_distances[0]])
    line_end = np.array([x[-1], norm_distances[-1]])

    # Function to calculate distance from a point to a line (start and end)
    def point_line_distance(point, start, end):
        return np.abs(np.cross(end-start, start-point)) / np.linalg.norm(end-start)

    # Calculate distances from each point to the line
    distances_to_line = np.apply_along_axis(point_line_distance, 1, np.vstack((x, norm_distances)).T, line_start, line_end)

    # Find the index of the point with the maximum distance to the line
    elbow_index = np.argmax(distances_to_line)

    eps_best = distances[elbow_index]
    

    # Plot the curve and the elbow point
    plt.figure(figsize=(10,6))
    plt.plot(x, norm_distances, label='K-nearest neighbor distance')
    plt.scatter(x[elbow_index], norm_distances[elbow_index], color='red', s=100, label='Elbow Point', zorder=5)
    plt.title('Elbow Method for Optimal Eps Value')
    plt.xlabel('Points sorted by distance')
    plt.ylabel('Normalized distance to 20th nearest neighbor')
    

    # Customizing legend with bigger bubbles
    handles, labels = plt.gca().get_legend_handles_labels()
    sizes = [200, 200]  # Sizes for the bubbles
    bubble_legend = plt.legend(handles, labels, loc='upper right', scatterpoints=1, fontsize='large')
    for bubble, size in zip(bubble_legend.legendHandles, sizes):
        bubble._sizes = [size]


    plt.show()

    return eps_best

def plot_clusters(data, clusters, entire_color, ax, series_label):
    print(clusters)
    # Convert the named color to an RGB tuple
    base_color = entire_color
    
    # Unique cluster labels
    unique_labels = set(clusters)
    
    random_colors = get_darker_colors(len(unique_labels))
    # Plot each cluster
    for k, col in zip(unique_labels, random_colors):
        # Filter data points that belong to the current cluster
        class_member_mask = (clusters == k)

        print(col)
        
        # Plot data points that are in the cluster
        ax.scatter(data.loc[class_member_mask, 'X'], data.loc[class_member_mask, 'Y'], color=col, s=1, label=series_label + " Cluster #" +str(k))
        ax.legend()
        

        # # Plot the outliers
        # if k == -1:
        #     xy = data[class_member_mask]
        #     ax.plot(xy[:, 0], xy[:, 1], markerfacecolor=col, markersize=1)

def convert_y_value(original_y, first_y_axis, second_y_axis):
    x0, y0 = first_y_axis
    x, y = second_y_axis
    # Apply the linear transformation formula
    new_y = x + (y - x) * (original_y - x0) / (y0 - x0)
    return new_y

# Function to adjust the lightness of a color
def adjust_lightness(color, factor):
    # Convert named color to RGB
    r, g, b = to_rgb(color)
    # Convert RGB to HLS
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    # Adjust lightness
    l = max(min(l * factor, 1), 0)
    # Convert back to RGB
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return r, g, b

def get_darker_colors(unique_labels_count, luminance_threshold=0.5):
    # Create a list of all CSS4 colors
    all_colors = list(mcolors.CSS4_COLORS.values())
    
    # Convert hex to RGB and filter based on luminance
    darker_colors = [color for color in all_colors if rgb_luminance(mcolors.hex2color(color)) < luminance_threshold]
    
    # Select a random subset of the darker colors
    random_colors = [mcolors.to_hex(random.choice(darker_colors)) for _ in range(unique_labels_count)]
    
    return random_colors

def rgb_luminance(color):
    # Calculate the perceived luminance of a color
    # The constants 0.2126, 0.7152, and 0.0722 are used for red, green, and blue respectively, which correspond to the sRGB luminance values.
    return 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2]

def plot_second_clusters(data, clusters, entire_color, ax, ax2, series_label, clusters_dict, first_y_axis, second_y_axis, x_title,  first_y_title, second_y_title):
    print(clusters)
    
    downsampled_data = pd.DataFrame()  # Initialize an empty DataFrame to collect downsampled data
    
    if not clusters_dict:
        ax.scatter(data['X'], data['Y'], color=entire_color, s=20, marker='o', label=series_label)
        ax2.scatter(data['X'], data['Y'], color=entire_color, s=20, marker='o', label=series_label)
        ax.legend(loc='best')
        return downsampled_data  # Return empty DataFrame if no clustering

    base_color = entire_color
    unique_clusters = set(clusters)

    new_colors = [entire_color for i in range(len(unique_clusters))]

    for k, col in zip(unique_clusters, new_colors):
        class_member_mask = (clusters == k)
        new_label = series_label + " Cluster #" + str(k)
        marker = 'o'
        
        data_subset = data[class_member_mask]
        n_samples = int(np.floor(0.4 * len(data_subset)))
        if n_samples == 0:
            continue
        
        downsampled_data_subset = data_subset.sample(n=n_samples, random_state=42)  # Using a fixed random state for reproducibility
        downsampled_data_subset['Axis'] = 'left'  # Default axis
        
        y_values = downsampled_data_subset['Y']
        x_values = downsampled_data_subset['X']

        data_to_add = pd.DataFrame()
        data_to_add[x_title] = x_values
        data_to_add[first_y_title] = None
        data_to_add[second_y_title] = None

        if new_label in clusters_dict:
            axis_side = clusters_dict[new_label]
            
            if axis_side == "right":
                marker = 'v'
                y_values = y_values.apply(convert_y_value, args=(first_y_axis, second_y_axis))
                ax2.scatter(
                    x_values, 
                    y_values, color=col, 
                    s=20, marker=marker, 
                    label=series_label, 
                    edgecolors='black'
                )
    
                data_to_add[second_y_title] = y_values
            
            else:
                ax.scatter(
                    x_values, 
                    y_values, 
                    color=col, 
                    s=20, 
                    marker=marker, 
                    label=series_label, 
                    edgecolors='black'
                )

                data_to_add[first_y_title] = y_values

        else:
            ax.scatter(
                x_values, 
                y_values, 
                color=col, 
                s=20, 
                marker=marker, 
                label=series_label, 
                edgecolors='black'
            )

            data_to_add[first_y_title] = y_values
        
        data_to_add['Type'] = series_label

        downsampled_data = pd.concat([downsampled_data, data_to_add], ignore_index=True)

        print("Color", col)
        print("Label", new_label)
        print("Marker", marker)

    print("END OF THIS")
    return downsampled_data  # Return the collected downsampled data DataFrame with axis information

def load_second_axis(data, left_axis_title, right_axis_title):

    # Initialize dictionary
    clusters_dict = {}

    # Process each cluster in the JSON data
    for cluster in data['clusters']:
        title = cluster['title']
        axis = cluster['axis']
        axis_title = cluster["axis_title"]
        
        if axis == 'left':
            left_axis_title = axis_title
        elif axis == 'right':
            right_axis_title = axis_title
        
        clusters_dict[title] = axis
    
    return left_axis_title, right_axis_title, clusters_dict

def finalize_legend(ax, ax2):
    # Create legend with bigger bubbles
    handles, labels = ax.get_legend_handles_labels()
    new_handles = handles
    new_labels = labels 
    if ax2:
        handles2, labels2 = ax2.get_legend_handles_labels()
        new_handles.extend(handles2)
        new_labels.extend(labels2) 

    legend = ax.legend(new_handles, new_labels, scatterpoints=1, fontsize='large', loc='upper left', bbox_to_anchor=(1.05, 1))
    
    # Customize bubble sizes in the legend
    for handle in legend.legendHandles:
        handle._sizes = [200]  # Set all sizes to 200 or any other value you prefer

    return "HERE"
    

def second_axis_clustering_elbow_method(image_num, rgb_colors, coordinates, series_labels, x_axis_title, y_axis_title, second_y_axis_title, min_samples_best, first_y_range, second_y_range):
    df = create_df(rgb_colors, coordinates)
    all_downsampled_data = pd.DataFrame()  # Initialize a DataFrame to collect all downsampled data


    clusters = {}
    for unique_color in rgb_colors:
        try:
            filtered_df = df[df['Color'] == unique_color]

            # Standardize the coordinates
            st = StandardScaler()
            std_df = pd.DataFrame(st.fit_transform(filtered_df[['X', 'Y']]), columns=['X', 'Y'])

            eps_best = get_optimal_eps_elbow(std_df)

            labels = DBSCAN(min_samples=min_samples_best, eps = eps_best).fit(std_df).labels_
            colored_clusters = len(Counter(labels))
            clusters[unique_color] = labels
            
            print(f"Number of clusters: {colored_clusters}")
            print(f"Number of outliers: {Counter(labels)[-1]}")
            print(f"Silhouette_score: {silhouette_score(std_df, labels)}")

            plt.figure(figsize=(10,8))
            sns.scatterplot(x=std_df['X'], y=std_df['Y'], hue=labels, palette='viridis')

        except:
            continue

    # Create a plot
    fig, ax = plt.subplots(figsize=(12, 8))

    print("HERE", second_y_range)
    ax2 = ax.twinx()
    ax.set_ylabel(y_axis_title)
    ax2.set_ylabel(second_y_axis_title)  # For the second y-axis
    ax2.set_ylim(second_y_range[0], second_y_range[1])  # Set limits for the second y-axis
    
    print("PLOTTING")
    # Plot each color series
    for color, label in zip(rgb_colors, series_labels):
        try:
            filtered_df = df[df['Color'] == color]
            points = filtered_df[['X', 'Y']]
            plot_clusters(points, clusters[color], color, ax, label)
        except:
            continue
    
    # Create legend with bigger bubbles
    handles, labels = ax.get_legend_handles_labels()
    sizes = [200] * len(handles)  # Ensure all bubbles are the same size
    legend = ax.legend(handles, labels, scatterpoints=1, fontsize='large')
    
    # Customize bubble sizes in the legend
    for handle in legend.legendHandles:
        handle._sizes = [200]  # Set all sizes to 200 or any other value you prefer


    # Title and labels
    plt.title('DBSCAN Clustering of Multicolored Data Points')
    plt.xlabel(x_axis_title)

    plt.savefig("../results/"+str(image_num)+"/cluster.png")
    # Show the plot
    plt.show()

    clusters_metadata = { "clusters" : [] }

    try: 
        response = ask_gpt_clusters("../plot_images/"+str(image_num)+".png", "../results/" + str(image_num) + "/cluster.png")
        answer = response["choices"][0]["message"]["content"]
        print(answer)
        cleaned = answer.replace('```json\n', '').replace('\n```', '')
        clusters_metadata = json.loads(cleaned)
    except:
        for label in labels:
            new_dict = {
                "title": label,
                "axis": "left",
                "axis_title": y_axis_title
            }
            clusters_metadata["clusters"].append(new_dict)
    
    print(clusters_metadata)

    with open("../results/" + str(image_num) + "/clusters_metadata.json", 'w') as json_file:
        json.dump(clusters_metadata, json_file, indent=4)

    print("CLUSTERED")
    
    left_axis_title, right_axis_title, clusters_dict = load_second_axis(clusters_metadata, y_axis_title, second_y_axis_title)

    print(clusters_dict)
    print("here")
    ## UPDATED PLOT
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_ylim(first_y_range[0], first_y_range[1])
    ax2 = ax.twinx()
    ax2.set_ylim(second_y_range[0], second_y_range[1])  # Set limits for the second y-axis
    ax.set_ylabel(left_axis_title)
    ax2.set_ylabel(right_axis_title)  # For the second y-axis

    print("here")
    # Plot each color series
    for color, label in zip(rgb_colors, series_labels):
        try:
            filtered_df = df[df['Color'] == color]
            points = filtered_df[['X', 'Y']]
            downsampled_data = plot_second_clusters(points, clusters[color], color, ax, ax2, label, clusters_dict, first_y_range, second_y_range, x_axis_title, left_axis_title, right_axis_title)
            all_downsampled_data = pd.concat([all_downsampled_data, downsampled_data], ignore_index=True)
        except:
            continue
    print(finalize_legend(ax, ax2))

    # Title and labels
    plt.title('Final')
    ax.set_xlabel(x_axis_title)
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust the right margin to make room for the legend

    plt.savefig("../results/"+str(image_num)+"/final_dual_axis.png")
    # Show the plot
    plt.show()

    return all_downsampled_data