import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D


# Paths for each of the images
# Path for cloud image
path1 = "/home/ddlouk/devel/project1/Pics/detection_pic1.png"

# Path for NDVI image
path2 = "/home/ddlouk/devel/project1/Pics/detection_pic2.png"

# Path for truecolor image 
path3 = "/home/ddlouk/devel/project1/Pics/detection_pic3.png"

# Image loading
# Cloud image loading
cloud_image = Image.open(path1)

# Checking if image already RGB mode
'''
if cloud_image.mode == "RGB":
    print("The cloud image is already in RGB mode.")
else:
    print("The cloud image is not in RGB mode.")
'''

# Convert the image to the RGB mode
cloud_rgb_image = cloud_image.convert("RGB")

# Get the pixels of the image
pixels = cloud_rgb_image.load()

# NDVI image loading 
ndvi_image = Image.open(path2)

# Truecolor image loading
truecolor_image = Image.open(path3)

# Checking if image already RGB mode
'''
if truecolor_image.mode == "RGB":
    print("The truecolor image is already in RGB mode.")
else:
    print("The truecolor image is not in RGB mode.")
'''

# Convert the image to the RGB mode
truecolor_rgb_image = truecolor_image.convert("RGB")

# Get the pixels of the image
pixels2 = truecolor_rgb_image.load()

# Experimenting with pixel data
'''
# Accessing and manipulating the individual pixel values
width, height = cloud_rgb_image.size
for x in range(width):
    for y in range(height):
        # Get the RGB values of the pixel at (x, y)
        r, g, b = pixels[x, y]

        # Perform operations on the pixel values (inversion)
        inverted_color = (255 - r, 255 - g, 255 - b)

        # Modify the pixel with the new values
        pixels[x, y] = inverted_color

# Save the modified image
cloud_rgb_image.save("inverted_rgb_image.png")
'''

#################################
# CLOUD IMAGE ANOMALY DETECTION #
#           K-MEANS             #
#################################

# Prepare the RGB data for K-means clustering
width, height = cloud_rgb_image.size
rgb_data = np.empty((width * height, 3), dtype=np.float32)

for x in range(width):
    for y in range(height):
        # Get the RGB values of the pixel at (x, y)
        r, g, b = pixels[x, y]

        # Store the RGB values in the data array
        rgb_data[x * height + y] = [r, g, b]

# K-Means Anomaly detection function for cloud data
def cloud_kmeans_anomaly_detection(data):
    # Normalize the data
    scaler = StandardScaler()
    rgb_data_normalized = scaler.fit_transform(data)

    # Perform K-means clustering
    n_clusters = 3  # Choose the number of clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(rgb_data_normalized)

    # Get the cluster centroids
    centroids = kmeans.cluster_centers_

    # Calculate the distance of each data point from its corresponding centroid
    distances = np.linalg.norm(rgb_data_normalized - centroids[kmeans.labels_], axis=1)

    # Define a threshold for anomaly detection (adjust as needed)
    threshold = np.percentile(distances, 95)

    # Identify anomalies (data points with distances above the threshold)
    anomalies = data[distances > threshold]

    # Visualize the clustered RGB data
    '''
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(rgb_data_normalized)), rgb_data_normalized, c=kmeans.labels_, cmap='viridis')
    plt.scatter(np.where(distances > threshold)[0], rgb_data_normalized[distances > threshold], c='red', label='Anomalies')
    plt.scatter(range(n_clusters), centroids, c='black', marker='x', s=100, label='Cluster Centers')
    plt.xlabel('Pixel Index')
    plt.ylabel('Normalized RGB Values')
    plt.legend()
    plt.title('K-means Anomaly Detection on RGB Image Data')
    plt.show()
    '''

    # Visualize clusters in 3D
    def visualize_clusters_3d(data, labels, centroids, anomalies, title):
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')

        # Plotting normal data points
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels, cmap='viridis', label='Normal Data')

        if anomalies.shape[0] > 0:  # Check if there are any anomalies
            # Plotting anomalies
            ax.scatter(anomalies[:, 0], anomalies[:, 1], anomalies[:, 2], c='red', label='Anomalies')

        # Plotting cluster centroids
        ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], c='black', marker='x', s=100, label='Cluster Centers')

        ax.set_xlabel('R')
        ax.set_ylabel('G')
        ax.set_zlabel('B')
        ax.set_title(title)
        ax.legend()

        plt.show()

    visualize_clusters_3d(rgb_data_normalized, kmeans.labels_, centroids, anomalies, 'K-means Anomaly Detection on Cloud RGB Data')

    # Output the anomalies
    print("Cloud RGB K-Means anomalies:")

    return anomalies


# DES AYTO TO KOMMATI KODIKA ###########################################
'''
# Function to perform K-means clustering on features and create a new image
def kmeans_clustering_image(data, n_clusters, output_path):
    # Normalize the data
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(data)

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(normalized_data)

    # Get cluster assignments for each data point
    cluster_labels = kmeans.labels_

    # Get cluster centroids
    cluster_centers = kmeans.cluster_centers_

    # Create a new image based on cluster centroids
    new_image = Image.new("RGB", (width, height))
    new_pixels = new_image.load()

    for x in range(width):
        for y in range(height):
            cluster_label = cluster_labels[x * height + y]
            new_color = tuple(int(val * 255) for val in cluster_centers[cluster_label])
            new_pixels[x, y] = new_color

    # Save the new image
    new_image.save(output_path)

    # Visualize the new image
    plt.imshow(new_image)
    plt.title('Clustered Image')
    plt.show()

# Function to extract features from the image (e.g., RGB values)
def extract_features(image):
    width, height = image.size
    features = np.empty((width * height, 3), dtype=np.float32)

    for x in range(width):
        for y in range(height):
            r, g, b = image.getpixel((x, y))
            features[x * height + y] = [r, g, b]

    return features

def test():
    # Extract features from the cloud RGB image
    cloud_features = extract_features(cloud_rgb_image)

    # Perform K-means clustering and create a new image
    kmeans_clustering_image(cloud_features, n_clusters=3, output_path="clustered_cloud_image.png")

test()
'''
#####################################################################

################################
# NDVI IMAGE ANOMALY DETECTION #
#           K-MEANS            #
################################

# Convert the image to a NumPy array
ndvi_data = np.array(ndvi_image)

# K-Means Anomaly detection function for ndvi data
def ndvi_kmeans_anomaly_detection(data):
    # Flatten the 2D array to a 1D array
    ndvi_values = data.flatten()

    # Normalize the data
    scaler = StandardScaler()
    ndvi_values_normalized = scaler.fit_transform(ndvi_values.reshape(-1, 1))

    # Perform K-means clustering
    # Choose the number of clusters
    n_clusters = 3
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(ndvi_values_normalized)

    # Get the cluster centroids
    centroids = kmeans.cluster_centers_

    # Calculate the distance of each data point from its corresponding centroid
    distances = np.linalg.norm(ndvi_values_normalized - centroids[kmeans.labels_], axis=1)

    # Define a threshold for anomaly detection (adjust as needed)
    threshold = np.percentile(distances, 95)

    # Identify anomalies (data points with distances above the threshold)
    anomalies = ndvi_values[distances > threshold]

    # Visualize the clustered NDVI data
    '''
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(ndvi_values_normalized)), ndvi_values_normalized, c=kmeans.labels_, cmap='viridis')
    plt.scatter(np.where(distances > threshold)[0], ndvi_values_normalized[distances > threshold], c='red', label='Anomalies')
    plt.scatter(range(n_clusters), centroids, c='black', marker='x', s=100, label='Cluster Centers')
    plt.xlabel('Pixel Index')
    plt.ylabel('Normalized NDVI Values')
    plt.legend()
    plt.title('K-means Anomaly Detection on NDVI Data')
    plt.show()
    '''
    return anomalies

def visualize_clusters_3d(data, labels, centroids, anomalies, title):
    if data.shape[1] == 3:  # Check if the data is 3D
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')

        # Plotting normal data points
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels, cmap='viridis', label='Normal Data')

        if anomalies.shape[0] > 0:  # Check if there are any anomalies
            # Plotting anomalies
            ax.scatter(anomalies[:, 0], anomalies[:, 1], anomalies[:, 2], c='red', label='Anomalies')

        # Plotting cluster centroids
        ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], c='black', marker='x', s=100, label='Cluster Centers')

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.set_title(title)
        ax.legend()
    else:
        # Data is not 3D, adapt the visualization to 2D or 1D
        if data.shape[1] == 2:
            # 2D scatter plot
            plt.figure(figsize=(10, 6))
            plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', label='Normal Data')

            if anomalies.shape[0] > 0:
                plt.scatter(anomalies[:, 0], anomalies[:, 1], c='red', label='Anomalies')

            plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='x', s=100, label='Cluster Centers')

            plt.xlabel('X Label')
            plt.ylabel('Y Label')
        else:
            # 1D scatter plot
            plt.figure(figsize=(10, 6))
            plt.scatter(data, labels, c=labels, cmap='viridis', label='Normal Data')

            if anomalies.shape[0] > 0:
                plt.scatter(anomalies, labels[anomalies], c='red', label='Anomalies')

            plt.scatter(centroids, [0] * len(centroids), c='black', marker='x', s=100, label='Cluster Centers')

            plt.xlabel('X Label')

        plt.title(title)
        plt.legend()

    plt.show()


#####################################
# TRUECOLOR IMAGE ANOMALY DETECTION #
#           K-MEANS                 #
#####################################

# Prepare the RGB data for K-means clustering for the truecolor image
width, height = truecolor_rgb_image.size
truecolor_rgb_data = np.empty((width * height, 3), dtype=np.float32)

for x in range(width):
    for y in range(height):
        # Get the RGB values of the pixel at (x, y)
        r, g, b = truecolor_rgb_image.getpixel((x, y)) # Or we can do it like before with pixels2[x, y] this time instead of getpixel(...)

        # Store the RGB values in the data array
        truecolor_rgb_data[x * height + y] = [r, g, b]

# K-Means Anomaly detection function for truecolor data
def truecolor_kmeans_anomaly_detection(data):
    # Normalize the data
    scaler = StandardScaler()
    rgb_data_normalized = scaler.fit_transform(data)

    # Perform K-means clustering
    # Choose the number of clusters
    n_clusters = 3 
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(rgb_data_normalized)

    # Get the cluster centroids
    centroids = kmeans.cluster_centers_

    # Calculate the distance of each data point from its corresponding centroid
    distances = np.linalg.norm(rgb_data_normalized - centroids[kmeans.labels_], axis=1)

    # Define a threshold for anomaly detection (adjust as needed)
    threshold = np.percentile(distances, 95)

    # Identify anomalies (data points with distances above the threshold)
    anomalies = data[distances > threshold]

    # Visualize the clustered RGB data
    '''
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(rgb_data_normalized)), rgb_data_normalized, c=kmeans.labels_, cmap='viridis')
    plt.scatter(np.where(distances > threshold)[0], rgb_data_normalized[distances > threshold], c='red', label='Anomalies')
    plt.scatter(range(n_clusters), centroids, c='black', marker='x', s=100, label='Cluster Centers')
    plt.xlabel('Pixel Index')
    plt.ylabel('Normalized RGB Values')
    plt.legend()
    plt.title('K-means Anomaly Detection on RGB Image Data')
    plt.show()
    '''

def visualize_clusters_3d(data, labels, centroids, anomalies, title):
    if data.shape[1] == 3:  # Check if the data is 3D
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')

        # Plotting normal data points
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels, cmap='viridis', label='Normal Data')

        if anomalies.shape[0] > 0:  # Check if there are any anomalies
            # Plotting anomalies
            ax.scatter(anomalies[:, 0], anomalies[:, 1], anomalies[:, 2], c='red', label='Anomalies')

        # Plotting cluster centroids
        ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], c='black', marker='x', s=100, label='Cluster Centers')

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.set_title(title)
        ax.legend()

        plt.show()
    else:
        # Handle 1D data visualization or other cases
        # You can customize this part based on your needs
        print("Data is not in 3D format, cannot visualize in 3D.")


# Calling the functions
def main():
    cloud_anomalies = cloud_kmeans_anomaly_detection(rgb_data)
    ndvi_anomalies = ndvi_kmeans_anomaly_detection(ndvi_data)
    truecolor_anomalies = truecolor_kmeans_anomaly_detection(truecolor_rgb_data)

    print("Cloud Anomalies:", cloud_anomalies)
    print("NDVI Anomalies:", ndvi_anomalies)
    print("Truecolor Anomalies:", truecolor_anomalies)

    
if __name__ == "__main__":
    main()