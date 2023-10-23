# Image Anomaly Detection with K-Means Clustering

## Overview

This project is aimed at detecting anomalies in various types of images using K-Means clustering. It includes code for analyzing cloud images, NDVI images, and truecolor images to identify regions that deviate from the norm. Anomalies can be indicative of issues or unusual patterns that may need attention.

## Prerequisites

Before running the code, ensure you have the following dependencies installed:

- Python 3
- Required Python packages (you can install them via `pip install <package_name>`):
    - pandas
    - numpy
    - matplotlib
    - scikit-learn
    - pillow (PIL)
    
## Usage

1. **Data Preparation:**

   Place your image files in the specified paths (`path1`, `path2`, `path3`) in the script. Make sure the images are in RGB format.

2. **Running the Anomaly Detection:**

   Run the `main()` function at the end of the script to perform anomaly detection for the provided images. Detected anomalies will be displayed in the console.

3. **Understanding the Output:**

   - Detected anomalies in the image are printed in the console.
   - For visualizations, 3D scatter plots show the clusters of normal data and anomalies. Anomalies are highlighted in red.

4. **Adapt the Code:**

   The code can be adapted for other types of image data by modifying the path to the image and the anomaly detection method.

## Configuration

- The number of clusters for K-Means clustering can be adjusted to optimize the detection threshold for your specific use case.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- The code uses popular Python libraries and follows best practices in anomaly detection.
- Inspiration for the project came from the need to identify anomalies in various types of image data.

