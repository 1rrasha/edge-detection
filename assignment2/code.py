import cv2  # Import OpenCV for image processing
import numpy as np  # Import NumPy for numerical operations
import matplotlib.pyplot as plt  # Import Matplotlib for plotting

#rasha mansour - 1210773

# Function to apply Roberts edge detection
def apply_roberts(image):
    # Define the Roberts cross operator kernels
    kernel_x = np.array([[1, 0], [0, -1]], dtype=int)
    kernel_y = np.array([[0, 1], [-1, 0]], dtype=int)
    
    # Apply the kernels to the image
    gx = cv2.filter2D(image, cv2.CV_64F, kernel_x)
    gy = cv2.filter2D(image, cv2.CV_64F, kernel_y)
    
    # Calculate the magnitude of the gradient
    edges = np.sqrt(gx**2 + gy**2)
    return edges

# Function to apply Sobel edge detection
def apply_sobel(image):
    # Apply the Sobel operator to get gradients in x and y directions
    gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    
    # Calculate the magnitude of the gradient
    edges = np.sqrt(gx**2 + gy**2)
    return edges

# Function to apply Prewitt edge detection
def apply_prewitt(image):
    # Define the Prewitt operator kernels
    kernel_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=int)
    kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
    
    # Apply the kernels to the image
    gx = cv2.filter2D(image, cv2.CV_64F, kernel_x)
    gy = cv2.filter2D(image, cv2.CV_64F, kernel_y)
    
    # Calculate the magnitude of the gradient
    edges = np.sqrt(gx**2 + gy**2)
    return edges

# Function to manually mark expected edge locations on the image
def manual_mark_edges(image):
    # Here, you would manually mark the expected edge locations on the image
    # For demonstration purposes, let's assume we create a binary image with marked edges
    manual_marked = np.zeros_like(image, dtype=np.uint8)
    manual_marked[100:200, 150:250] = 255  # Example: marking edges in a specific region
    return manual_marked

# Load the image
image = cv2.imread('lena.jpg', 0)

# Manually mark expected edge locations on the image
manual_marked = manual_mark_edges(image)

# Apply edge detection methods
roberts_edges = apply_roberts(image)
sobel_edges = apply_sobel(image)
prewitt_edges = apply_prewitt(image)

# Display the manually marked edges and the detected edges
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.title('Manual Marked')
plt.imshow(manual_marked, cmap='gray')
plt.subplot(2, 2, 2)
plt.title('Roberts Edges')
plt.imshow(roberts_edges, cmap='gray')
plt.subplot(2, 2, 3)
plt.title('Sobel Edges')
plt.imshow(sobel_edges, cmap='gray')
plt.subplot(2, 2, 4)
plt.title('Prewitt Edges')
plt.imshow(prewitt_edges, cmap='gray')
plt.show()

# Evaluate the results
# Compare the automatically detected edges with the manual markings
# Calculate metrics or visually inspect the results to determine the similarity

# Calculate the intersection over union (IoU) between manual markings and detected edges
def calculate_iou(manual, detected):
    intersection = np.logical_and(manual, detected)
    union = np.logical_or(manual, detected)
    iou = np.sum(intersection) / np.sum(union)
    return iou

# Calculate IoU for each method
iou_roberts = calculate_iou(manual_marked, roberts_edges)
iou_sobel = calculate_iou(manual_marked, sobel_edges)
iou_prewitt = calculate_iou(manual_marked, prewitt_edges)

print("Intersection over Union (IoU) Results:")
print(f"Roberts: {iou_roberts}")
print(f"Sobel: {iou_sobel}")
print(f"Prewitt: {iou_prewitt}")

# Can adjusting the threshold value in the detection step improve the results?
# Experiment with different threshold values for each edge detection method
# Adjust the threshold value and display the thresholded results

# Threshold values to experiment with
threshold_values = [50, 100, 150, 200]

for thresh in threshold_values:
    # Apply different threshold values
    _, roberts_thresh = cv2.threshold(roberts_edges, thresh, 255, cv2.THRESH_BINARY)
    _, sobel_thresh = cv2.threshold(sobel_edges, thresh, 255, cv2.THRESH_BINARY)
    _, prewitt_thresh = cv2.threshold(prewitt_edges, thresh, 255, cv2.THRESH_BINARY)
    
    # Display the results with different threshold values
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.title(f'Roberts - Threshold {thresh}')
    plt.imshow(roberts_thresh, cmap='gray')
    plt.subplot(2, 2, 2)
    plt.title(f'Sobel - Threshold {thresh}')
    plt.imshow(sobel_thresh, cmap='gray')
    plt.subplot(2, 2, 3)
    plt.title(f'Prewitt - Threshold {thresh}')
    plt.imshow(prewitt_thresh, cmap='gray')
    plt.show()
