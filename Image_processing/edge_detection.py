import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os
import math
import preprocessing as pr 
import colour_and_segmentation as cs 


def sobel_operator(image):
    """Apply Sobel operator for edge detection"""
    # Ensure image is grayscale
    if len(image.shape) == 3:
        gray_image = cs.rgb_to_grayscale(image)
    else:
        gray_image = image
    
    # Define Sobel kernels
    sobel_x = np.array([[-1, 0, 1], 
                        [-2, 0, 2], 
                        [-1, 0, 1]])
    
    sobel_y = np.array([[-1, -2, -1], 
                        [0, 0, 0], 
                        [1, 2, 1]])
    
    # Get image dimensions
    height, width = gray_image.shape
    
    # Pad image
    padded_image = np.pad(gray_image, 1, mode='reflect')
    
    # Initialize gradient magnitude and direction
    gradient_magnitude = np.zeros((height, width))
    gradient_direction = np.zeros((height, width))
    
    # Calculate gradients
    for i in range(height):
        for j in range(width):
            # Get 3x3 neighborhood
            neighborhood = padded_image[i:i+3, j:j+3]
            
            # Calculate gradients
            gx = np.sum(neighborhood * sobel_x)
            gy = np.sum(neighborhood * sobel_y)
            
            # Calculate magnitude and direction
            gradient_magnitude[i, j] = np.sqrt(gx**2 + gy**2)
            gradient_direction[i, j] = np.arctan2(gy, gx)
    
    return gradient_magnitude, gradient_direction

def non_maximum_suppression(gradient_magnitude, gradient_direction):
    """Apply non-maximum suppression to thin edges"""
    height, width = gradient_magnitude.shape
    suppressed = np.zeros((height, width))
    
    # Convert radians to degrees and make positive
    angle = gradient_direction * 180 / np.pi
    angle[angle < 0] += 180
    
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            # Get gradient direction
            theta = angle[i, j]
            
            # Find neighboring pixels in gradient direction
            if (0 <= theta < 22.5) or (157.5 <= theta <= 180):
                # 0 degrees - horizontal
                neighbors = [gradient_magnitude[i, j-1], gradient_magnitude[i, j+1]]
            elif 22.5 <= theta < 67.5:
                # 45 degrees - diagonal
                neighbors = [gradient_magnitude[i+1, j-1], gradient_magnitude[i-1, j+1]]
            elif 67.5 <= theta < 112.5:
                # 90 degrees - vertical
                neighbors = [gradient_magnitude[i-1, j], gradient_magnitude[i+1, j]]
            else:
                # 135 degrees - diagonal
                neighbors = [gradient_magnitude[i-1, j-1], gradient_magnitude[i+1, j+1]]
            
            # Check if current pixel is local maximum
            if gradient_magnitude[i, j] >= max(neighbors):
                suppressed[i, j] = gradient_magnitude[i, j]
    
    return suppressed

def double_thresholding(suppressed, low_threshold=10, high_threshold=30):
    """Apply double thresholding to categorize strong and weak edges"""
    height, width = suppressed.shape
    result = np.zeros((height, width))
    
    # Strong edge pixels
    strong_edges = suppressed >= high_threshold
    # Weak edge pixels
    weak_edges = (suppressed >= low_threshold) & (suppressed < high_threshold)
    
    # Set values
    result[strong_edges] = 255  # Strong edges
    result[weak_edges] = 75     # Weak edges
    
    return result

def edge_tracking(thresholded):
    """Apply edge tracking by hysteresis to connect weak edges to strong edges"""
    height, width = thresholded.shape
    tracked = np.copy(thresholded)
    
    # Direction vectors (8-connectivity)
    dx = [-1, -1, 0, 1, 1, 1, 0, -1]
    dy = [0, 1, 1, 1, 0, -1, -1, -1]
    
    # Find all strong edge pixels
    strong_i, strong_j = np.where(tracked == 255)
    
    # Add all strong edge pixels to queue
    edges = [(i, j) for i, j in zip(strong_i, strong_j)]
    
    # Process queue
    while edges:
        i, j = edges.pop(0)
        
        # Check 8-connected neighbors
        for d in range(8):
            ni, nj = i + dx[d], j + dy[d]
            
            if (0 <= ni < height and 0 <= nj < width and 
                tracked[ni, nj] == 75):
                # Convert weak edge to strong edge
                tracked[ni, nj] = 255
                edges.append((ni, nj))
    
    # Remove remaining weak edges
    tracked[tracked != 255] = 0
    
    return tracked

def canny_edge_detection(image, low_threshold=10, high_threshold=30):
    """Apply Canny edge detection algorithm"""
    # Ensure image is grayscale
    if len(image.shape) == 3:
        gray_image = cs.rgb_to_grayscale(image)
    else:
        gray_image = image
    
    # Step 1: Noise reduction using Gaussian filter
    smoothed = pr.gaussian_filter(gray_image)
    
    # Step 2: Calculate gradient magnitude and direction
    gradient_magnitude, gradient_direction = sobel_operator(smoothed)
    
    # Step 3: Non-maximum suppression
    suppressed = non_maximum_suppression(gradient_magnitude, gradient_direction)
    
    # Step 4: Double thresholding
    thresholded = double_thresholding(suppressed, low_threshold, high_threshold)
    
    # Step 5: Edge tracking by hysteresis
    edges = edge_tracking(thresholded)
    
    return edges

