import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os
import math

def rgb_to_grayscale(rgb_image):
    """Convert RGB image to grayscale using standard weights"""
    if len(rgb_image.shape) == 3 and rgb_image.shape[2] == 3:
        # Standard luminance conversion formula
        return 0.299 * rgb_image[:,:,0] + 0.587 * rgb_image[:,:,1] + 0.114 * rgb_image[:,:,2]
    else:
        return rgb_image  # Already grayscale

def rgb_to_hsv(rgb_image):
    """Manual conversion from RGB to HSV using NumPy"""
    # Normalize RGB values to [0, 1]
    rgb_normalized = rgb_image.astype(np.float32) / 255.0
    
    r, g, b = rgb_normalized[:,:,0], rgb_normalized[:,:,1], rgb_normalized[:,:,2]
    
    # Calculate Value (V)
    v = np.max(rgb_normalized, axis=2)
    
    # Calculate Saturation (S)
    min_rgb = np.min(rgb_normalized, axis=2)
    delta = v - min_rgb
    
    # Initialize saturation array
    s = np.zeros_like(v)
    # Avoid division by zero
    non_zero_v = v > 0
    s[non_zero_v] = delta[non_zero_v] / v[non_zero_v]
    
    # Calculate Hue (H)
    h = np.zeros_like(v)
    
    # When delta is 0, hue is undefined (set to 0)
    non_zero_delta = delta > 0
    
    # Red is max
    mask_r = (v == r) & non_zero_delta
    h[mask_r] = ((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6
    
    # Green is max
    mask_g = (v == g) & non_zero_delta
    h[mask_g] = 2.0 + (b[mask_g] - r[mask_g]) / delta[mask_g]
    
    # Blue is max
    mask_b = (v == b) & non_zero_delta
    h[mask_b] = 4.0 + (r[mask_b] - g[mask_b]) / delta[mask_b]
    
    # Convert hue to degrees
    h = h * 60.0
    # Handle negative hues
    h[h < 0] += 360
    
    # Scale hue to [0, 180], saturation and value to [0, 255] for OpenCV compatibility
    h = h / 2.0
    s = s * 255.0
    v = v * 255.0
    
    # Stack channels and return
    hsv_image = np.stack([h, s, v], axis=2).astype(np.uint8)
    return hsv_image

def segment_red(hsv_image):
    """Segment red areas in HSV image"""
    h, s, v = hsv_image[:, :, 0], hsv_image[:, :, 1], hsv_image[:, :, 2]
    red_mask1 = (h >= 0) & (h <= 15) & (s >= 100) & (v >= 80)
    red_mask2 = (h >= 165) & (h <= 180) & (s >= 100) & (v >= 80)
    red_mask = red_mask1 | red_mask2
    return (red_mask.astype(np.uint8) * 255)

def segment_blue(hsv_image):
    """Segment blue areas in HSV image"""
    h, s, v = hsv_image[:, :, 0], hsv_image[:, :, 1], hsv_image[:, :, 2]
    blue_mask = (h >= 100) & (h <= 130) & (s >= 100) & (v >= 80)
    return (blue_mask.astype(np.uint8) * 255)

def segment_traffic_sign(hsv_image):
    """Segment traffic signs based on HSV ranges"""
    # Extract HSV channels
    red_mask = segment_red(hsv_image)
    blue_mask = segment_blue(hsv_image)
    
    # Combine masks
    combined_mask = red_mask | blue_mask
    
    # Convert to binary mask
    binary_mask = combined_mask.astype(np.uint8)
    
    return binary_mask

def morphological_operations(binary_mask, operation='opening', kernel_size=5):
    """Apply morphological operations to binary mask"""
    # Create kernel
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    if operation == 'erosion':
        # Erosion
        result = np.zeros_like(binary_mask)
        padded = np.pad(binary_mask, kernel_size//2, mode='constant', constant_values=0)
        for i in range(binary_mask.shape[0]):
            for j in range(binary_mask.shape[1]):
                if np.all(padded[i:i+kernel_size, j:j+kernel_size] == 255):
                    result[i, j] = 255
        return result
        
    elif operation == 'dilation':
        # Dilation
        result = np.zeros_like(binary_mask)
        padded = np.pad(binary_mask, kernel_size//2, mode='constant', constant_values=0)
        for i in range(binary_mask.shape[0]):
            for j in range(binary_mask.shape[1]):
                if np.any(padded[i:i+kernel_size, j:j+kernel_size] == 255):
                    result[i, j] = 255
        return result
        
    elif operation == 'opening':
        # Opening (erosion followed by dilation)
        eroded = morphological_operations(binary_mask, 'erosion', kernel_size)
        return morphological_operations(eroded, 'dilation', kernel_size)
        
    elif operation == 'closing':
        # Closing (dilation followed by erosion)
        dilated = morphological_operations(binary_mask, 'dilation', kernel_size)
        return morphological_operations(dilated, 'erosion', kernel_size)
        
    else:
        return binary_mask

def connected_component_filtering(binary_mask, min_area=100):
    """Remove small connected components from binary mask"""
    # Label connected components
    label_count = 0
    height, width = binary_mask.shape
    labels = np.zeros((height, width), dtype=np.int32)
    
    # Direction vectors (8-connectivity)
    dx = [-1, -1, 0, 1, 1, 1, 0, -1]
    dy = [0, 1, 1, 1, 0, -1, -1, -1]
    
    # 8-connectivity labeling algorithm
    for i in range(height):
        for j in range(width):
            if binary_mask[i, j] == 255 and labels[i, j] == 0:
                label_count += 1
                queue = [(i, j)]
                labels[i, j] = label_count
                
                while queue:
                    x, y = queue.pop(0)
                    
                    for d in range(8):
                        nx, ny = x + dx[d], y + dy[d]
                        if (0 <= nx < height and 0 <= ny < width and 
                            binary_mask[nx, ny] == 255 and labels[nx, ny] == 0):
                            labels[nx, ny] = label_count
                            queue.append((nx, ny))
    
    # Calculate area of each component
    areas = {}
    for i in range(1, label_count + 1):
        areas[i] = np.sum(labels == i)
    
    # Filter by area
    filtered_mask = np.zeros_like(binary_mask)
    for i in range(1, label_count + 1):
        if areas[i] >= min_area:
            filtered_mask[labels == i] = 255
            
    return filtered_mask

def fill_holes(binary_mask):
    """Fill holes in binary mask"""
    # Invert mask
    inverted = 255 - binary_mask
    
    # Label background components
    height, width = inverted.shape
    labels = np.zeros((height, width), dtype=np.int32)
    
    # Direction vectors (4-connectivity)
    dx = [0, 1, 0, -1]
    dy = [1, 0, -1, 0]
    
    # Label the outer background component
    queue = [(0, 0)]
    labels[0, 0] = 1
    
    while queue:
        x, y = queue.pop(0)
        
        for d in range(4):
            nx, ny = x + dx[d], y + dy[d]
            if (0 <= nx < height and 0 <= ny < width and 
                inverted[nx, ny] == 255 and labels[nx, ny] == 0):
                labels[nx, ny] = 1
                queue.append((nx, ny))
    
    # Fill holes by inverting the background
    filled_mask = np.copy(binary_mask)
    filled_mask[labels == 0] = 255
    
    return filled_mask

# Make all functions available when imported
__all__ = [
    'rgb_to_grayscale',
    'rgb_to_hsv',
    'segment_red',
    'segment_blue',
    'segment_traffic_sign',
    'morphological_operations',
    'connected_component_filtering',
    'fill_holes'
]