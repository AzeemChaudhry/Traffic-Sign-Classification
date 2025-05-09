import numpy as np
import cv2
from PIL import Image
# Feature Extraction Functions

def extract_harris_corners(img, blockSize=2, ksize=3, k=0.04, threshold=0.01):
    """
    Extract corners using Harris Corner Detection manually with NumPy
    Args:
        img: Input image (grayscale)
        blockSize: Size of neighborhood considered for corner detection
        ksize: Aperture parameter for Sobel derivative
        k: Harris detector free parameter
        threshold: Threshold for detecting corners
    Returns:
        Number of corners detected and corner coordinates
    """
    if len(img.shape) == 3:
        # Convert BGR to grayscale manually
        gray = np.dot(img[..., :3], [0.114, 0.587, 0.299])
    else:
        gray = img.copy()
    
    # Convert to float32 for better precision
    gray = gray.astype(np.float32)
    
    # Manual implementation of Sobel operators for gradient calculation
    height, width = gray.shape
    
    # Create Sobel kernels
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    
    # Add padding to the image
    pad_size = ksize // 2
    padded = np.pad(gray, pad_size, mode='reflect')
    
    # Initialize gradient images
    Ix = np.zeros_like(gray)
    Iy = np.zeros_like(gray)
    
    # Apply Sobel operators manually
    for y in range(height):
        for x in range(width):
            # Extract local region
            region = padded[y:y+ksize, x:x+ksize]
            # Apply sobel kernels
            Ix[y, x] = np.sum(region * sobel_x)
            Iy[y, x] = np.sum(region * sobel_y)
    
    # Compute products of derivatives for the Harris matrix elements
    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy
    
    # Apply Gaussian window to the Harris matrix elements
    # First, create a Gaussian kernel manually
    def gaussian_kernel(ksize, sigma):
        x, y = np.mgrid[-ksize//2 + 1:ksize//2 + 1, -ksize//2 + 1:ksize//2 + 1]
        g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
        return g/g.sum()

    gaussian = gaussian_kernel(blockSize, 1.0)
    
    # Apply window by convolution (manual convolution)
    pad_size = blockSize // 2
    padded_Ixx = np.pad(Ixx, pad_size, mode='reflect')
    padded_Iyy = np.pad(Iyy, pad_size, mode='reflect')
    padded_Ixy = np.pad(Ixy, pad_size, mode='reflect')
    
    windowedIxx = np.zeros_like(Ixx)
    windowedIyy = np.zeros_like(Iyy)
    windowedIxy = np.zeros_like(Ixy)
    
    for y in range(height):
        for x in range(width):
            windowedIxx[y, x] = np.sum(padded_Ixx[y:y+blockSize, x:x+blockSize] * gaussian)
            windowedIyy[y, x] = np.sum(padded_Iyy[y:y+blockSize, x:x+blockSize] * gaussian)
            windowedIxy[y, x] = np.sum(padded_Ixy[y:y+blockSize, x:x+blockSize] * gaussian)
    
    # Calculate Harris response
    det = (windowedIxx * windowedIyy) - (windowedIxy**2)
    trace = windowedIxx + windowedIyy
    harris_response = det - k * (trace**2)
    
    # Perform non-maximum suppression (simplified)
    # First, apply threshold
    corner_threshold = threshold * harris_response.max()
    corners_mask = harris_response > corner_threshold
    
    # Dilate corners to get more robust detection (manual dilation)
    dilated_corners = np.zeros_like(corners_mask)
    for y in range(1, height-1):
        for x in range(1, width-1):
            # 3x3 neighborhood
            if np.any(corners_mask[y-1:y+2, x-1:x+2]):
                dilated_corners[y, x] = True
    
    # Find coordinates of corners
    corner_coords = np.where(dilated_corners)
    corner_count = len(corner_coords[0])
    
    return corner_count, corner_coords

def find_contours(binary_img):
    """
    Find contours in a binary image using NumPy
    Args:
        binary_img: Binary image (0 and 255)
    Returns:
        List of contours where each contour is a list of (x,y) coordinates
    """
    # Make sure the image is binary (0 and 255)
    binary = np.where(binary_img > 0, 1, 0).astype(np.uint8)
    
    # Find connected components
    from scipy.ndimage import label, find_objects
    labeled, num_features = label(binary)
    
    contours = []
    for i in range(1, num_features + 1):
        # Get the slice for this component
        obj_slice = find_objects(labeled == i)[0]
        y_slice, x_slice = obj_slice
        
        # Extract the component
        component = (labeled[obj_slice] == i).astype(np.uint8)
        
        # Find the boundary pixels
        padded = np.pad(component, 1, mode='constant')
        boundary = np.zeros_like(padded)
        
        # A pixel is on the boundary if it's 1 and has at least one 0 neighbor
        for y in range(1, padded.shape[0] - 1):
            for x in range(1, padded.shape[1] - 1):
                if padded[y, x] == 1:
                    neighbors = padded[y-1:y+2, x-1:x+2]
                    if np.any(neighbors == 0):
                        boundary[y, x] = 1
        
        # Get coordinates of boundary pixels and adjust for padding
        y_coords, x_coords = np.where(boundary == 1)
        y_coords = y_coords - 1 + y_slice.start
        x_coords = x_coords - 1 + x_slice.start
        
        # Convert to contour format [(x1,y1), (x2,y2), ...]
        contour = np.column_stack((x_coords, y_coords))
        contours.append(contour)
    
    return contours

def calculate_contour_area(contour):
    """
    Calculate area of a contour using Shoelace formula
    Args:
        contour: Numpy array of shape (n, 2) containing (x,y) coordinates
    Returns:
        Area of the contour
    """
    # Ensure contour is closed (first and last points are the same)
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack([contour, contour[0]])
    
    x = contour[:, 0]
    y = contour[:, 1]
    
    # Shoelace formula
    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    return area

def calculate_contour_perimeter(contour):
    """
    Calculate perimeter of a contour
    Args:
        contour: Numpy array of shape (n, 2) containing (x,y) coordinates
    Returns:
        Perimeter of the contour
    """
    # Ensure contour is closed (first and last points are the same)
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack([contour, contour[0]])
    
    # Calculate Euclidean distances between consecutive points
    diff = np.diff(contour, axis=0)
    distances = np.sqrt(np.sum(diff**2, axis=1))
    perimeter = np.sum(distances)
    
    return perimeter

def calculate_circularity(contour):
    """
    Calculate circularity: C = 4π×Area/(Perimeter)²
    Args:
        contour: Contour of the object as array of (x,y) coordinates
    Returns:
        Circularity value (1 for perfect circle, decreases as shape becomes less circular)
    """
    area = calculate_contour_area(contour)
    perimeter = calculate_contour_perimeter(contour)
    
    if perimeter == 0:  # Avoid division by zero
        return 0
    
    circularity = 4 * np.pi * area / (perimeter ** 2)
    return circularity

def calculate_bounding_rect(contour):
    """
    Calculate bounding rectangle of a contour
    Args:
        contour: Contour of the object as array of (x,y) coordinates
    Returns:
        x, y, width, height of the bounding rectangle
    """
    x_min = np.min(contour[:, 0])
    y_min = np.min(contour[:, 1])
    x_max = np.max(contour[:, 0])
    y_max = np.max(contour[:, 1])
    
    width = x_max - x_min + 1
    height = y_max - y_min + 1
    
    return x_min, y_min, width, height

def calculate_aspect_ratio(contour):
    """
    Calculate aspect ratio (width/height) of the bounding box
    Args:
        contour: Contour of the object as array of (x,y) coordinates
    Returns:
        Aspect ratio of the bounding box
    """
    _, _, w, h = calculate_bounding_rect(contour)
    
    if h == 0:  # Avoid division by zero
        return 0
    
    aspect_ratio = float(w) / h
    return aspect_ratio

def calculate_extent(contour):
    """
    Calculate extent: ratio of contour area to bounding rectangle area
    Args:
        contour: Contour of the object as array of (x,y) coordinates
    Returns:
        Extent ratio
    """
    area = calculate_contour_area(contour)
    _, _, w, h = calculate_bounding_rect(contour)
    rect_area = w * h
    
    if rect_area == 0:  # Avoid division by zero
        return 0
    
    extent = float(area) / rect_area
    return extent

def rgb_to_hsv(rgb):
    """
    Convert an RGB NumPy image to HSV with:
      - H in [0, 360)
      - S, V in [0, 1]
    Args:
      rgb: uint8 NumPy array of shape (H, W, 3), in RGB order
    Returns:
      float32 NumPy array of shape (H, W, 3): HSV
    """
    # 1) Create a PIL image and convert it to HSV mode
    pil_img = Image.fromarray(rgb, mode='RGB')
    hsv_pil = pil_img.convert('HSV')

    # 2) Split into H, S, V channels (each 8-bit 0–255)
    h_pil, s_pil, v_pil = hsv_pil.split()

    # 3) Turn each into a float array and rescale
    h = np.asarray(h_pil, dtype=np.float32) * (360.0 / 255.0)
    s = np.asarray(s_pil, dtype=np.float32) / 255.0
    v = np.asarray(v_pil, dtype=np.float32) / 255.0

    # 4) Stack back into (H, W, 3)
    return np.stack([h, s, v], axis=-1)

def calculate_average_hue(img):
    """
    Calculate average hue of the image
    Args:
        img: Input BGR image
    Returns:
        Average hue value (0-179)
    """
    # Convert BGR to RGB
    rgb = img[:, :, ::-1]
    
    # Convert to HSV
    hsv = rgb_to_hsv(rgb)
    
    # Extract hue channel
    hue = hsv[:, :, 0]
    
    # Calculate average hue (circular mean to handle the circular nature of hue)
    sin_sum = np.sum(np.sin(2 * np.pi * hue / 360))
    cos_sum = np.sum(np.cos(2 * np.pi * hue / 360))
    
    avg_angle = np.arctan2(sin_sum, cos_sum)
    avg_hue = int((avg_angle * 180 / np.pi) % 180)
    
    return avg_hue

def kmeans_clustering(data, k, max_iter=100, tol=1e-4):
    """
    K-means clustering implementation using NumPy
    Args:
        data: Input data points (N, D)
        k: Number of clusters
        max_iter: Maximum number of iterations
        tol: Convergence tolerance
    Returns:
        centers: cluster centers (k, D)
        labels: cluster assignments (N,)
    """
    n_samples, n_features = data.shape
    
    # Initialize cluster centers randomly
    idx = np.random.choice(n_samples, k, replace=False)
    centers = data[idx].copy()
    
    # Initialize labels
    labels = np.zeros(n_samples, dtype=np.int32)
    
    # K-means loop
    for _ in range(max_iter):
        # Compute distances to centers
        distances = np.zeros((n_samples, k))
        for i in range(k):
            distances[:, i] = np.sqrt(np.sum((data - centers[i])**2, axis=1))
        
        # Assign points to nearest center
        new_labels = np.argmin(distances, axis=1)
        
        # Check for convergence
        if np.all(new_labels == labels):
            break
        
        labels = new_labels
        
        # Update centers
        for i in range(k):
            mask = labels == i
            if np.any(mask):
                centers[i] = np.mean(data[mask], axis=0)
    
    return centers, labels

def get_dominant_colors(img, k=3):
    """
    Get dominant colors in the image using k-means clustering
    Args:
        img: Input BGR image
        k: Number of color clusters
    Returns:
        List of dominant colors (BGR) and their percentages
    """
    # Reshape the image to be a list of pixels
    pixels = img.reshape(-1, 3).astype(np.float32)
    
    # Apply k-means clustering
    # identify and quantify the most representative colors in an image
    centers, labels = kmeans_clustering(pixels, k)
    
    # Count labels to find most popular
    counts = np.bincount(labels, minlength=k)
    
    # Convert to percentages
    percentages = counts / len(labels) * 100
    
    # Sort by percentage (descending)
    sorted_indices = np.argsort(percentages)[::-1]
    centers = centers[sorted_indices]
    percentages = percentages[sorted_indices]
    
    return centers, percentages

def bgr_to_gray(img):
    return np.dot(img[..., :3], [0.114, 0.587, 0.299])

def gray_to_binary(gray, method='otsu'):
    """
    Convert grayscale to binary image using NumPy
    Args:
        gray: Grayscale image
        method: Thresholding method ('otsu' or threshold value)
    Returns:
        Binary image (0 and 255)
    """
    if method == 'otsu':
        # Implement Otsu's method to find optimal threshold
        hist, bin_edges = np.histogram(gray.flatten(), 256, [0, 256])
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Calculate histogram
        hist = hist.astype(float)
        hist_norm = hist / hist.sum()
        
        # Calculate cumulative sums
        weight1 = np.cumsum(hist_norm)
        weight2 = 1 - weight1
        
        # Calculate cumulative means
        mean1 = np.cumsum(hist_norm * bin_centers) / (weight1 + 1e-10)
        mean2 = (np.cumsum(hist_norm * bin_centers)[-1] - np.cumsum(hist_norm * bin_centers)) / (weight2 + 1e-10)
        
        # Calculate inter-class variance
        variance = weight1 * weight2 * (mean1 - mean2)**2
        
        # Find the threshold that maximizes the variance
        idx = np.argmax(variance)
        threshold = bin_centers[idx]
    else:
        threshold = method
    
    # Apply threshold
    binary = np.zeros_like(gray)
    binary[gray < threshold] = 255  # Inverse binary for THRESH_BINARY_INV
    
    return binary

def extract_features(img):
    """
    Extract all features from an image
    Args:
        img: Input BGR image
    Returns:
        Dictionary of features
    """
    # Make a copy to avoid modifying the original
    image = img.copy()
    
    # Convert to grayscale for contour detection
    if len(image.shape) == 3:
        gray = bgr_to_gray(image)
    else:
        gray = image.copy()
        # Create a 3-channel BGR image for color analysis
        image = np.stack([gray, gray, gray], axis=2)
    
    # Apply threshold to get binary image
    binary = gray_to_binary(gray, method='otsu')
    
    # Find contours
    contours = find_contours(binary)
    
    # If no contours found, return default values
    if not contours:
        return {
            'corner_count': 0,
            'circularity': 0,
            'aspect_ratio': 1,
            'extent': 0,
            'avg_hue': 0,
            'dominant_colors': [],
            'dominant_percentages': []
        }
    
    # Get the largest contour (assuming it's the road sign)
    largest_contour = max(contours, key=calculate_contour_area)
    
    # Extract features
    corner_count, _ = extract_harris_corners(gray)
    circularity = calculate_circularity(largest_contour)
    aspect_ratio = calculate_aspect_ratio(largest_contour)
    extent = calculate_extent(largest_contour)
    avg_hue = calculate_average_hue(image)
    dominant_colors, dominant_percentages = get_dominant_colors(image)
    
    # Create and return features dictionary
    features = {
        'corner_count': corner_count,
        'circularity': circularity,
        'aspect_ratio': aspect_ratio,
        'extent': extent,
        'avg_hue': avg_hue,
        'dominant_colors': dominant_colors,
        'dominant_percentages': dominant_percentages
    }
    
    return features