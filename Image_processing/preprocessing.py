import os
import numpy as np 
import cv2

filtered_data_dir = r'Data\filtered_data.csv' ## path to the filtered data csv file

## reading the csv file and extracting the images of each respective class 

## mean filter 
def mean_filter(image, kernel_size=3):
        """Apply mean filter to image using NumPy"""
        # Get image dimensions
        height, width = image.shape[:2]
        
        # Padding size
        pad = kernel_size // 2
        
        # Create padded image
        if len(image.shape) == 3:  # RGB image
            padded_image = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
            filtered_image = np.zeros_like(image)
            
            # Apply filter to each channel
            for c in range(image.shape[2]):
                for i in range(height):
                    for j in range(width):
                        filtered_image[i, j, c] = np.mean(padded_image[i:i+kernel_size, j:j+kernel_size, c])
                        
        else:  # Grayscale image
            padded_image = np.pad(image, ((pad, pad), (pad, pad)), mode='reflect')
            filtered_image = np.zeros_like(image)
            
            for i in range(height):
                for j in range(width):
                    filtered_image[i, j] = np.mean(padded_image[i:i+kernel_size, j:j+kernel_size])
        
        return filtered_image


## gaussian filter
def gaussian_filter(image, kernel_size=3, sigma=1.0):
    """Apply Gaussian filter to image using NumPy"""
    # Create Gaussian kernel
    k = kernel_size // 2
    x, y = np.mgrid[-k:k+1, -k:k+1]
    gaussian_kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
    
    # Get image dimensions
    height, width = image.shape[:2]
    
    # Padding size
    pad = kernel_size // 2
    
    # Create padded image
    if len(image.shape) == 3:  # RGB image
        padded_image = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
        filtered_image = np.zeros_like(image)
        
        # Apply filter to each channel
        for c in range(image.shape[2]):
            for i in range(height):
                for j in range(width):
                    window = padded_image[i:i+kernel_size, j:j+kernel_size, c]
                    filtered_image[i, j, c] = np.sum(window * gaussian_kernel)
                    
    else:  # Grayscale image
        padded_image = np.pad(image, ((pad, pad), (pad, pad)), mode='reflect')
        filtered_image = np.zeros_like(image)
        
        for i in range(height):
            for j in range(width):
                window = padded_image[i:i+kernel_size, j:j+kernel_size]
                filtered_image[i, j] = np.sum(window * gaussian_kernel)
    
    return filtered_image


## median filter
def median_filter(image, kernel_size=3):
        """Apply median filter to image using NumPy"""
        # Get image dimensions
        height, width = image.shape[:2]
        
        # Padding size
        pad = kernel_size // 2
        
        # Create padded image
        if len(image.shape) == 3:  # RGB image
            padded_image = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
            filtered_image = np.zeros_like(image)
            
            # Apply filter to each channel
            for c in range(image.shape[2]):
                for i in range(height):
                    for j in range(width):
                        filtered_image[i, j, c] = np.median(padded_image[i:i+kernel_size, j:j+kernel_size, c])
                        
        else:  # Grayscale image
            padded_image = np.pad(image, ((pad, pad), (pad, pad)), mode='reflect')
            filtered_image = np.zeros_like(image)
            
            for i in range(height):
                for j in range(width):
                    filtered_image[i, j] = np.median(padded_image[i:i+kernel_size, j:j+kernel_size])
        
        return filtered_image


##adaptive filter
def adaptive_median_filter(img, max_ksize=7):
    if len(img.shape) == 3:  # Convert to grayscale if it's a color image
        img = np.mean(img, axis=-1).astype(np.uint8)

    h, w = img.shape
    filtered_img = np.zeros_like(img)

    # Apply the adaptive median filter
    padded_img = np.pad(img, max_ksize // 2, mode='edge')

    for i in range(h):
        for j in range(w):
            ksize = 3
            while ksize <= max_ksize:
                r = ksize // 2
                region = padded_img[i:i+ksize, j:j+ksize]
                z_med = np.median(region)
                z_min = np.min(region)
                z_max = np.max(region)
                z_xy = img[i, j]

                if z_min < z_med < z_max:
                    if z_min < z_xy < z_max:
                        filtered_img[i, j] = z_xy
                    else:
                        filtered_img[i, j] = z_med
                    break
                else:
                    ksize += 2
            if ksize > max_ksize:
                filtered_img[i, j] = z_med

    return filtered_img


## Unsharp masking 
def unsharp_mask(img, ksize=5, sigma=1.0, alpha=1.5):
    blurred = gaussian_filter(img, ksize, sigma)
    sharpened = img + alpha * (img - blurred)
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)  # Ensure values are within 0-255
    return sharpened


##


#np.save('images_by_label.npy', images) # save the original compiled images to a numpy file

