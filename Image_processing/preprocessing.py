import os
import numpy as np 
import cv2

filtered_data_dir = r'Data\filtered_data.csv' ## path to the filtered data csv file

## reading the csv file and extracting the images of each respective class 

def read_csv(file_path, base_dir=''):
    data = np.genfromtxt(file_path, delimiter=',', dtype=None, encoding='utf-8-sig')

    # Dictionary to store images grouped by label
    images_by_label = {}

    for row in data[1:]:  # Skip header
        label = row[0]
        image_path = os.path.normpath(row[1])  # Normalize the path
        full_image_path = os.path.normpath(os.path.join(base_dir, image_path))
        print(full_image_path, os.path.exists(full_image_path))
        if label not in images_by_label:
            images_by_label[label] = []

        images_by_label[label].append({
            "path": image_path,
            "image": cv2.imread(full_image_path)
        })
    return images_by_label
## mean filter 
def mean_filter(img):
    kernel = np.ones((3, 3), dtype=np.float32) / 9
    h, w = img.shape
    filtered_img = np.zeros_like(img)

    # Apply the kernel (filter) over the image with padding
    padded_img = np.pad(img, 1, mode='constant', constant_values=0)

    for i in range(h):
        for j in range(w):
            region = padded_img[i:i+3, j:j+3]
            filtered_img[i, j] = np.sum(region * kernel)

    return filtered_img


## gaussian filter
def gaussian_filter(img, sigma=1.0, ksize=3):
    # Generate the Gaussian kernel
    kernel_range = np.linspace(-ksize//2, ksize//2, ksize)
    x, y = np.meshgrid(kernel_range, kernel_range)
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel /= np.sum(kernel)  # Normalize the kernel
    
    h, w = img.shape
    filtered_img = np.zeros_like(img)

    # Apply the kernel (filter) over the image with padding
    padded_img = np.pad(img, ksize//2, mode='constant', constant_values=0)

    for i in range(h):
        for j in range(w):
            region = padded_img[i:i+ksize, j:j+ksize]
            filtered_img[i, j] = np.sum(region * kernel)

    return filtered_img


## median filter
def median_filter(img, ksize=3):
    h, w = img.shape
    filtered_img = np.zeros_like(img)

    # Apply the median filter over the image with padding
    padded_img = np.pad(img, ksize//2, mode='constant', constant_values=0)

    for i in range(h):
        for j in range(w):
            region = padded_img[i:i+ksize, j:j+ksize]
            filtered_img[i, j] = np.median(region)

    return filtered_img


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
    blurred = gaussian_filter(img, sigma, ksize)
    sharpened = img + alpha * (img - blurred)
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)  # Ensure values are within 0-255
    return sharpened


##

images = read_csv(filtered_data_dir)
print(images)
#np.save('images_by_label.npy', images) # save the original compiled images to a numpy file

