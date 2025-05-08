import os
import numpy as np 
import cv2

filtered_data_dir = r'Data\filtered_data.csv' ## path to the filtered data csv file

## reading the csv file and extracting the images of each respective class 

def read_csv(file_path, test_dir):
    data = np.genfromtxt(file_path, delimiter=',', dtype=None, encoding=None)

    # Dictionary to store images grouped by label
    images_by_label = {}

    for row in data[1:]:  # skip header
        label = row[0]
        image_path = row[1]
        full_image_path = os.path.join(test_dir, image_path)
        
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
    return cv2.filter2D(img, -1, kernel)

## gaussian filter
def gaussian_filter(img, sigma=1.0, ksize=3):
    k = cv2.getGaussianKernel(ksize, sigma)
    kernel = k @ k.T  # 2D Gaussian
    return cv2.filter2D(img, -1, kernel)

## median filter
def median_filter(img, ksize=3):
    return cv2.medianBlur(img, ksize)

##adaptive filter
def adaptive_median_filter(img, max_ksize=7):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    padded_img = np.pad(img, max_ksize // 2, mode='edge')
    filtered = np.zeros_like(img)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
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
                        filtered[i, j] = z_xy
                    else:
                        filtered[i, j] = z_med
                    break
                else:
                    ksize += 2
            if ksize > max_ksize:
                filtered[i, j] = z_med
    return filtered

## Unsharp masking 
def unsharp_mask(img, ksize=5, sigma=1.0, alpha=1.5):
    blurred = gaussian_filter(img, sigma, ksize)
    sharpened = cv2.addWeighted(img, 1 + alpha, blurred, -alpha, 0)
    return sharpened

##
images = read_csv(filtered_data_dir,'Data/')
np.save('images_by_label.npy', images) # save the original compiledd images to a numpy file

