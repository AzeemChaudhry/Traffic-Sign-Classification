import numpy as np

def rotate_image(img, angle_degrees):
    # Convert the angle to radians
    angle = np.deg2rad(angle_degrees)
    
    # Get image dimensions
    h, w = img.shape
    center = (w // 2, h // 2)  # Center of the image

    # Rotation matrix (2D)
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])

    # Create a new blank image for rotated output
    rotated_img = np.zeros_like(img)

    for i in range(h):
        for j in range(w):
            # Calculate new position of pixel (i, j)
            x = j - center[0]
            y = i - center[1]
            new_pos = np.dot(rotation_matrix, [x, y])
            new_x = int(round(new_pos[0] + center[0]))
            new_y = int(round(new_pos[1] + center[1]))

            # Assign the pixel value if within bounds
            if 0 <= new_x < w and 0 <= new_y < h:
                rotated_img[new_y, new_x] = img[i, j]

    return rotated_img

def scale_image(img, target_size=(200, 200)):
    # Get image dimensions
    h, w = img.shape
    target_h, target_w = target_size

    # Calculate scaling factors
    scale_x = target_w / w
    scale_y = target_h / h

    # Create a new blank image for the scaled output
    scaled_img = np.zeros((target_h, target_w), dtype=img.dtype)

    for i in range(target_h):
        for j in range(target_w):
            # Find corresponding pixel in the original image
            orig_x = int(j / scale_x)
            orig_y = int(i / scale_y)
            
            # Assign the pixel value if within bounds
            if orig_x < w and orig_y < h:
                scaled_img[i, j] = img[orig_y, orig_x]

    return scaled_img

def perspective_transform(img, transform_matrix):
    h, w = img.shape
    new_img = np.zeros_like(img)

    for i in range(h):
        for j in range(w):
            # Apply the perspective transformation matrix
            xy = np.array([j, i, 1])  # The homogeneous coordinates
            transformed_xy = np.dot(transform_matrix, xy)
            transformed_xy /= transformed_xy[2]  # Normalize by the third coordinate

            new_x, new_y = int(round(transformed_xy[0])), int(round(transformed_xy[1]))

            # Assign the pixel value if within bounds
            if 0 <= new_x < w and 0 <= new_y < h:
                new_img[new_y, new_x] = img[i, j]

    return new_img

