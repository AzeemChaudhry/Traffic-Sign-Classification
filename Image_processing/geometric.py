import numpy as np

def rotate_image(img, angle_degrees):
    angle = np.deg2rad(angle_degrees)
    h, w = img.shape[:2]
    center = (w // 2, h // 2)

    if img.ndim == 3:
        c = img.shape[2]
        rotated_img = np.zeros_like(img)
    else:
        rotated_img = np.zeros((h, w), dtype=img.dtype)

    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])

    for i in range(h):
        for j in range(w):
            x = j - center[0]
            y = i - center[1]
            new_pos = np.dot(rotation_matrix, [x, y])
            new_x = int(round(new_pos[0] + center[0]))
            new_y = int(round(new_pos[1] + center[1]))

            if 0 <= new_x < w and 0 <= new_y < h:
                rotated_img[new_y, new_x] = img[i, j]

    return rotated_img


def scale_image(img, target_size=(200, 200)):
    """
    Scale an image to the target size.
    Handles both grayscale and RGB images.
    """
    h, w = img.shape[:2]
    target_h, target_w = target_size
    scale_x = target_w / w
    scale_y = target_h / h

    if img.ndim == 3:
        c = img.shape[2]
        scaled_img = np.zeros((target_h, target_w, c), dtype=img.dtype)
    else:
        scaled_img = np.zeros((target_h, target_w), dtype=img.dtype)

    for i in range(target_h):
        for j in range(target_w):
            orig_x = int(j / scale_x)
            orig_y = int(i / scale_y)

            if orig_x < w and orig_y < h:
                scaled_img[i, j] = img[orig_y, orig_x]

    return scaled_img


def scale_image_by_factor(img, scale_x, scale_y):
    """
    Scale an image by the specified factors.
    
    Args:
        img: Input image as numpy array
        scale_x: Scaling factor for width
        scale_y: Scaling factor for height
        
    Returns:
        Scaled image as numpy array
    """
    # Get image dimensions
    h, w = img.shape[:2]
    
    # Calculate new dimensions
    new_h = int(h * scale_y)
    new_w = int(w * scale_x)
    
    # Call the main scaling function with the new dimensions
    return scale_image(img, (new_h, new_w))

def perspective_transform(img, transform_matrix):
    h, w = img.shape[:2]
    if img.ndim == 3:
        c = img.shape[2]
        new_img = np.zeros((h, w, c), dtype=img.dtype)
    else:
        new_img = np.zeros((h, w), dtype=img.dtype)

    try:
        inv_matrix = np.linalg.inv(transform_matrix)
    except np.linalg.LinAlgError:
        inv_matrix = transform_matrix

    for y in range(h):
        for x in range(w):
            src_coords = np.dot(inv_matrix, [x, y, 1])
            if src_coords[2] != 0:
                src_x, src_y = src_coords[0] / src_coords[2], src_coords[1] / src_coords[2]
                src_x_int, src_y_int = int(round(src_x)), int(round(src_y))
                if 0 <= src_x_int < w and 0 <= src_y_int < h:
                    new_img[y, x] = img[src_y_int, src_x_int]

    return new_img

def create_perspective_matrix(src_points, dst_points):
    """
    Create a perspective transformation matrix from source and destination points.
    
    Args:
        src_points: List of 4 source points [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
        dst_points: List of 4 destination points [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
        
    Returns:
        3x3 perspective transformation matrix
    """
    # Create matrices for the linear equation system
    A = np.zeros((8, 8))
    b = np.zeros(8)
    
    for i in range(4):
        src_x, src_y = src_points[i]
        dst_x, dst_y = dst_points[i]
        
        A[2*i, :] = [src_x, src_y, 1, 0, 0, 0, -src_x*dst_x, -src_y*dst_x]
        A[2*i+1, :] = [0, 0, 0, src_x, src_y, 1, -src_x*dst_y, -src_y*dst_y]
        
        b[2*i] = dst_x
        b[2*i+1] = dst_y
    
    # Solve the system
    x = np.linalg.solve(A, b)
    
    # Form the perspective transform matrix
    matrix = np.array([
        [x[0], x[1], x[2]],
        [x[3], x[4], x[5]],
        [x[6], x[7], 1.0]
    ])
    
    return matrix

def geometric_normalization(img, angle=0, target_size=(200, 200), apply_perspective=True):
    """
    Apply geometric normalization to an image:
    1. Rotate to upright orientation
    2. Scale to fixed size
    3. Optionally apply perspective transform
    
    Args:
        img: Input image as numpy array
        angle: Rotation angle in degrees
        target_size: Target size (height, width)
        apply_perspective: Whether to apply perspective transform
        
    Returns:
        Normalized image as numpy array
    """
    # Step 1: Rotate image to upright orientation
    rotated = rotate_image(img, angle)
    
    # Step 2: Scale to fixed size
    scaled = scale_image(rotated, target_size)
    
    # Step 3: Optionally apply perspective transform
    if apply_perspective:
        h, w = scaled.shape[:2]
        # Example perspective transformation (simulating slight viewpoint change)
        src_points = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32)
        dst_points = np.array([[w*0.05, h*0.05], [w*0.95, h*0.03], [w*0.9, h*0.9], [w*0.1, h*0.95]], dtype=np.float32)
        
        perspective_matrix = create_perspective_matrix(src_points, dst_points)
        result = perspective_transform(scaled, perspective_matrix)
        return result
    
    return scaled