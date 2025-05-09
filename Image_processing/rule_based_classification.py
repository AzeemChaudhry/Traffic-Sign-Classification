import numpy as np
from feature_extraction import extract_features

# Color helper functions for rule-based classification
def is_red_dominant(dominant_colors, dominant_percentages, threshold=30):
    """Check if red is dominant in the image"""
    for color, percentage in zip(dominant_colors, dominant_percentages):
        b, g, r = color
        if r > 1.5*g and r > 1.5*b and percentage > threshold:
            return True
    return False

def is_blue_dominant(dominant_colors, dominant_percentages, threshold=30):
    """Check if blue is dominant in the image"""
    for color, percentage in zip(dominant_colors, dominant_percentages):
        b, g, r = color
        if b > 1.5*g and b > 1.5*r and percentage > threshold:
            return True
    return False

def is_yellow_dominant(dominant_colors, dominant_percentages, threshold=30):
    """Check if yellow is dominant in the image"""
    for color, percentage in zip(dominant_colors, dominant_percentages):
        b, g, r = color
        if r > 1.5*b and g > 1.5*b and r*0.7 < g < r*1.3 and percentage > threshold:
            return True
    return False

def is_white_dominant(dominant_colors, dominant_percentages, threshold=30):
    """Check if white is dominant in the image"""
    for color, percentage in zip(dominant_colors, dominant_percentages):
        b, g, r = color
        if b > 150 and g > 150 and r > 150 and percentage > threshold:
            return True
    return False

def is_black_dominant(dominant_colors, dominant_percentages, threshold=30):
    """Check if black is dominant in the image"""
    for color, percentage in zip(dominant_colors, dominant_percentages):
        b, g, r = color
        if b < 70 and g < 70 and r < 70 and percentage > threshold:
            return True
    return False

# Rule-Based Classification
def classify_road_sign(features):
    """
    Rule-based classifier for road signs
    Args:
        features: Dictionary of features
    Returns:
        ClassId and name of the predicted road sign
    """
    corner_count = features['corner_count']
    circularity = features['circularity']
    aspect_ratio = features['aspect_ratio']
    extent = features['extent']
    avg_hue = features['avg_hue']
    dominant_colors = features['dominant_colors']
    dominant_percentages = features['dominant_percentages']
    
    # Stop Sign (red, octagonal)
    if (is_red_dominant(dominant_colors, dominant_percentages) and 
        7 <= corner_count <= 15 and 
        0.85 <= aspect_ratio <= 1.15 and 
        circularity > 0.7 and 
        extent > 0.7):
        return 1, "Stop Sign"
    
    # Yield Sign (red border, white/yellow triangular)
    elif (is_red_dominant(dominant_colors, dominant_percentages) and 
          3 <= corner_count <= 8 and 
          0.8 <= aspect_ratio <= 1.2 and 
          circularity < 0.7 and 
          extent < 0.6):
        return 2, "Yield Sign"
    
    # Speed Limit Sign (white circular with black numbers)
    elif (is_white_dominant(dominant_colors, dominant_percentages) and 
          corner_count >= 15 and  
          0.9 <= aspect_ratio <= 1.1 and 
          circularity > 0.8):
        return 3, "Speed Limit Sign"
    
    # No Entry/Do Not Enter (red circle with white horizontal bar)
    elif (is_red_dominant(dominant_colors, dominant_percentages) and 
          corner_count <= 10 and 
          0.9 <= aspect_ratio <= 1.1 and 
          circularity > 0.8 and 
          not (7 <= corner_count <= 15 and extent > 0.7)):  # Differentiate from Stop sign
        return 4, "No Entry Sign"
    
    # Right Turn Only (blue circle with white arrow)
    elif (is_blue_dominant(dominant_colors, dominant_percentages) and 
          corner_count <= 15 and  
          0.9 <= aspect_ratio <= 1.1 and 
          circularity > 0.7):
        return 5, "Right Turn Only Sign"
    
    # Left Turn Only (blue circle with white arrow)
    elif (is_blue_dominant(dominant_colors, dominant_percentages) and 
          corner_count > 15 and  
          0.9 <= aspect_ratio <= 1.1 and 
          circularity > 0.7):
        return 6, "Left Turn Only Sign"
    
    # Warning Sign (yellow diamond)
    elif (is_yellow_dominant(dominant_colors, dominant_percentages) and 
          4 <= corner_count <= 10 and 
          0.9 <= aspect_ratio <= 1.1 and 
          circularity < 0.7):
        return 7, "Warning Sign"
    
    # No Parking Sign (blue circle with red cross over a 'P')
    elif (is_blue_dominant(dominant_colors, dominant_percentages) and 
          is_red_dominant(dominant_colors, dominant_percentages, threshold=15) and  # Lower threshold for secondary color
          corner_count >= 10):
        return 8, "No Parking Sign"
    
    # Default: Unknown
    return 0, "Unknown Sign"

# Process and classify image
def process_and_classify_image(img):
    """
    Process an image and classify the road sign
    Args:
        img: Input BGR image
    Returns:
        ClassId, class name, and extracted features
    """
    # Extract features
    features = extract_features(img)
#     Why Z-Score is usually “best” for image‐classification features
    # 1)Your features (corner counts, circularity, hue values, color percentages, etc.) each live on very different scales.
    # 2)Standardization ensures each contributes comparably.
    # 3)Empirically it speeds up convergence and often boosts accuracy.

    # Feature normalization BONUS
    keys = ['corner_count','circularity','aspect_ratio','extent','avg_hue']
    X = np.array([[d[k] for k in keys] for d in features], dtype=np.float32)

    # Standardize
    means = X.mean(axis=0)
    stds  = X.std(axis=0)
    X_scaled = (X - means) / (stds + 1e-8)
    
    # Classify based on features
    class_id, class_name = classify_road_sign(X_scaled)
    
    return class_id, class_name, X_scaled

# Function to test the classifier with sample images
def test_classifier(images_by_label):
    """
    Test the classifier with sample images
    Args:
        images_by_label: Dictionary of images grouped by label
    Returns:
        Dictionary of accuracy results
    """
    total_images = 0
    correct_predictions = 0
    
    class_accuracies = {}
    
    for true_label, images in images_by_label.items():
        class_total = len(images)
        class_correct = 0
        
        for img_data in images:
            img = img_data["image"]
            predicted_label, _, _ = process_and_classify_image(img)
            
            if int(predicted_label) == int(true_label):
                correct_predictions += 1
                class_correct += 1
            
            total_images += 1
        
        class_accuracies[true_label] = {
            'total': class_total,
            'correct': class_correct,
            'accuracy': class_correct / class_total if class_total > 0 else 0
        }
    
    overall_accuracy = correct_predictions / total_images if total_images > 0 else 0
    
    return {
        'overall_accuracy': overall_accuracy,
        'class_accuracies': class_accuracies
    }

# Example of how to use these functions
if __name__ == "__main__":
    # Load images if they've been saved
    try:
        images_by_label = np.load('images_by_label.npy', allow_pickle=True).item()
        
        # Test with a single image as an example
        if images_by_label:
            first_label = list(images_by_label.keys())[0]
            sample_img = images_by_label[first_label][0]["image"]
            
            # Display sample image
            # cv2.imshow("Sample Sign", sample_img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            
            # Extract features and classify
            class_id, class_name, features = process_and_classify_image(sample_img)
            
            print(f"Predicted: {class_name} (Class ID: {class_id})")
            print("Features:")
            for feature, value in features.items():
                if feature not in ['dominant_colors', 'dominant_percentages']:
                    print(f"  {feature}: {value}")
            
            # Run test on all images
            accuracy_results = test_classifier(images_by_label)
            print(f"Overall Accuracy: {accuracy_results['overall_accuracy']:.2f}")
            
    except FileNotFoundError:
        print("No saved images found. Please run the preprocessing code first.")