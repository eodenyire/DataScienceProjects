"""
Pan Card Tampering Detector using OpenCV
This module provides functionality to detect tampering in Pan Card images by comparing them with an original image.
"""

import cv2
from skimage.metrics import structural_similarity as ssim
import imutils


def detect_tampering(original_image_path, user_image_path):
    """
    Detect tampering between original and user-uploaded Pan Card images.
    
    Args:
        original_image_path (str): Path to the original Pan Card image
        user_image_path (str): Path to the user-uploaded Pan Card image
    
    Returns:
        tuple: (similarity_score, difference_image, threshold_image, contours_image)
    """
    # Load the two input images
    original = cv2.imread(original_image_path)
    user_image = cv2.imread(user_image_path)
    
    if original is None or user_image is None:
        raise ValueError("Could not load one or both images")
    
    # Resize user image to match original dimensions
    user_image = cv2.resize(user_image, (original.shape[1], original.shape[0]))
    
    # Convert images to grayscale
    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    user_gray = cv2.cvtColor(user_image, cv2.COLOR_BGR2GRAY)
    
    # Compute Structural Similarity Index (SSIM)
    similarity_score, diff = ssim(original_gray, user_gray, full=True)
    diff = (diff * 255).astype("uint8")
    
    # Threshold the difference image
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    # Find contours on thresholded image
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    # Create a copy of the user image to draw contours
    contours_image = user_image.copy()
    
    # Draw rectangles around differences
    for c in cnts:
        # Compute the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(c)
        # Draw rectangle on the contours image
        cv2.rectangle(contours_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    
    return similarity_score, diff, thresh, contours_image


def compare_images(original_path, user_path):
    """
    Compare two images and return the similarity score.
    
    Args:
        original_path (str): Path to the original image
        user_path (str): Path to the user image
    
    Returns:
        float: Similarity score between 0 and 1 (1 = identical)
    """
    try:
        similarity_score, _, _, _ = detect_tampering(original_path, user_path)
        return similarity_score
    except Exception as e:
        print(f"Error comparing images: {str(e)}")
        return 0.0


def is_tampered(original_path, user_path, threshold=0.95):
    """
    Check if an image has been tampered with based on similarity threshold.
    
    Args:
        original_path (str): Path to the original image
        user_path (str): Path to the user image
        threshold (float): Similarity threshold (default: 0.95)
    
    Returns:
        bool: True if tampered (similarity < threshold), False otherwise
    """
    similarity = compare_images(original_path, user_path)
    return similarity < threshold
