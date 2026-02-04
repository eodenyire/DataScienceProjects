"""
Example usage of the Pan Card Tampering Detector
This script demonstrates how to use the detector programmatically
"""

from pan_card_detector import detect_tampering, compare_images, is_tampered
import cv2

def example_basic_usage():
    """Basic usage example"""
    print("=== Basic Usage Example ===\n")
    
    # Compare two images
    original_path = 'image_data/original_pan_card.jpg'
    test_path = 'image_data/original_pan_card.jpg'  # Replace with your test image
    
    # Get similarity score
    similarity = compare_images(original_path, test_path)
    print(f"Similarity Score: {similarity * 100:.2f}%")
    
    # Check if tampered
    tampered = is_tampered(original_path, test_path, threshold=0.95)
    print(f"Is Tampered: {tampered}")
    
    print()


def example_detailed_analysis():
    """Detailed analysis example with image outputs"""
    print("=== Detailed Analysis Example ===\n")
    
    original_path = 'image_data/original_pan_card.jpg'
    test_path = 'image_data/original_pan_card.jpg'  # Replace with your test image
    
    # Get full detection results
    similarity, diff, thresh, contours = detect_tampering(original_path, test_path)
    
    print(f"Similarity: {similarity * 100:.2f}%")
    print(f"Difference Image Shape: {diff.shape}")
    print(f"Threshold Image Shape: {thresh.shape}")
    print(f"Contours Image Shape: {contours.shape}")
    
    # Save the analysis images
    cv2.imwrite('output_difference.jpg', diff)
    cv2.imwrite('output_threshold.jpg', thresh)
    cv2.imwrite('output_contours.jpg', contours)
    
    print("\nAnalysis images saved:")
    print("  - output_difference.jpg")
    print("  - output_threshold.jpg")
    print("  - output_contours.jpg")
    
    print()


def example_custom_threshold():
    """Example with custom threshold"""
    print("=== Custom Threshold Example ===\n")
    
    original_path = 'image_data/original_pan_card.jpg'
    test_path = 'image_data/original_pan_card.jpg'  # Replace with your test image
    
    # Use custom thresholds
    thresholds = [0.99, 0.95, 0.90, 0.85]
    
    for threshold in thresholds:
        tampered = is_tampered(original_path, test_path, threshold=threshold)
        print(f"Threshold {threshold * 100:.0f}%: {'Tampered' if tampered else 'Authentic'}")
    
    print()


if __name__ == '__main__':
    print("Pan Card Tampering Detector - Usage Examples")
    print("=" * 50)
    print()
    
    example_basic_usage()
    example_detailed_analysis()
    example_custom_threshold()
    
    print("=" * 50)
    print("For web interface, run: python app.py")
    print("Then visit: http://localhost:5000")
