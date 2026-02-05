"""
Example usage of the Vehicle Detection Module

This script demonstrates how to use the VehicleDetector class
for detecting and counting vehicles in images.

Author: Emmanuel Odenyire
Email: eodenyire@gmail.com
"""

import os
from vehicle_detector import VehicleDetector, create_sample_image_with_vehicles
import cv2


def example_basic_detection():
    """
    Example 1: Basic vehicle detection on a sample image.
    """
    print("=" * 60)
    print("Example 1: Basic Vehicle Detection")
    print("=" * 60)
    
    # Create a sample image
    print("\n1. Creating sample traffic image...")
    sample_path = create_sample_image_with_vehicles('sample_traffic.jpg')
    
    # Initialize detector
    print("2. Initializing vehicle detector...")
    detector = VehicleDetector(min_contour_area=500)
    
    # Detect vehicles
    print("3. Detecting vehicles...")
    processed_image, vehicle_count, details = detector.detect_vehicles_in_image(sample_path)
    
    # Display results
    print(f"\n✅ Detection Complete!")
    print(f"   Total vehicles detected: {vehicle_count}")
    print(f"   Image dimensions: {details['image_size']['width']}x{details['image_size']['height']}")
    
    # Save result
    output_path = 'result_basic_detection.jpg'
    cv2.imwrite(output_path, processed_image)
    print(f"   Result saved to: {output_path}")
    
    # Show vehicle details
    if details['vehicles']:
        print("\n   Individual Vehicle Details:")
        for vehicle in details['vehicles']:
            print(f"   - Vehicle {vehicle['id']}: "
                  f"Position ({vehicle['x']}, {vehicle['y']}), "
                  f"Size {vehicle['width']}x{vehicle['height']}, "
                  f"Area {vehicle['area']:.2f}px²")
    
    return processed_image, vehicle_count, details


def example_with_custom_settings():
    """
    Example 2: Vehicle detection with custom settings.
    """
    print("\n" + "=" * 60)
    print("Example 2: Detection with Custom Settings")
    print("=" * 60)
    
    # Create sample image
    print("\n1. Creating sample traffic image...")
    sample_path = create_sample_image_with_vehicles('sample_traffic_2.jpg')
    
    # Initialize detector with custom settings
    print("2. Initializing detector with custom minimum area (1000px²)...")
    detector = VehicleDetector(min_contour_area=1000)  # Higher threshold
    
    # Detect vehicles
    print("3. Detecting larger vehicles...")
    processed_image, vehicle_count, details = detector.detect_vehicles_in_image(sample_path)
    
    # Display results
    print(f"\n✅ Detection Complete!")
    print(f"   Vehicles detected (with higher threshold): {vehicle_count}")
    
    # Save result
    output_path = 'result_custom_settings.jpg'
    cv2.imwrite(output_path, processed_image)
    print(f"   Result saved to: {output_path}")
    
    return processed_image, vehicle_count, details


def example_compare_thresholds():
    """
    Example 3: Compare detection with different thresholds.
    """
    print("\n" + "=" * 60)
    print("Example 3: Comparing Different Thresholds")
    print("=" * 60)
    
    # Create sample image
    sample_path = create_sample_image_with_vehicles('sample_traffic_compare.jpg')
    
    thresholds = [300, 500, 1000, 2000]
    results = []
    
    print("\nTesting different minimum contour area thresholds...")
    for threshold in thresholds:
        detector = VehicleDetector(min_contour_area=threshold)
        _, count, _ = detector.detect_vehicles_in_image(sample_path)
        results.append((threshold, count))
        print(f"   Threshold {threshold}px²: {count} vehicles detected")
    
    print("\n✅ Comparison Complete!")
    print("   Lower thresholds detect more objects (including small noise)")
    print("   Higher thresholds detect only larger, more prominent vehicles")
    
    return results


def example_processing_statistics():
    """
    Example 4: Get detailed processing statistics.
    """
    print("\n" + "=" * 60)
    print("Example 4: Detailed Processing Statistics")
    print("=" * 60)
    
    # Create and process image
    sample_path = create_sample_image_with_vehicles('sample_traffic_stats.jpg')
    detector = VehicleDetector(min_contour_area=500)
    processed_image, vehicle_count, details = detector.detect_vehicles_in_image(sample_path)
    
    # Calculate additional statistics
    if details['vehicles']:
        areas = [v['area'] for v in details['vehicles']]
        widths = [v['width'] for v in details['vehicles']]
        heights = [v['height'] for v in details['vehicles']]
        
        print(f"\n✅ Statistics:")
        print(f"   Total vehicles: {vehicle_count}")
        print(f"   Average area: {sum(areas)/len(areas):.2f}px²")
        print(f"   Largest vehicle: {max(areas):.2f}px²")
        print(f"   Smallest vehicle: {min(areas):.2f}px²")
        print(f"   Average width: {sum(widths)/len(widths):.2f}px")
        print(f"   Average height: {sum(heights)/len(heights):.2f}px")
    
    return details


def main():
    """
    Main function to run all examples.
    """
    print("\n" + "=" * 60)
    print("VEHICLE DETECTION MODULE - EXAMPLE USAGE")
    print("=" * 60)
    print("\nThis script demonstrates various uses of the VehicleDetector class")
    
    try:
        # Run examples
        example_basic_detection()
        example_with_custom_settings()
        example_compare_thresholds()
        example_processing_statistics()
        
        print("\n" + "=" * 60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nGenerated files:")
        print("  - sample_traffic*.jpg (input images)")
        print("  - result_*.jpg (processed images with detections)")
        print("\nYou can now view these images to see the detection results.")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
