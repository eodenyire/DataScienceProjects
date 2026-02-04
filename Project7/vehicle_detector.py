"""
Vehicle Detection and Counting Module

This module provides functionality to detect and count vehicles in images and videos
using OpenCV's background subtraction and contour detection techniques.

Author: Emmanuel Odenyire
Email: eodenyire@gmail.com
"""

import cv2
import numpy as np
from typing import Tuple, List, Dict
import imutils


class VehicleDetector:
    """
    A class for detecting and counting vehicles in images and video frames.
    
    This uses background subtraction and contour detection to identify moving vehicles.
    """
    
    def __init__(self, min_contour_area: int = 500, detect_shadows: bool = False):
        """
        Initialize the Vehicle Detector.
        
        Args:
            min_contour_area: Minimum contour area to be considered a vehicle
            detect_shadows: Whether to detect shadows (can reduce false positives)
        """
        self.min_contour_area = min_contour_area
        
        # Initialize background subtractor
        # Using MOG2 algorithm for background subtraction
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=16,
            detectShadows=detect_shadows
        )
        
        # Counter for vehicles
        self.vehicle_count = 0
        
        # Line position for counting (percentage from top)
        self.count_line_position = 0.5
        
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess the frame for vehicle detection.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            Preprocessed frame
        """
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        return blurred
    
    def detect_vehicles_in_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, int, List[Tuple[int, int, int, int]]]:
        """
        Detect vehicles in a single frame.
        
        Args:
            frame: Input frame
            
        Returns:
            Tuple of (processed_frame, vehicle_count, bounding_boxes)
        """
        # Preprocess
        preprocessed = self.preprocess_frame(frame)
        
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(preprocessed)
        
        # Apply morphological operations to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours = cv2.findContours(fg_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        
        # Filter and draw bounding boxes
        bounding_boxes = []
        output_frame = frame.copy()
        vehicle_count = 0
        
        for contour in contours:
            # Filter by area
            if cv2.contourArea(contour) < self.min_contour_area:
                continue
            
            # Get bounding box
            (x, y, w, h) = cv2.boundingRect(contour)
            
            # Filter by aspect ratio (vehicles typically have certain proportions)
            aspect_ratio = w / float(h)
            if 0.2 < aspect_ratio < 5.0:  # Reasonable vehicle proportions
                bounding_boxes.append((x, y, w, h))
                vehicle_count += 1
                
                # Draw bounding box
                cv2.rectangle(output_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Draw vehicle number
                cv2.putText(output_frame, f'V{vehicle_count}', (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw count line
        height = frame.shape[0]
        count_line_y = int(height * self.count_line_position)
        cv2.line(output_frame, (0, count_line_y), (frame.shape[1], count_line_y), (0, 0, 255), 2)
        
        # Display count
        cv2.putText(output_frame, f'Vehicles: {vehicle_count}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return output_frame, vehicle_count, bounding_boxes
    
    def detect_vehicles_in_image(self, image_path: str) -> Tuple[np.ndarray, int, Dict[str, any]]:
        """
        Detect vehicles in a static image using edge detection and contours.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Tuple of (processed_image, vehicle_count, details)
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image from {image_path}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Dilate edges to close gaps
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours and count vehicles
        output_image = image.copy()
        vehicle_count = 0
        detected_vehicles = []
        
        for contour in contours:
            # Filter by area
            area = cv2.contourArea(contour)
            if area < self.min_contour_area:
                continue
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by aspect ratio
            aspect_ratio = w / float(h)
            if 0.5 < aspect_ratio < 4.0:  # Reasonable vehicle proportions
                vehicle_count += 1
                detected_vehicles.append({
                    'id': vehicle_count,
                    'x': int(x),
                    'y': int(y),
                    'width': int(w),
                    'height': int(h),
                    'area': float(area)
                })
                
                # Draw bounding box
                cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Label vehicle
                cv2.putText(output_image, f'V{vehicle_count}', (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Add vehicle count text
        cv2.putText(output_image, f'Total Vehicles: {vehicle_count}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        
        details = {
            'total_count': vehicle_count,
            'vehicles': detected_vehicles,
            'image_size': {'width': image.shape[1], 'height': image.shape[0]}
        }
        
        return output_image, vehicle_count, details
    
    def process_video(self, video_path: str, output_path: str = None) -> Dict[str, any]:
        """
        Process a video file to detect and count vehicles.
        
        Args:
            video_path: Path to input video
            output_path: Optional path to save processed video
            
        Returns:
            Dictionary with processing results
        """
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video from {video_path}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize video writer if output path provided
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Process frames
        frame_count = 0
        max_vehicles = 0
        total_vehicles_detected = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect vehicles in frame
            processed_frame, vehicle_count, _ = self.detect_vehicles_in_frame(frame)
            
            # Update statistics
            frame_count += 1
            total_vehicles_detected += vehicle_count
            max_vehicles = max(max_vehicles, vehicle_count)
            
            # Write frame if output path provided
            if out:
                out.write(processed_frame)
        
        # Release resources
        cap.release()
        if out:
            out.release()
        
        # Calculate statistics
        avg_vehicles = total_vehicles_detected / frame_count if frame_count > 0 else 0
        
        results = {
            'total_frames': frame_count,
            'max_vehicles_in_frame': max_vehicles,
            'average_vehicles_per_frame': avg_vehicles,
            'output_video': output_path if output_path else None
        }
        
        return results
    
    def reset(self):
        """Reset the detector state."""
        self.vehicle_count = 0
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=16,
            detectShadows=False
        )


def create_sample_image_with_vehicles(output_path: str = 'sample_traffic.jpg'):
    """
    Create a sample image with simple vehicle-like shapes for testing.
    
    Args:
        output_path: Path to save the sample image
    """
    # Create a blank image (road scene)
    img = np.ones((400, 600, 3), dtype=np.uint8) * 128
    
    # Draw road
    cv2.rectangle(img, (0, 100), (600, 300), (80, 80, 80), -1)
    
    # Draw lane markings
    for x in range(50, 600, 100):
        cv2.rectangle(img, (x, 195), (x + 40, 205), (255, 255, 255), -1)
    
    # Draw simple vehicle shapes (rectangles)
    vehicles = [
        (50, 120, 80, 50),    # x, y, width, height
        (200, 230, 90, 55),
        (400, 140, 85, 52),
        (500, 220, 80, 50)
    ]
    
    for i, (x, y, w, h) in enumerate(vehicles):
        # Vehicle body
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 200), -1)
        # Windows
        cv2.rectangle(img, (x + 5, y + 5), (x + w - 5, y + 20), (100, 150, 255), -1)
        # Wheels
        cv2.circle(img, (x + 15, y + h), 8, (0, 0, 0), -1)
        cv2.circle(img, (x + w - 15, y + h), 8, (0, 0, 0), -1)
    
    # Save image
    cv2.imwrite(output_path, img)
    print(f"Sample image created at {output_path}")
    return output_path


if __name__ == "__main__":
    """
    Example usage of the VehicleDetector class.
    """
    print("Vehicle Detection Module")
    print("=" * 50)
    
    # Create sample image
    sample_image = create_sample_image_with_vehicles('sample_traffic.jpg')
    
    # Initialize detector
    detector = VehicleDetector(min_contour_area=500)
    
    # Process the sample image
    print("\nProcessing sample image...")
    processed_img, count, details = detector.detect_vehicles_in_image(sample_image)
    
    print(f"Detected {count} vehicles")
    print(f"Details: {details}")
    
    # Save result
    cv2.imwrite('result_traffic.jpg', processed_img)
    print("Result saved as 'result_traffic.jpg'")
