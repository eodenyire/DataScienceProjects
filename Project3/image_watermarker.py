"""
Image Watermarking Module

This module provides functionality to add text and image watermarks to images.
It supports various customization options including position, opacity, size, and color.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class ImageWatermarker:
    """
    A class to handle image watermarking operations.
    
    Supports both text and image (logo) watermarks with customizable properties.
    """
    
    def __init__(self):
        """Initialize the ImageWatermarker."""
        self.supported_positions = [
            'top-left', 'top-right', 'bottom-left', 'bottom-right', 'center'
        ]
    
    def add_text_watermark(self, image_path, text, output_path=None, 
                          position='bottom-right', font_size=36, 
                          color=(255, 255, 255), opacity=0.5):
        """
        Add text watermark to an image.
        
        Args:
            image_path (str): Path to the input image
            text (str): Text to use as watermark
            output_path (str): Path to save the watermarked image (optional)
            position (str): Position of watermark ('top-left', 'top-right', 
                          'bottom-left', 'bottom-right', 'center')
            font_size (int): Font size for the text
            color (tuple): RGB color tuple for text (default: white)
            opacity (float): Opacity of watermark (0.0 to 1.0)
            
        Returns:
            numpy.ndarray: Watermarked image
        """
        # Read image using PIL for better text rendering
        img_pil = Image.open(image_path).convert('RGBA')
        
        # Create a transparent overlay
        overlay = Image.new('RGBA', img_pil.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        # Try to use a better font, fall back to default if not available
        try:
            # Try common font paths
            font_paths = [
                '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
                '/System/Library/Fonts/Helvetica.ttc',
                'C:\\Windows\\Fonts\\arial.ttf',
                '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf'
            ]
            font = None
            for font_path in font_paths:
                try:
                    font = ImageFont.truetype(font_path, font_size)
                    break
                except:
                    continue
            
            if font is None:
                font = ImageFont.load_default()
        except:
            font = ImageFont.load_default()
        
        # Get text bounding box
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Calculate position
        x, y = self._calculate_position(
            img_pil.width, img_pil.height, 
            text_width, text_height, position
        )
        
        # Calculate color with opacity
        text_color = color + (int(255 * opacity),)
        
        # Draw text on overlay
        draw.text((x, y), text, font=font, fill=text_color)
        
        # Composite the overlay with the original image
        watermarked = Image.alpha_composite(img_pil, overlay)
        
        # Convert back to RGB
        watermarked = watermarked.convert('RGB')
        
        # Save if output path is provided
        if output_path:
            watermarked.save(output_path)
        
        # Convert to numpy array for OpenCV compatibility
        return np.array(watermarked)
    
    def add_image_watermark(self, image_path, logo_path, output_path=None,
                           position='bottom-right', size_ratio=0.15, opacity=0.5):
        """
        Add image (logo) watermark to an image.
        
        Args:
            image_path (str): Path to the input image
            logo_path (str): Path to the logo/watermark image
            output_path (str): Path to save the watermarked image (optional)
            position (str): Position of watermark ('top-left', 'top-right', 
                          'bottom-left', 'bottom-right', 'center')
            size_ratio (float): Size of logo relative to image (0.0 to 1.0)
            opacity (float): Opacity of watermark (0.0 to 1.0)
            
        Returns:
            numpy.ndarray: Watermarked image
        """
        # Read main image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image from {image_path}")
        
        # Read logo
        logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
        if logo is None:
            raise ValueError(f"Could not read logo from {logo_path}")
        
        # Convert BGR to BGRA if needed
        if logo.shape[2] == 3:
            logo = cv2.cvtColor(logo, cv2.COLOR_BGR2BGRA)
        
        # Calculate logo size based on ratio
        img_height, img_width = img.shape[:2]
        logo_width = int(img_width * size_ratio)
        aspect_ratio = logo.shape[0] / logo.shape[1]
        logo_height = int(logo_width * aspect_ratio)
        
        # Resize logo
        logo = cv2.resize(logo, (logo_width, logo_height), 
                         interpolation=cv2.INTER_AREA)
        
        # Calculate position
        x, y = self._calculate_position(
            img_width, img_height, 
            logo_width, logo_height, position
        )
        
        # Apply watermark with transparency
        watermarked = self._overlay_transparent(img, logo, x, y, opacity)
        
        # Save if output path is provided
        if output_path:
            cv2.imwrite(output_path, watermarked)
        
        return watermarked
    
    def _calculate_position(self, img_width, img_height, 
                           overlay_width, overlay_height, position):
        """
        Calculate the position coordinates for the watermark.
        
        Args:
            img_width (int): Width of the main image
            img_height (int): Height of the main image
            overlay_width (int): Width of the watermark
            overlay_height (int): Height of the watermark
            position (str): Desired position
            
        Returns:
            tuple: (x, y) coordinates for the watermark
        """
        margin = 20  # Margin from edges
        
        if position == 'top-left':
            x, y = margin, margin
        elif position == 'top-right':
            x = img_width - overlay_width - margin
            y = margin
        elif position == 'bottom-left':
            x = margin
            y = img_height - overlay_height - margin
        elif position == 'bottom-right':
            x = img_width - overlay_width - margin
            y = img_height - overlay_height - margin
        elif position == 'center':
            x = (img_width - overlay_width) // 2
            y = (img_height - overlay_height) // 2
        else:
            # Default to bottom-right
            x = img_width - overlay_width - margin
            y = img_height - overlay_height - margin
        
        return max(0, x), max(0, y)
    
    def _overlay_transparent(self, background, overlay, x, y, opacity=1.0):
        """
        Overlay a transparent image on a background image.
        
        Args:
            background (numpy.ndarray): Background image
            overlay (numpy.ndarray): Overlay image with alpha channel
            x (int): X coordinate for overlay placement
            y (int): Y coordinate for overlay placement
            opacity (float): Opacity of overlay (0.0 to 1.0)
            
        Returns:
            numpy.ndarray: Combined image
        """
        background_copy = background.copy()
        
        # Get dimensions
        bg_h, bg_w = background_copy.shape[:2]
        ov_h, ov_w = overlay.shape[:2]
        
        # Adjust overlay size if it goes out of bounds
        if x + ov_w > bg_w:
            ov_w = bg_w - x
            overlay = overlay[:, :ov_w]
        if y + ov_h > bg_h:
            ov_h = bg_h - y
            overlay = overlay[:ov_h, :]
        
        # Extract alpha channel
        if overlay.shape[2] == 4:
            alpha_overlay = overlay[:, :, 3] / 255.0
            alpha_overlay = alpha_overlay * opacity
            alpha_background = 1.0 - alpha_overlay
            
            # Blend the images
            for c in range(3):
                background_copy[y:y+ov_h, x:x+ov_w, c] = (
                    alpha_overlay * overlay[:, :, c] +
                    alpha_background * background_copy[y:y+ov_h, x:x+ov_w, c]
                )
        else:
            # No alpha channel, use simple blending
            alpha = opacity
            background_copy[y:y+ov_h, x:x+ov_w] = cv2.addWeighted(
                overlay[:, :, :3], alpha,
                background_copy[y:y+ov_h, x:x+ov_w], 1 - alpha, 0
            )
        
        return background_copy


def create_text_watermark(image_path, text, output_path, **kwargs):
    """
    Convenience function to create a text watermark.
    
    Args:
        image_path (str): Path to input image
        text (str): Watermark text
        output_path (str): Path to save watermarked image
        **kwargs: Additional arguments for add_text_watermark
        
    Returns:
        numpy.ndarray: Watermarked image
    """
    watermarker = ImageWatermarker()
    return watermarker.add_text_watermark(image_path, text, output_path, **kwargs)


def create_image_watermark(image_path, logo_path, output_path, **kwargs):
    """
    Convenience function to create an image watermark.
    
    Args:
        image_path (str): Path to input image
        logo_path (str): Path to logo image
        output_path (str): Path to save watermarked image
        **kwargs: Additional arguments for add_image_watermark
        
    Returns:
        numpy.ndarray: Watermarked image
    """
    watermarker = ImageWatermarker()
    return watermarker.add_image_watermark(image_path, logo_path, output_path, **kwargs)
