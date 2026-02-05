"""
Text Extractor Module for extracting text from images using OCR.

This module provides functionality to extract text from images using
Tesseract OCR. It supports various image formats and provides both
basic and advanced text extraction features.

Author: Emmanuel Odenyire
Email: eodenyire@gmail.com
"""

import cv2
import pytesseract
from PIL import Image
import numpy as np
import os


class TextExtractor:
    """
    A class for extracting text from images using Tesseract OCR.
    
    This class provides methods to extract text from images with various
    preprocessing options to improve OCR accuracy.
    """
    
    def __init__(self, tesseract_cmd=None):
        """
        Initialize the TextExtractor.
        
        Args:
            tesseract_cmd (str, optional): Path to tesseract executable.
                If None, uses system default.
        """
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    
    def extract_text(self, image_path, preprocess=False, language='eng'):
        """
        Extract text from an image file.
        
        Args:
            image_path (str): Path to the image file
            preprocess (bool): Whether to preprocess the image for better OCR
            language (str): Language code for OCR (default: 'eng')
        
        Returns:
            str: Extracted text from the image
        
        Raises:
            FileNotFoundError: If the image file doesn't exist
            Exception: If OCR processing fails
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        try:
            # Load image
            if preprocess:
                image = self._preprocess_image(image_path)
            else:
                image = Image.open(image_path)
            
            # Extract text using Tesseract
            text = pytesseract.image_to_string(image, lang=language)
            
            return text.strip()
        
        except Exception as e:
            raise Exception(f"Error extracting text: {str(e)}")
    
    def extract_text_with_confidence(self, image_path, preprocess=False, language='eng'):
        """
        Extract text with confidence scores from an image.
        
        Args:
            image_path (str): Path to the image file
            preprocess (bool): Whether to preprocess the image
            language (str): Language code for OCR
        
        Returns:
            dict: Dictionary containing extracted text and detailed information
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        try:
            # Load image
            if preprocess:
                image = self._preprocess_image(image_path)
            else:
                image = Image.open(image_path)
            
            # Extract detailed OCR data
            data = pytesseract.image_to_data(image, lang=language, output_type=pytesseract.Output.DICT)
            
            # Process results
            result = {
                'text': pytesseract.image_to_string(image, lang=language).strip(),
                'words': [],
                'average_confidence': 0
            }
            
            # Extract words with confidence scores
            confidences = []
            for i, text in enumerate(data['text']):
                if text.strip():
                    conf = int(data['conf'][i])
                    if conf > 0:  # Filter out invalid confidence scores
                        word_info = {
                            'text': text,
                            'confidence': conf,
                            'position': {
                                'x': data['left'][i],
                                'y': data['top'][i],
                                'width': data['width'][i],
                                'height': data['height'][i]
                            }
                        }
                        result['words'].append(word_info)
                        confidences.append(conf)
            
            # Calculate average confidence
            if confidences:
                result['average_confidence'] = sum(confidences) / len(confidences)
            
            return result
        
        except Exception as e:
            raise Exception(f"Error extracting text with confidence: {str(e)}")
    
    def _preprocess_image(self, image_path):
        """
        Preprocess image to improve OCR accuracy.
        
        Args:
            image_path (str): Path to the image file
        
        Returns:
            PIL.Image: Preprocessed image
        """
        # Read image using OpenCV
        img = cv2.imread(image_path)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding to make text more distinct
        # Using Otsu's thresholding
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(thresh, h=10)
        
        # Convert back to PIL Image
        pil_image = Image.fromarray(denoised)
        
        return pil_image
    
    def extract_text_from_region(self, image_path, x, y, width, height, preprocess=False):
        """
        Extract text from a specific region of an image.
        
        Args:
            image_path (str): Path to the image file
            x (int): X coordinate of top-left corner
            y (int): Y coordinate of top-left corner
            width (int): Width of the region
            height (int): Height of the region
            preprocess (bool): Whether to preprocess the image
        
        Returns:
            str: Extracted text from the region
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        try:
            # Load image
            image = Image.open(image_path)
            
            # Crop to region
            region = image.crop((x, y, x + width, y + height))
            
            # Apply preprocessing if requested
            if preprocess:
                # Convert to OpenCV format
                region_cv = cv2.cvtColor(np.array(region), cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(region_cv, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                region = Image.fromarray(thresh)
            
            # Extract text
            text = pytesseract.image_to_string(region)
            
            return text.strip()
        
        except Exception as e:
            raise Exception(f"Error extracting text from region: {str(e)}")
    
    def save_preprocessed_image(self, image_path, output_path):
        """
        Save a preprocessed version of the image.
        
        Args:
            image_path (str): Path to the input image
            output_path (str): Path to save the preprocessed image
        
        Returns:
            str: Path to the saved preprocessed image
        """
        preprocessed = self._preprocess_image(image_path)
        preprocessed.save(output_path)
        return output_path


def extract_text_simple(image_path):
    """
    Simple function to extract text from an image.
    
    This is a convenience function for quick text extraction.
    
    Args:
        image_path (str): Path to the image file
    
    Returns:
        str: Extracted text
    """
    extractor = TextExtractor()
    return extractor.extract_text(image_path)


def extract_text_preprocessed(image_path):
    """
    Extract text from an image with preprocessing.
    
    This function applies preprocessing to improve OCR accuracy.
    
    Args:
        image_path (str): Path to the image file
    
    Returns:
        str: Extracted text
    """
    extractor = TextExtractor()
    return extractor.extract_text(image_path, preprocess=True)
