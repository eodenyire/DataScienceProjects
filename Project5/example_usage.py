"""
Example Usage of Text Extractor Module

This script demonstrates various ways to use the text_extractor module
programmatically for extracting text from images.

Author: Emmanuel Odenyire
Email: eodenyire@gmail.com
"""

from text_extractor import TextExtractor, extract_text_simple, extract_text_preprocessed
import os


def example_basic_extraction():
    """
    Example 1: Basic text extraction from an image
    """
    print("=" * 60)
    print("Example 1: Basic Text Extraction")
    print("=" * 60)
    
    # Initialize the text extractor
    extractor = TextExtractor()
    
    # Path to your image (replace with actual image path)
    image_path = "image_data/sample_text_image.jpg"
    
    if os.path.exists(image_path):
        # Extract text
        text = extractor.extract_text(image_path)
        print(f"\nExtracted Text:\n{text}")
    else:
        print(f"Image not found: {image_path}")
        print("Please add a sample image to the image_data folder")
    
    print("\n")


def example_with_preprocessing():
    """
    Example 2: Text extraction with image preprocessing
    """
    print("=" * 60)
    print("Example 2: Text Extraction with Preprocessing")
    print("=" * 60)
    
    extractor = TextExtractor()
    image_path = "image_data/sample_text_image.jpg"
    
    if os.path.exists(image_path):
        # Extract text with preprocessing
        text = extractor.extract_text(image_path, preprocess=True)
        print(f"\nExtracted Text (with preprocessing):\n{text}")
    else:
        print(f"Image not found: {image_path}")
    
    print("\n")


def example_with_confidence_scores():
    """
    Example 3: Text extraction with confidence scores
    """
    print("=" * 60)
    print("Example 3: Text Extraction with Confidence Scores")
    print("=" * 60)
    
    extractor = TextExtractor()
    image_path = "image_data/sample_text_image.jpg"
    
    if os.path.exists(image_path):
        # Extract text with confidence scores
        result = extractor.extract_text_with_confidence(image_path)
        
        print(f"\nExtracted Text:\n{result['text']}")
        print(f"\nAverage Confidence: {result['average_confidence']:.2f}%")
        print(f"\nTotal Words Detected: {len(result['words'])}")
        
        # Show first 10 words with their confidence scores
        print("\nFirst 10 Words with Confidence Scores:")
        for i, word_info in enumerate(result['words'][:10], 1):
            print(f"{i}. '{word_info['text']}' - Confidence: {word_info['confidence']}%")
    else:
        print(f"Image not found: {image_path}")
    
    print("\n")


def example_multiple_languages():
    """
    Example 4: Text extraction with different languages
    """
    print("=" * 60)
    print("Example 4: Multi-Language Text Extraction")
    print("=" * 60)
    
    extractor = TextExtractor()
    
    # Example with English (default)
    image_path = "image_data/sample_text_image.jpg"
    
    if os.path.exists(image_path):
        text_eng = extractor.extract_text(image_path, language='eng')
        print(f"\nEnglish Text:\n{text_eng}")
    
    # Example with other languages (if you have images with different languages)
    # Uncomment and replace with actual image paths
    # text_fra = extractor.extract_text("image_data/french_text.jpg", language='fra')
    # text_spa = extractor.extract_text("image_data/spanish_text.jpg", language='spa')
    
    print("\n")


def example_region_extraction():
    """
    Example 5: Extract text from a specific region of an image
    """
    print("=" * 60)
    print("Example 5: Region-Based Text Extraction")
    print("=" * 60)
    
    extractor = TextExtractor()
    image_path = "image_data/sample_text_image.jpg"
    
    if os.path.exists(image_path):
        # Extract text from specific region (x, y, width, height)
        # These coordinates are examples - adjust based on your image
        x, y, width, height = 50, 50, 300, 100
        
        text = extractor.extract_text_from_region(
            image_path, x, y, width, height, preprocess=True
        )
        print(f"\nExtracted Text from Region ({x}, {y}, {width}x{height}):\n{text}")
    else:
        print(f"Image not found: {image_path}")
    
    print("\n")


def example_save_preprocessed():
    """
    Example 6: Save preprocessed image for inspection
    """
    print("=" * 60)
    print("Example 6: Save Preprocessed Image")
    print("=" * 60)
    
    extractor = TextExtractor()
    image_path = "image_data/sample_text_image.jpg"
    output_path = "static/uploads/preprocessed_example.jpg"
    
    if os.path.exists(image_path):
        # Save preprocessed image
        saved_path = extractor.save_preprocessed_image(image_path, output_path)
        print(f"\nPreprocessed image saved to: {saved_path}")
        print("You can view this image to see how preprocessing affects the image")
    else:
        print(f"Image not found: {image_path}")
    
    print("\n")


def example_convenience_functions():
    """
    Example 7: Using convenience functions
    """
    print("=" * 60)
    print("Example 7: Convenience Functions")
    print("=" * 60)
    
    image_path = "image_data/sample_text_image.jpg"
    
    if os.path.exists(image_path):
        # Simple extraction
        text1 = extract_text_simple(image_path)
        print(f"\nSimple Extraction:\n{text1}")
        
        # Preprocessed extraction
        text2 = extract_text_preprocessed(image_path)
        print(f"\nPreprocessed Extraction:\n{text2}")
    else:
        print(f"Image not found: {image_path}")
    
    print("\n")


def main():
    """
    Run all examples
    """
    print("\n")
    print("*" * 60)
    print("TEXT EXTRACTOR - USAGE EXAMPLES")
    print("*" * 60)
    print("\n")
    
    # Create necessary directories
    os.makedirs("image_data", exist_ok=True)
    os.makedirs("static/uploads", exist_ok=True)
    
    # Note about sample images
    print("NOTE: These examples require sample images in the 'image_data' folder.")
    print("Please add sample images before running the examples.")
    print("You can use any image containing text (screenshots, scanned documents, etc.)")
    print("\n")
    
    # Run examples
    try:
        example_basic_extraction()
        example_with_preprocessing()
        example_with_confidence_scores()
        example_multiple_languages()
        example_region_extraction()
        example_save_preprocessed()
        example_convenience_functions()
    except Exception as e:
        print(f"Error running examples: {str(e)}")
        print("\nMake sure you have:")
        print("1. Installed all requirements: pip install -r requirements.txt")
        print("2. Installed Tesseract OCR on your system")
        print("3. Added sample images to the image_data folder")
    
    print("*" * 60)
    print("END OF EXAMPLES")
    print("*" * 60)


if __name__ == "__main__":
    main()
