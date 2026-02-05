"""
Example usage of the Image Watermarking module

This script demonstrates how to use the image_watermarker module 
to add text and image watermarks to images programmatically.
"""

from image_watermarker import ImageWatermarker
import os


def example_text_watermark():
    """Example: Add a text watermark to an image."""
    print("Example 1: Adding Text Watermark")
    print("-" * 50)
    
    watermarker = ImageWatermarker()
    
    # Sample parameters
    image_path = "image_data/sample_image.jpg"
    output_path = "static/uploads/text_watermarked.jpg"
    text = "© 2024 Your Name"
    
    # Check if sample image exists
    if not os.path.exists(image_path):
        print(f"Sample image not found at {image_path}")
        print("Please add a sample image to the image_data folder")
        return
    
    # Add text watermark
    try:
        result = watermarker.add_text_watermark(
            image_path=image_path,
            text=text,
            output_path=output_path,
            position='bottom-right',
            font_size=40,
            color=(255, 255, 255),  # White text
            opacity=0.7
        )
        print(f"✓ Text watermark added successfully!")
        print(f"  Output saved to: {output_path}")
        print(f"  Text: {text}")
        print(f"  Position: bottom-right")
        print(f"  Font size: 40px")
        print(f"  Opacity: 0.7")
    except Exception as e:
        print(f"✗ Error: {str(e)}")
    
    print()


def example_image_watermark():
    """Example: Add an image/logo watermark to an image."""
    print("Example 2: Adding Image/Logo Watermark")
    print("-" * 50)
    
    watermarker = ImageWatermarker()
    
    # Sample parameters
    image_path = "image_data/sample_image.jpg"
    logo_path = "static/logos/sample_logo.png"
    output_path = "static/uploads/logo_watermarked.jpg"
    
    # Check if files exist
    if not os.path.exists(image_path):
        print(f"Sample image not found at {image_path}")
        print("Please add a sample image to the image_data folder")
        return
    
    if not os.path.exists(logo_path):
        print(f"Logo not found at {logo_path}")
        print("Please add a logo to the static/logos folder")
        return
    
    # Add image watermark
    try:
        result = watermarker.add_image_watermark(
            image_path=image_path,
            logo_path=logo_path,
            output_path=output_path,
            position='bottom-right',
            size_ratio=0.15,  # 15% of image width
            opacity=0.6
        )
        print(f"✓ Image watermark added successfully!")
        print(f"  Output saved to: {output_path}")
        print(f"  Logo: {logo_path}")
        print(f"  Position: bottom-right")
        print(f"  Size: 15% of image width")
        print(f"  Opacity: 0.6")
    except Exception as e:
        print(f"✗ Error: {str(e)}")
    
    print()


def example_custom_positions():
    """Example: Test different watermark positions."""
    print("Example 3: Testing Different Positions")
    print("-" * 50)
    
    watermarker = ImageWatermarker()
    image_path = "image_data/sample_image.jpg"
    
    if not os.path.exists(image_path):
        print(f"Sample image not found at {image_path}")
        return
    
    positions = ['top-left', 'top-right', 'bottom-left', 'bottom-right', 'center']
    
    print("Creating watermarks at different positions...")
    for position in positions:
        output_path = f"static/uploads/watermark_{position}.jpg"
        try:
            watermarker.add_text_watermark(
                image_path=image_path,
                text=f"Position: {position}",
                output_path=output_path,
                position=position,
                font_size=30,
                color=(255, 255, 0),  # Yellow text
                opacity=0.8
            )
            print(f"  ✓ Created watermark at {position}")
        except Exception as e:
            print(f"  ✗ Error at {position}: {str(e)}")
    
    print()


def example_batch_watermark():
    """Example: Add watermarks to multiple images."""
    print("Example 4: Batch Watermarking")
    print("-" * 50)
    
    watermarker = ImageWatermarker()
    
    # Create image_data directory if it doesn't exist
    os.makedirs("image_data", exist_ok=True)
    os.makedirs("static/uploads", exist_ok=True)
    
    # List of images (add your own images here)
    images = [
        "image_data/sample_image.jpg",
    ]
    
    text = "© Protected"
    
    for image_path in images:
        if os.path.exists(image_path):
            filename = os.path.basename(image_path)
            output_path = f"static/uploads/batch_{filename}"
            
            try:
                watermarker.add_text_watermark(
                    image_path=image_path,
                    text=text,
                    output_path=output_path,
                    position='bottom-right',
                    font_size=36,
                    color=(255, 255, 255),
                    opacity=0.6
                )
                print(f"  ✓ Processed: {filename}")
            except Exception as e:
                print(f"  ✗ Error processing {filename}: {str(e)}")
        else:
            print(f"  ⚠ File not found: {image_path}")
    
    print()


def main():
    """Run all examples."""
    print("=" * 50)
    print("Image Watermarking Examples")
    print("=" * 50)
    print()
    
    # Create necessary directories
    os.makedirs("image_data", exist_ok=True)
    os.makedirs("static/uploads", exist_ok=True)
    os.makedirs("static/logos", exist_ok=True)
    
    # Run examples
    example_text_watermark()
    example_image_watermark()
    example_custom_positions()
    example_batch_watermark()
    
    print("=" * 50)
    print("Examples completed!")
    print("=" * 50)
    print()
    print("Note: To run these examples successfully, make sure to:")
    print("1. Add sample images to the image_data folder")
    print("2. Add a logo to the static/logos folder")
    print("3. Install required dependencies: pip install -r requirements.txt")


if __name__ == "__main__":
    main()
