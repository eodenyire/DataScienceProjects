# Project 1: Pan Card Tampering Detector

## Introduction to Pan Card Tampering Detector

The Pan Card Tampering Detector is an advanced web application that uses artificial intelligence and computer vision techniques to identify tampering or modifications in Pan Card (Permanent Account Number) images. This tool leverages OpenCV and the Structural Similarity Index (SSIM) algorithm to detect even minor alterations in document images, making it valuable for document verification and fraud detection.

### Key Features
- 🔍 **Advanced Detection**: Uses SSIM algorithm to detect even minor tampering
- ⚡ **Fast Processing**: Results in seconds with detailed analysis
- 📊 **Visual Analysis**: Shows difference, threshold, and contour images
- 🎯 **High Accuracy**: Precise similarity scoring system
- 🌐 **Web Interface**: Easy-to-use Flask-based web application

## Loading Libraries and Data Set

### Required Libraries
The project uses the following Python libraries:

```python
# Web Framework
Flask==2.3.2
Werkzeug>=3.0.3

# Computer Vision & Image Processing
opencv-python>=4.8.1.78
Pillow>=10.3.0
numpy>=1.24.0
scikit-image (for SSIM calculation)
imutils (for contour processing)
```

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/eodenyire/DataScienceProjects.git
cd DataScienceProjects/Project1
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Prepare the dataset:**
   - Place the original Pan Card image in the `image_data/` folder
   - Name it `original_pan_card.jpg`

## Creating the Pan Card Detector with OpenCV

The core detection logic is implemented in `pan_card_detector.py`, which includes:

### Detection Algorithm

1. **Image Loading and Preprocessing**
   - Load both original and user-uploaded images
   - Resize images to match dimensions
   - Convert to grayscale for analysis

2. **SSIM Calculation**
   - Compute Structural Similarity Index between images
   - Generate difference image highlighting changes

3. **Threshold Processing**
   - Apply binary threshold to difference image
   - Use Otsu's method for automatic threshold determination

4. **Contour Detection**
   - Find contours in thresholded image
   - Draw bounding boxes around detected changes

### Key Functions

- `detect_tampering(original_path, user_path)`: Main detection function
- `compare_images(original_path, user_path)`: Returns similarity score
- `is_tampered(original_path, user_path, threshold)`: Boolean check for tampering

## Creating the Flask App

The Flask application (`app.py`) provides a web interface for the detector:

### Features

1. **File Upload Handling**
   - Accepts PNG, JPG, and JPEG formats
   - File size limit: 16MB
   - Secure filename processing

2. **Image Processing Pipeline**
   - Validate uploaded file
   - Perform tampering detection
   - Generate result images
   - Calculate similarity score

3. **Result Display**
   - Show original vs uploaded image
   - Display difference and threshold images
   - Highlight tampered regions with contours
   - Provide similarity percentage

### Routes

- `/` - Home page with upload form
- `/upload` - Handle file upload and detection (POST)
- `/about` - Information about the application

## Creating Important Functions

### Image Processing Functions

1. **`allowed_file(filename)`**
   - Validates file extension
   - Ensures only image files are accepted

2. **`save_processed_images(diff, thresh, contours, upload_folder)`**
   - Saves analysis result images
   - Returns paths to saved images

3. **`detect_tampering(original_image_path, user_image_path)`**
   - Core detection algorithm
   - Returns similarity score and processed images

## Usage

### Running the Application

1. **Start the Flask server:**
```bash
# For development (with debug mode)
export FLASK_ENV=development
export SECRET_KEY='dev-secret-key'
python app.py

# For production (recommended)
export FLASK_ENV=production
export SECRET_KEY='your-production-secret-key'
# Use a production WSGI server like gunicorn
# pip install gunicorn
# gunicorn -w 4 -b 0.0.0.0:5000 app:app
python app.py
```

2. **Access the application:**
   - Open your browser and navigate to `http://localhost:5000`

3. **Upload a Pan Card:**
   - Click "Choose File" and select a Pan Card image
   - Click "Check for Tampering"

4. **View Results:**
   - See the similarity score
   - Review visual analysis
   - Check highlighted differences

### Expected Output

- **Similarity Score**: Percentage match with original (0-100%)
- **Status**: Authentic (≥95%) or Tampered (<95%)
- **Visual Analysis**:
  - Original Pan Card
  - Uploaded Pan Card
  - Difference Image (pixel-level changes)
  - Threshold Image (significant changes)
  - Contours Image (highlighted tampered regions)

## Project Structure

```
Project1/
├── app.py                      # Flask application
├── pan_card_detector.py        # Detection logic
├── requirements.txt            # Python dependencies
├── README.md                   # Documentation
├── templates/                  # HTML templates
│   ├── base.html              # Base template
│   ├── index.html             # Home page
│   ├── result.html            # Results page
│   └── about.html             # About page
├── static/
│   └── uploads/               # Uploaded and processed images
└── image_data/
    └── original_pan_card.jpg  # Reference image
```

## How It Works

1. **Upload Phase**: User uploads a Pan Card image through the web interface
2. **Preprocessing**: System resizes and converts images to grayscale
3. **SSIM Analysis**: Calculates structural similarity between images
4. **Difference Detection**: Identifies pixel-level differences
5. **Contour Marking**: Highlights tampered regions with red rectangles
6. **Result Display**: Shows similarity score and visual analysis

## Technical Details

### Structural Similarity Index (SSIM)

SSIM is a perceptual metric that quantifies image quality degradation. Unlike simple pixel-by-pixel comparison, SSIM considers:
- **Luminance**: Brightness comparison
- **Contrast**: Local contrast comparison
- **Structure**: Structural information comparison

Formula: SSIM(x,y) = [l(x,y)]^α · [c(x,y)]^β · [s(x,y)]^γ

Where:
- l(x,y) = luminance comparison
- c(x,y) = contrast comparison
- s(x,y) = structure comparison

### Threshold Value

The default threshold is **95%** similarity:
- **≥95%**: Image is considered authentic
- **<95%**: Tampering detected

This threshold can be adjusted based on requirements.

## Applications

- Banking and financial institutions for KYC verification
- Government agencies for identity verification
- Online platforms requiring document authentication
- Security systems for fraud detection
- Digital forensics and investigation

## Limitations

- Image quality and lighting conditions can affect accuracy
- Very sophisticated tampering techniques may require additional verification
- The system requires a high-quality reference image
- Should be used as part of a comprehensive verification process

## Future Enhancements

- Support for multiple document types
- Machine learning-based classification
- Batch processing capability
- API endpoint for integration
- Database storage for audit trail
- Advanced preprocessing for better accuracy

## License

This project is part of the DataScienceProjects repository.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Author

Developed as part of the Data Science Projects series.
