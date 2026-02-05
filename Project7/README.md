# Project 7: Vehicle Detection and Counting Application Flask

**Developer:** Emmanuel Odenyire  
**Email:** eodenyire@gmail.com

## Introduction

The Vehicle Detection and Counting Application is an intelligent web application that uses computer vision and image processing to automatically detect and count vehicles in images. Built with Flask and OpenCV, this application provides an intuitive interface for traffic analysis, parking lot monitoring, and transportation research. The system uses advanced edge detection, contour analysis, and morphological operations to identify vehicles with high accuracy.

### Key Features

- 🚗 **Automatic Detection**: Uses OpenCV's edge detection and contour analysis
- ⚡ **Fast Processing**: Get results in seconds with efficient algorithms
- 📊 **Detailed Statistics**: Shows vehicle count, positions, and dimensions
- 🎯 **Visual Feedback**: Highlights detected vehicles with numbered bounding boxes
- 🌐 **Web Interface**: User-friendly Flask-based web application
- 📈 **Scalable**: Works with various image sizes and formats

## Importing Libraries and Data

### Required Libraries

The project uses the following Python libraries for computer vision and web development:

```python
# Web Framework
Flask==2.3.2
Werkzeug>=3.0.3

# Computer Vision & Image Processing
opencv-python>=4.8.1.78
opencv-contrib-python>=4.8.1.78
Pillow>=10.3.0
numpy>=1.24.0

# Utilities
imutils>=0.5.4
```

### Installation

Follow these steps to set up the project:

1. **Clone the repository:**
```bash
git clone https://github.com/eodenyire/DataScienceProjects.git
cd DataScienceProjects/Project7
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Create necessary directories:**
```bash
mkdir -p static/uploads image_data video_data
```

### Data Preparation

The application works with:
- **Image Formats**: PNG, JPG, JPEG, BMP
- **Recommended Input**: Clear, well-lit images of roads, highways, or parking areas
- **Image Size**: Up to 16MB

## Transforming Images and Data

The vehicle detection system uses a sophisticated multi-stage image processing pipeline:

### 1. Image Preprocessing

```python
def preprocess_frame(frame):
    """
    Preprocess the frame for vehicle detection.
    """
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    return blurred
```

**Purpose**: Remove noise and smooth the image while preserving edges.

### 2. Edge Detection

The system uses Canny edge detection to identify object boundaries:

```python
# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply edge detection
edges = cv2.Canny(blurred, 50, 150)
```

**Parameters**:
- Lower threshold: 50
- Upper threshold: 150
- These values detect significant edges while filtering noise

### 3. Morphological Operations

Morphological operations enhance the detected edges:

```python
# Create kernel for morphological operations
kernel = np.ones((3, 3), np.uint8)

# Dilate edges to close gaps
dilated = cv2.dilate(edges, kernel, iterations=2)
```

**Purpose**: Connect broken edges and fill small gaps in vehicle boundaries.

### 4. Contour Detection and Filtering

The system finds and filters contours based on vehicle characteristics:

```python
# Find contours
contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter contours
for contour in contours:
    area = cv2.contourArea(contour)
    if area < min_contour_area:
        continue
    
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / float(h)
    
    # Vehicle proportions: width to height ratio
    if 0.5 < aspect_ratio < 4.0:
        # This is likely a vehicle
        vehicle_count += 1
```

**Filtering Criteria**:
- **Minimum Area**: 500 pixels² (configurable)
- **Aspect Ratio**: 0.5 to 4.0 (reasonable vehicle proportions)
- **Contour Type**: External contours only

### 5. Vehicle Detection Class

The `VehicleDetector` class encapsulates all detection logic:

```python
class VehicleDetector:
    def __init__(self, min_contour_area=500):
        self.min_contour_area = min_contour_area
        # Background subtractor for video processing
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=16,
            detectShadows=False
        )
    
    def detect_vehicles_in_image(self, image_path):
        """Detect vehicles in a static image."""
        # Load and preprocess image
        # Apply edge detection
        # Find and filter contours
        # Return processed image, count, and details
```

### Data Transformation Steps

1. **Input Image** → Color image in BGR format
2. **Grayscale Conversion** → Single channel for processing
3. **Gaussian Blur** → Noise reduction (5×5 kernel)
4. **Edge Detection** → Binary edge map
5. **Morphological Dilation** → Enhanced edge connectivity
6. **Contour Detection** → List of detected objects
7. **Filtering** → Valid vehicle contours only
8. **Visualization** → Annotated output image

### Advanced Features

#### Background Subtraction (for Video)

```python
# MOG2 background subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=500,        # Number of frames for background model
    varThreshold=16,    # Threshold for pixel classification
    detectShadows=False # Disable shadow detection
)

# Apply to frame
fg_mask = bg_subtractor.apply(frame)
```

#### Morphological Operations

```python
# Create elliptical kernel for better vehicle shape matching
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# Close small holes
fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

# Remove small noise
fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
```

## Creating a Flask App

The Flask application provides a web interface for vehicle detection:

### Application Structure

```
Project7/
├── app.py                      # Main Flask application
├── vehicle_detector.py         # Detection logic
├── requirements.txt            # Dependencies
├── README.md                   # This file
├── .gitignore                 # Git ignore rules
├── templates/                  # HTML templates
│   ├── base.html              # Base template with styling
│   ├── index.html             # Upload page
│   ├── result.html            # Results display
│   └── about.html             # Information page
├── static/
│   └── uploads/               # Uploaded and processed images
├── image_data/                 # Sample images
└── video_data/                 # Sample videos (future)
```

### Flask Application (app.py)

#### 1. Application Configuration

```python
import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from vehicle_detector import VehicleDetector

# Initialize Flask app
app = Flask(__name__)

# Configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key')
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp'}

# Initialize detector
vehicle_detector = VehicleDetector(min_contour_area=500)
```

#### 2. Helper Functions

```python
def allowed_file(filename):
    """Check if file has allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def save_processed_image(image, filename):
    """Save processed image to upload folder."""
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    cv2.imwrite(filepath, image)
    return filepath
```

#### 3. Routes

**Home Route** - Upload form:
```python
@app.route('/')
def index():
    """Display upload form."""
    return render_template('index.html')
```

**Upload Route** - Process image:
```python
@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and detection."""
    # Validate file
    if 'file' not in request.files:
        flash('No file uploaded', 'error')
        return redirect(url_for('index'))
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No file selected', 'error')
        return redirect(url_for('index'))
    
    if not allowed_file(file.filename):
        flash('Invalid file type', 'error')
        return redirect(url_for('index'))
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Detect vehicles
        processed_image, vehicle_count, details = \
            vehicle_detector.detect_vehicles_in_image(filepath)
        
        # Save processed image
        processed_filename = f'processed_{filename}'
        save_processed_image(processed_image, processed_filename)
        
        # Return results
        result_data = {
            'original_image': filename,
            'processed_image': processed_filename,
            'vehicle_count': vehicle_count,
            'details': details
        }
        
        return render_template('result.html', result=result_data)
    
    except Exception as e:
        flash(f'Error: {str(e)}', 'error')
        return redirect(url_for('index'))
```

**About Route**:
```python
@app.route('/about')
def about():
    """Display information about the application."""
    return render_template('about.html')
```

#### 4. Error Handlers

```python
@app.errorhandler(413)
def too_large(e):
    """Handle file too large error."""
    flash('File is too large. Maximum size is 16MB.', 'error')
    return redirect(url_for('index'))

@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors."""
    return render_template('404.html'), 404
```

### HTML Templates

#### Base Template (base.html)

Provides the common layout and styling for all pages:

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{% block title %}{% endblock %} - Vehicle Detection</title>
    <style>
        /* Modern, responsive styling with gradient backgrounds */
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        /* Additional styling for professional appearance */
    </style>
</head>
<body>
    <header>
        <h1>🚗 Vehicle Detection & Counting</h1>
        <p>AI-Powered Vehicle Detection using Computer Vision</p>
    </header>
    
    <nav>
        <ul>
            <li><a href="{{ url_for('index') }}">Home</a></li>
            <li><a href="{{ url_for('about') }}">About</a></li>
        </ul>
    </nav>
    
    <div class="content">
        {% block content %}{% endblock %}
    </div>
    
    <footer>
        <p>&copy; 2024 Vehicle Detection App | Emmanuel Odenyire</p>
    </footer>
</body>
</html>
```

#### Index Page (index.html)

Upload form with drag-and-drop functionality:

```html
{% extends "base.html" %}
{% block content %}
<div class="upload-section">
    <h2>Upload Image for Vehicle Detection</h2>
    <form method="POST" action="{{ url_for('upload_file') }}" 
          enctype="multipart/form-data">
        <input type="file" name="file" accept=".png,.jpg,.jpeg,.bmp" required>
        <button type="submit">🔍 Detect Vehicles</button>
    </form>
</div>

<div class="features">
    <!-- Feature cards showing capabilities -->
</div>
{% endblock %}
```

#### Results Page (result.html)

Displays detection results with statistics:

```html
{% extends "base.html" %}
{% block content %}
<div class="vehicle-count">
    <h3>{{ result.vehicle_count }} Vehicles Detected</h3>
</div>

<div class="images-grid">
    <div class="image-container">
        <h3>Original Image</h3>
        <img src="{{ url_for('uploaded_file', filename=result.original_image) }}">
    </div>
    <div class="image-container">
        <h3>Detected Vehicles</h3>
        <img src="{{ url_for('uploaded_file', filename=result.processed_image) }}">
    </div>
</div>

<div class="details-section">
    <!-- Vehicle statistics and details -->
</div>
{% endblock %}
```

### Running the Flask Application

#### Development Mode

```bash
# Set environment variables
export FLASK_ENV=development
export SECRET_KEY='dev-secret-key'

# Run the application
python app.py
```

#### Production Mode

```bash
# Set environment variables
export FLASK_ENV=production
export SECRET_KEY='your-production-secret-key-here'

# Run with production server
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

#### Access the Application

Open your browser and navigate to:
```
http://localhost:5000
```

### Usage Instructions

1. **Start the Server**: Run `python app.py`
2. **Open Browser**: Navigate to `http://localhost:5000`
3. **Upload Image**: Click "Choose Image" and select a traffic/parking image
4. **View Results**: See detected vehicles with bounding boxes and statistics
5. **Try More**: Upload additional images to analyze different scenes

## Download The Projects Files

### Repository Access

All project files are available in the GitHub repository:

```bash
# Clone the complete repository
git clone https://github.com/eodenyire/DataScienceProjects.git

# Navigate to Project 7
cd DataScienceProjects/Project7

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

### Project Files

**Core Files:**
- `app.py` - Main Flask application
- `vehicle_detector.py` - Vehicle detection logic
- `example_usage.py` - Usage examples and demonstrations
- `requirements.txt` - Python dependencies

**Templates:**
- `templates/base.html` - Base template with styling
- `templates/index.html` - Upload page
- `templates/result.html` - Results display
- `templates/about.html` - Information page

**Configuration:**
- `.gitignore` - Git ignore rules
- `README.md` - This documentation

**Directories:**
- `static/uploads/` - Uploaded and processed images
- `image_data/` - Sample images for testing
- `video_data/` - Sample videos (future feature)

### Direct Download

Individual files can be downloaded from:
```
https://github.com/eodenyire/DataScienceProjects/tree/main/Project7
```

## Usage Examples

### Example 1: Basic Detection

```python
from vehicle_detector import VehicleDetector

# Initialize detector
detector = VehicleDetector(min_contour_area=500)

# Detect vehicles in an image
processed_image, count, details = detector.detect_vehicles_in_image('traffic.jpg')

print(f"Detected {count} vehicles")
print(f"Details: {details}")
```

### Example 2: Custom Settings

```python
# Initialize with custom minimum area
detector = VehicleDetector(min_contour_area=1000)

# Process image
processed_image, count, details = detector.detect_vehicles_in_image('parking.jpg')

# Save result
cv2.imwrite('result.jpg', processed_image)
```

### Example 3: Running Examples

```bash
# Run the example usage script
python example_usage.py
```

This will:
- Create sample traffic images
- Demonstrate various detection settings
- Compare different thresholds
- Show detailed statistics

## Technical Specifications

### System Requirements

- **Python**: 3.8 or higher
- **RAM**: 2GB minimum (4GB recommended)
- **Storage**: 1GB for dependencies
- **CPU**: Multi-core processor recommended
- **GPU**: Not required (CPU-based processing)

### Detection Parameters

- **Minimum Contour Area**: 500 pixels² (default, configurable)
- **Aspect Ratio Range**: 0.5 to 4.0 (vehicle proportions)
- **Edge Detection Thresholds**: 50 (low), 150 (high)
- **Gaussian Blur Kernel**: 5×5
- **Morphological Kernel**: 3×3

### Performance

- **Processing Time**: 1-3 seconds per image (depends on resolution)
- **Supported Image Sizes**: Up to 4000×3000 pixels
- **File Size Limit**: 16MB
- **Accuracy**: High for clear, well-lit images

## Applications

This vehicle detection system can be used for:

- **Traffic Management**: Monitor traffic flow and congestion
- **Parking Systems**: Track parking lot occupancy
- **Highway Monitoring**: Count vehicles on highways and roads
- **Smart Cities**: Integrate with intelligent transportation systems
- **Research**: Transportation and urban planning studies
- **Security**: Surveillance and monitoring applications
- **Analytics**: Traffic pattern analysis and reporting

## Limitations

- Works best with clear, well-lit images
- Detection accuracy depends on image quality and viewing angle
- May have difficulty with heavily occluded or overlapping vehicles
- Designed for static image analysis (video support coming soon)
- Best results with aerial or elevated camera angles
- May detect non-vehicle objects with similar shapes

## Future Enhancements

Planned improvements include:

- [ ] Real-time video processing and streaming
- [ ] Deep learning-based detection (YOLO, SSD)
- [ ] Vehicle type classification (car, truck, bus, motorcycle)
- [ ] Speed estimation from video
- [ ] License plate detection and recognition
- [ ] Traffic density heatmaps
- [ ] Batch image processing
- [ ] REST API for third-party integration
- [ ] Database storage for historical analysis
- [ ] Multi-camera support
- [ ] Mobile application (iOS/Android)

## Troubleshooting

### Common Issues

**1. Module Not Found Error**
```bash
Solution: Install dependencies
pip install -r requirements.txt
```

**2. Upload Folder Permission Error**
```bash
Solution: Create folder manually
mkdir -p static/uploads
chmod 755 static/uploads
```

**3. OpenCV Import Error**
```bash
Solution: Install OpenCV
pip install opencv-python opencv-contrib-python
```

**4. File Too Large Error**
```
Solution: Compress image or adjust MAX_CONTENT_LENGTH in app.py
```

**5. Port Already in Use**
```bash
Solution: Change port or kill existing process
python app.py  # Will use port 5000 by default
# Or specify a different port:
export PORT=8000
python app.py
```

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/NewFeature`)
3. Commit your changes (`git commit -m 'Add NewFeature'`)
4. Push to the branch (`git push origin feature/NewFeature`)
5. Open a Pull Request

## License

This project is part of the DataScienceProjects repository by Emmanuel Odenyire.

## Acknowledgments

- **OpenCV Community**: For the excellent computer vision library
- **Flask**: For the lightweight and powerful web framework
- **NumPy**: For efficient numerical computations
- **Python Community**: For comprehensive documentation and support

## Contact & Support

**Developer**: Emmanuel Odenyire  
**Email**: eodenyire@gmail.com  
**Project**: Data Science Projects Series - Project 7

For questions, issues, or suggestions:
- Open an issue on GitHub
- Check the troubleshooting section
- Review example usage scripts

---

**Note**: This is an educational project demonstrating computer vision applications in transportation and traffic analysis. For production deployment, consider implementing additional error handling, security measures, and performance optimizations.
