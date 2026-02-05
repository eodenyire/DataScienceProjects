# Project 3: Image Watermarking App

## Introduction to Image Watermarking App

The Image Watermarking App is a professional web application that enables users to protect their digital images by adding customizable watermarks. Built with Flask and powered by OpenCV and Pillow, this application provides an intuitive interface for adding both text and image-based watermarks to your photographs, artwork, or any digital content.

### Key Features

- 📝 **Text Watermarks**: Add custom text with adjustable font size, color, and opacity
- 🖼️ **Image Watermarks**: Apply logos or images as watermarks with size control
- 🎯 **Flexible Positioning**: Place watermarks in any corner or center of your image
- ✨ **Opacity Control**: Adjust transparency for subtle or prominent watermarks
- ⚡ **Fast Processing**: Get watermarked images in seconds
- 💾 **Easy Download**: Download protected images with one click
- 🌐 **User-Friendly Interface**: Clean, modern web interface with tabbed navigation
- 🔒 **Secure**: File validation and size limits for security

## Importing Libraries and Logo

### Required Libraries

The project uses the following Python libraries:

```python
# Web Framework
Flask==2.3.2
Werkzeug>=3.0.3

# Image Processing
opencv-python>=4.8.1.78  # For image watermarking
Pillow>=10.3.0          # For advanced text rendering
numpy>=1.24.0           # For array operations
```

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/eodenyire/DataScienceProjects.git
cd DataScienceProjects/Project3
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Create necessary directories:**
```bash
mkdir -p static/uploads static/logos image_data
```

### Preparing Logos

To use image watermarks:

1. Place your logo files in the `static/logos/` folder
2. Supported formats: PNG (recommended for transparency), JPG, JPEG
3. For best results, use PNG images with transparent backgrounds
4. The app will automatically detect and display available logos

## Create Text and Image WaterMark

### Text Watermark Implementation

The core watermarking functionality is implemented in `image_watermarker.py`:

#### Key Features of Text Watermarking

```python
from image_watermarker import ImageWatermarker

# Initialize watermarker
watermarker = ImageWatermarker()

# Add text watermark
watermarker.add_text_watermark(
    image_path="input.jpg",
    text="© 2024 Your Name",
    output_path="output.jpg",
    position='bottom-right',  # Position options
    font_size=40,             # Font size in pixels
    color=(255, 255, 255),    # RGB color tuple (white)
    opacity=0.7               # Transparency (0.0 to 1.0)
)
```

**Position Options:**
- `top-left` - Top left corner
- `top-right` - Top right corner  
- `bottom-left` - Bottom left corner
- `bottom-right` - Bottom right corner (default)
- `center` - Center of the image

**Text Customization:**
- **Font Size**: Adjustable from 12px to 120px
- **Color**: Any RGB color using color picker
- **Opacity**: From 0.0 (invisible) to 1.0 (opaque)
- **Font**: Uses system fonts (DejaVu, Liberation, Helvetica, Arial)

### Image Watermark Implementation

Add logos or images as watermarks:

```python
from image_watermarker import ImageWatermarker

watermarker = ImageWatermarker()

# Add image watermark
watermarker.add_image_watermark(
    image_path="input.jpg",
    logo_path="logo.png",
    output_path="output.jpg",
    position='bottom-right',
    size_ratio=0.15,          # 15% of image width
    opacity=0.6               # Transparency
)
```

**Size Control:**
- `size_ratio`: Controls logo size relative to image width (0.05 to 0.5)
- Maintains aspect ratio automatically
- Recommended: 0.10 to 0.20 for subtle watermarks

**Opacity Control:**
- Supports images with alpha channels (PNG transparency)
- Adjustable opacity for blending with the background
- Perfect for semi-transparent logo overlays

### Advanced Features

#### Automatic Font Selection

The app tries multiple system fonts and falls back gracefully:
- DejaVu Sans Bold (Linux)
- Liberation Sans Bold (Linux)
- Helvetica (macOS)
- Arial (Windows)
- Default bitmap font (fallback)

#### Transparent Image Support

- Handles PNG images with alpha channels
- Proper alpha blending for smooth watermarks
- Maintains transparency in logos

#### Smart Positioning

- Automatic margin calculation (20px from edges)
- Prevents watermarks from going out of bounds
- Centers watermarks properly

## Create the App

The Flask application (`app.py`) provides a complete web interface:

### Application Structure

```
Project3/
├── app.py                      # Flask application
├── image_watermarker.py        # Watermarking logic
├── requirements.txt            # Python dependencies
├── README.md                   # Documentation
├── example_usage.py            # Usage examples
├── .gitignore                  # Git ignore rules
├── templates/                  # HTML templates
│   ├── base.html              # Base template
│   ├── index.html             # Upload page
│   ├── result.html            # Results page
│   └── about.html             # About page
├── static/
│   ├── uploads/               # Temporary upload folder
│   └── logos/                 # Logo storage
└── image_data/                # Sample images (optional)
```

### Flask Routes

#### 1. Home Route (`/`)
- Displays upload form
- Shows available logos
- Tabbed interface for text/image watermarks

#### 2. Text Watermark Route (`/watermark/text`)
- Handles file upload
- Processes text watermark
- Returns result page

#### 3. Image Watermark Route (`/watermark/image`)
- Handles file and logo upload
- Processes image watermark
- Returns result page

#### 4. Download Route (`/download/<filename>`)
- Serves watermarked images
- Enables file download

#### 5. About Route (`/about`)
- Application information
- Usage instructions
- Technical details

### Security Features

```python
# File upload security
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    """Validate file extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
```

### Error Handling

- File validation (type, size, existence)
- Graceful error messages using Flask's flash system
- Try-catch blocks for image processing
- Redirect to home page on errors

## Deploying the App in Heroku

### Local Development

1. **Set environment variables:**
```bash
export FLASK_ENV=development
export SECRET_KEY='dev-secret-key'
```

2. **Run the application:**
```bash
python app.py
```

3. **Access the application:**
   - Open browser: `http://localhost:5000`

### Production Deployment

#### Using Gunicorn (Recommended)

1. **Install Gunicorn:**
```bash
pip install gunicorn
```

2. **Set production environment:**
```bash
export FLASK_ENV=production
export SECRET_KEY='your-secure-secret-key-here'
```

3. **Run with Gunicorn:**
```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

#### Deploying on Heroku

1. **Create a `Procfile`:**
```
web: gunicorn app:app
```

2. **Create `runtime.txt`:**
```
python-3.9.13
```

3. **Deploy to Heroku:**
```bash
# Login to Heroku
heroku login

# Create app
heroku create your-watermark-app

# Set secret key
heroku config:set SECRET_KEY='your-secure-secret-key'
heroku config:set FLASK_ENV=production

# Deploy
git push heroku main

# Open app
heroku open
```

#### Docker Deployment

1. **Create `Dockerfile`:**
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# Copy application
COPY . .

# Create directories
RUN mkdir -p static/uploads static/logos image_data

# Expose port
EXPOSE 5000

# Run application
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

2. **Build and run:**
```bash
# Build image
docker build -t watermark-app .

# Run container
docker run -p 5000:5000 -e SECRET_KEY='your-secret-key' watermark-app
```

#### Environment Configuration

For production, always set:
- `SECRET_KEY`: Strong random secret key
- `FLASK_ENV`: Set to `production`
- Use HTTPS with SSL certificate
- Set up proper logging
- Configure file cleanup jobs

## Download The Projects Files

### Repository Access

All project files are available in the GitHub repository:

```bash
# Clone the repository
git clone https://github.com/eodenyire/DataScienceProjects.git

# Navigate to Project3
cd DataScienceProjects/Project3

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

### File Descriptions

- **`app.py`**: Main Flask application with routes and handlers
- **`image_watermarker.py`**: Core watermarking module with classes and functions
- **`requirements.txt`**: Python package dependencies
- **`example_usage.py`**: Code examples for programmatic usage
- **`templates/`**: HTML templates for the web interface
  - `base.html`: Base template with header, navigation, and styling
  - `index.html`: Home page with upload forms
  - `result.html`: Results display page
  - `about.html`: Application information page
- **`.gitignore`**: Files to exclude from version control

## Usage Guide

### Web Interface Usage

#### Adding Text Watermark

1. **Navigate to Home Page**
   - Open `http://localhost:5000` in your browser

2. **Select Text Watermark Tab**
   - Click on "📝 Text Watermark" tab

3. **Upload Image**
   - Click "Choose File" and select your image
   - Supported: PNG, JPG, JPEG (max 16MB)

4. **Customize Text Watermark**
   - Enter watermark text (e.g., "© 2024 Your Name")
   - Select position (bottom-right, top-left, etc.)
   - Adjust font size (12-120px)
   - Choose text color using color picker
   - Set opacity (0.0 - 1.0)

5. **Apply Watermark**
   - Click "Add Text Watermark" button
   - Wait for processing (usually < 2 seconds)

6. **Download Result**
   - View comparison of original and watermarked images
   - Click "📥 Download Watermarked Image"

#### Adding Image Watermark

1. **Select Image Watermark Tab**
   - Click on "🖼️ Image Watermark" tab

2. **Upload Image**
   - Click "Choose File" and select your image

3. **Upload or Select Logo**
   - Option 1: Upload a new logo file
   - Option 2: Select from available logos

4. **Customize Image Watermark**
   - Select position
   - Adjust logo size (5-50% of image width)
   - Set opacity (0.0 - 1.0)

5. **Apply and Download**
   - Click "Add Image Watermark"
   - Download watermarked image

### Programmatic Usage

Use the watermarking module in your Python code:

```python
from image_watermarker import ImageWatermarker

# Initialize
watermarker = ImageWatermarker()

# Text watermark
watermarker.add_text_watermark(
    image_path="my_photo.jpg",
    text="© 2024 My Name",
    output_path="watermarked.jpg",
    position='bottom-right',
    font_size=40,
    color=(255, 255, 255),
    opacity=0.7
)

# Image watermark
watermarker.add_image_watermark(
    image_path="my_photo.jpg",
    logo_path="my_logo.png",
    output_path="watermarked.jpg",
    position='bottom-right',
    size_ratio=0.15,
    opacity=0.6
)
```

For more examples, see `example_usage.py`.

## Technical Details

### Image Processing Pipeline

#### Text Watermarking

1. **Load Image**: Read image using PIL for better text rendering
2. **Create Overlay**: Create transparent RGBA overlay
3. **Font Loading**: Try system fonts, fall back to default
4. **Text Rendering**: Render text with antialiasing
5. **Position Calculation**: Calculate position with margins
6. **Alpha Compositing**: Blend overlay with original image
7. **Save Result**: Convert to RGB and save

#### Image Watermarking

1. **Load Images**: Read main image and logo using OpenCV
2. **Format Conversion**: Ensure BGRA format for transparency
3. **Resize Logo**: Scale logo based on size_ratio
4. **Position Calculation**: Calculate placement coordinates
5. **Alpha Blending**: Blend logo with background using opacity
6. **Save Result**: Write final image to disk

### Color Formats

- **RGB**: Used for colors (255, 255, 255)
- **RGBA**: RGB + Alpha channel for transparency
- **BGR/BGRA**: OpenCV's color format

### Coordinate System

- Origin (0, 0) is top-left corner
- X increases rightward
- Y increases downward
- Margins: 20px from edges

## Best Practices

### For Best Results

1. **Image Quality**
   - Use high-resolution images
   - Avoid heavily compressed JPEGs
   - Ensure good lighting and contrast

2. **Text Watermarks**
   - Keep text concise
   - Use contrasting colors
   - Adjust opacity for subtle effect
   - Test different positions

3. **Image Watermarks**
   - Use PNG logos with transparency
   - Keep logo size proportional (10-20%)
   - Position in corner for non-intrusive mark
   - Use semi-transparent logos (0.5-0.7 opacity)

4. **Security**
   - Don't watermark original files
   - Keep backups of unwatermarked images
   - Use strong watermarks for valuable content

## Applications

This watermarking app is perfect for:

- 📸 **Photographers**: Protect portfolio images with copyright notices
- 🎨 **Digital Artists**: Sign artwork with custom text or logo
- 💼 **Businesses**: Brand marketing materials and product photos
- 📱 **Social Media**: Add attribution before sharing content
- 🏢 **Corporate**: Watermark presentations and documents
- 🎓 **Education**: Mark educational materials and course content
- 🌐 **Content Creators**: Protect blog images and graphics
- 📰 **News Organizations**: Watermark photojournalism
- 🏪 **E-commerce**: Protect product photography
- 🎬 **Video Production**: Watermark video stills and thumbnails

## Troubleshooting

### Common Issues

1. **"Model not found" or Import Errors**
   ```
   Solution: Install dependencies: pip install -r requirements.txt
   ```

2. **Upload Folder Permission Error**
   ```
   Solution: Create folder manually: mkdir -p static/uploads
   ```

3. **Font Not Found / Default Font Used**
   ```
   This is normal - the app falls back to default font
   Solution: Install system fonts or specify custom font path
   ```

4. **Large File Upload Error**
   ```
   Solution: Compress image or increase MAX_CONTENT_LENGTH in app.py
   ```

5. **Port Already in Use**
   ```
   Solution: Change port in app.py or kill process:
   lsof -ti:5000 | xargs kill -9
   ```

### Debug Mode

Enable debug mode for development:
```bash
export FLASK_ENV=development
python app.py
```

## Performance

### Speed

- Text watermarking: ~0.5-2 seconds per image
- Image watermarking: ~1-3 seconds per image
- Depends on image size and system resources

### Memory Usage

- Low memory footprint (~50-200MB)
- Efficient image processing with OpenCV
- Automatic cleanup of processed files

### Scalability

For high-volume processing:
- Use multiple Gunicorn workers
- Implement task queue (Celery + Redis)
- Add caching layer
- Use CDN for static files
- Implement batch processing API

## Future Enhancements

Planned features:

- [ ] Batch watermarking for multiple images
- [ ] More font options and custom font upload
- [ ] Advanced text effects (shadow, outline, gradient)
- [ ] Watermark templates and presets
- [ ] Drag-and-drop positioning interface
- [ ] Real-time preview before processing
- [ ] Video watermarking support
- [ ] REST API for programmatic access
- [ ] User accounts and watermark history
- [ ] Cloud storage integration (S3, Drive)
- [ ] Mobile app (iOS/Android)
- [ ] Bulk processing via ZIP upload
- [ ] Watermark removal detection
- [ ] Advanced positioning with pixel coordinates

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is part of the DataScienceProjects repository.

## Acknowledgments

- **Flask Team**: For the excellent web framework
- **OpenCV**: For powerful image processing capabilities
- **Pillow**: For advanced image manipulation
- **Bootstrap Icons**: For beautiful icons (if used)

## Author

Developed as part of the Data Science Projects series.

## Contact & Support

For questions, issues, or suggestions:
- Open an issue on GitHub
- Check existing documentation
- Review troubleshooting section

---

**Made with ❤️ for the developer community**
