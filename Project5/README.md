# Project 5: Text Extraction from Images Application

## Introduction

The Text Extraction from Images Application is a powerful web-based tool that uses Optical Character Recognition (OCR) technology to extract text from images. Built with Flask and powered by Tesseract OCR, this application provides an intuitive interface for converting visual text content into editable, searchable digital text.

**Developed by Emmanuel Odenyire**  
**Email:** eodenyire@gmail.com

### Key Features

- 📝 **Advanced OCR**: Powered by Tesseract, one of the most accurate open-source OCR engines
- 🌍 **Multi-Language Support**: Extract text in 10+ languages including English, Spanish, French, German, and more
- 🖼️ **Image Preprocessing**: Optional preprocessing to improve accuracy for low-quality images
- 📊 **Detailed Statistics**: Word count, character count, line count, and confidence scores
- ⚡ **Fast Processing**: Extract text from images in seconds
- 💾 **Easy Export**: Copy text to clipboard or download processed images
- 🎯 **Confidence Scores**: View OCR confidence for each recognized word
- 🌐 **Web Interface**: User-friendly Flask-based web application

## Importing Libraries and Data

### Required Libraries

The project uses the following Python libraries:

```python
# Web Framework
Flask==2.3.2
Werkzeug>=3.0.3

# Image Processing & Computer Vision
opencv-python>=4.8.1.78
Pillow>=10.3.0
numpy>=1.24.0

# OCR Engine
pytesseract>=0.3.10
```

### System Requirements

Before installing the Python libraries, you need to install Tesseract OCR on your system:

#### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
sudo apt-get install libtesseract-dev

# Install additional language packs (optional)
sudo apt-get install tesseract-ocr-fra  # French
sudo apt-get install tesseract-ocr-spa  # Spanish
sudo apt-get install tesseract-ocr-deu  # German
```

#### macOS
```bash
brew install tesseract

# Install additional languages (optional)
brew install tesseract-lang
```

#### Windows
1. Download the installer from [Tesseract GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
2. Run the installer and note the installation path
3. Add Tesseract to your system PATH

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/eodenyire/DataScienceProjects.git
cd DataScienceProjects/Project5
```

2. **Install Python dependencies:**
```bash
pip install -r requirements.txt
```

3. **Create necessary directories:**
```bash
mkdir -p static/uploads image_data
```

4. **Verify Tesseract installation:**
```bash
tesseract --version
```

## Extracting the Text from Image

### Core Text Extraction Module

The core functionality is implemented in `text_extractor.py`, which provides the `TextExtractor` class with the following methods:

#### Basic Text Extraction

```python
from text_extractor import TextExtractor

# Initialize the extractor
extractor = TextExtractor()

# Extract text from an image
text = extractor.extract_text("image.jpg")
print(text)
```

#### Text Extraction with Preprocessing

For better accuracy with low-quality images:

```python
# Extract text with preprocessing
text = extractor.extract_text("image.jpg", preprocess=True)
```

The preprocessing pipeline includes:
1. **Grayscale Conversion**: Converts color image to grayscale
2. **Otsu's Thresholding**: Automatic threshold determination for binarization
3. **Denoising**: Removes noise using fast non-local means denoising
4. **Binary Image**: Creates high-contrast black and white image

#### Text Extraction with Confidence Scores

Get detailed information about the OCR process:

```python
# Extract text with confidence scores
result = extractor.extract_text_with_confidence("image.jpg")

print(f"Extracted Text: {result['text']}")
print(f"Average Confidence: {result['average_confidence']}%")

# Access individual word information
for word in result['words']:
    print(f"Word: {word['text']}, Confidence: {word['confidence']}%")
    print(f"Position: ({word['position']['x']}, {word['position']['y']})")
```

#### Multi-Language Support

```python
# Extract French text
text_fr = extractor.extract_text("french_text.jpg", language='fra')

# Extract Spanish text
text_es = extractor.extract_text("spanish_text.jpg", language='spa')

# Extract German text
text_de = extractor.extract_text("german_text.jpg", language='deu')
```

Supported languages include: English (`eng`), French (`fra`), German (`deu`), Spanish (`spa`), Italian (`ita`), Portuguese (`por`), Russian (`rus`), Chinese Simplified (`chi_sim`), Japanese (`jpn`), Arabic (`ara`), and many more.

## Modifying the Extractor

### Customizing Tesseract Path

If Tesseract is not in your system PATH:

```python
# Specify custom Tesseract path
extractor = TextExtractor(tesseract_cmd='/usr/local/bin/tesseract')
```

### Customizing Preprocessing

You can modify the preprocessing in `text_extractor.py` to implement custom image enhancement techniques tailored to your specific use case.

## Creating the Extractor App

The Flask application (`app.py`) provides a complete web interface for text extraction with routes for uploading images, extracting text, viewing results, and downloading processed files.

### Application Structure

```
Project5/
├── app.py                      # Flask application
├── text_extractor.py           # OCR module
├── requirements.txt            # Python dependencies
├── README.md                   # Documentation
├── example_usage.py            # Usage examples
├── templates/                  # HTML templates
│   ├── base.html              # Base template
│   ├── index.html             # Upload page
│   ├── result.html            # Results page
│   └── about.html             # About page
├── static/
│   └── uploads/               # Temporary upload folder
└── image_data/                # Sample images
```

## Running the Extractor App

### Development Mode

1. **Set environment variables:**
```bash
export FLASK_DEBUG=1
export SECRET_KEY='dev-secret-key'
```

2. **Run the application:**
```bash
python app.py
```

3. **Access the application:**
   - Open your browser and navigate to `http://localhost:5000`

### Using the Web Application

1. **Navigate to Home Page** - Open `http://localhost:5000` in your browser
2. **Upload Image** - Click "Choose File" and select an image containing text
3. **Configure Options** - Select language, enable preprocessing, enable confidence scores
4. **Extract Text** - Click "Extract Text" button
5. **View Results** - See extracted text, statistics, and confidence scores

### Production Deployment

For production, use Gunicorn:

```bash
pip install gunicorn
export FLASK_DEBUG=0
export SECRET_KEY='your-secure-secret-key-here'
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## Download The Projects Files

### Repository Access

All project files are available in the GitHub repository:

```bash
# Clone the repository
git clone https://github.com/eodenyire/DataScienceProjects.git

# Navigate to Project5
cd DataScienceProjects/Project5

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

### Programmatic Usage Examples

See `example_usage.py` for comprehensive examples:

```bash
# Run example scripts
python example_usage.py
```

## Applications

This text extraction application is perfect for:

- 📄 **Document Digitization**: Convert scanned documents to editable text
- 📸 **Photo Text Extraction**: Extract text from photographs and screenshots
- 💳 **Card Recognition**: Extract information from business cards and ID cards
- 📊 **Data Entry Automation**: Automate data entry from printed forms
- 🔍 **Content Indexing**: Make image-based content searchable
- ♿ **Accessibility**: Convert visual content to text for screen readers
- 📚 **Archive Digitization**: Convert historical documents to digital text
- 🌐 **Translation**: Extract text for translation purposes

## Best Practices

### For Best OCR Results

1. **Image Quality** - Use high-resolution images (300 DPI or higher)
2. **Text Characteristics** - Clear, legible fonts work best
3. **Lighting and Background** - Even, consistent lighting with plain background
4. **Preprocessing** - Enable preprocessing for scanned documents
5. **Language Selection** - Always select the correct language

## Troubleshooting

### Common Issues

1. **"Tesseract not found" Error**
   - Install Tesseract OCR and ensure it's in your PATH

2. **Poor OCR Accuracy**
   - Enable preprocessing, check image quality, ensure correct language

3. **Upload Folder Permission Error**
   - Create folder manually: `mkdir -p static/uploads`

## Performance

- Simple text extraction: ~1-2 seconds per image
- Extraction with preprocessing: ~2-4 seconds per image
- High-quality digital text: 95-99% accuracy
- Scanned documents: 85-95% accuracy (with preprocessing)

## Future Enhancements

- Batch processing for multiple images
- PDF support with multi-page extraction
- Handwriting recognition with custom models
- API endpoint for programmatic access

## Author

**Emmanuel Odenyire**  
Email: eodenyire@gmail.com

Developed as part of the Data Science Projects series.

---

**Made with ❤️ for the developer community**
