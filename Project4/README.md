# Project 4: Traffic Sign Classification using Python

## Introduction to Traffic Sign Classification

The Traffic Sign Classification project is an advanced AI-powered web application that uses deep learning and computer vision techniques to automatically recognize and classify traffic signs. This tool leverages state-of-the-art neural networks and transfer learning with MobileNetV2 to identify 43 different types of traffic signs from the German Traffic Sign Recognition Benchmark (GTSRB) dataset.

### Key Features
- 🎯 **High Accuracy**: Deep learning model with excellent classification performance
- ⚡ **Fast Processing**: Real-time classification with confidence scores in seconds
- 🧠 **Transfer Learning**: Leverages pre-trained MobileNetV2 architecture
- 🌐 **Web Interface**: Easy-to-use Flask-based web application
- 📊 **Top-3 Predictions**: Shows most confident predictions with probability scores
- 🚦 **43 Sign Classes**: Supports comprehensive traffic sign recognition

### Developer
**Emmanuel Odenyire**

## Importing the Data and Libraries

### Required Libraries
The project uses the following Python libraries:

```python
# Web Framework
Flask==2.3.2
Werkzeug>=3.0.3

# Deep Learning & Neural Networks
tensorflow>=2.13.0

# Image Processing & Computer Vision
opencv-python>=4.8.1.78
Pillow>=10.3.0
numpy>=1.24.0

# Machine Learning Utilities
scikit-learn>=1.3.0
```

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/eodenyire/DataScienceProjects.git
cd DataScienceProjects/Project4
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the application:**
```bash
python app.py
```

4. **Access the application:**
   - Open your browser and navigate to `http://localhost:5000`

## Image Processing

The image processing pipeline includes several crucial steps to prepare traffic sign images for classification:

### Preprocessing Steps

1. **Image Loading**
   - Accept PNG, JPG, and JPEG formats
   - Load images using PIL/Pillow library
   - Convert to RGB color space

2. **Image Resizing**
   - Resize all images to 224×224 pixels
   - Maintain aspect ratio with padding if needed
   - Match MobileNetV2 input requirements

3. **Normalization**
   - Apply MobileNetV2 preprocessing function
   - Normalize pixel values to appropriate range
   - Ensure consistent input format

4. **Data Augmentation (Training)**
   - Rotation: ±15 degrees
   - Width/Height shift: ±10%
   - Shear transformation: 10%
   - Zoom: ±10%
   - No horizontal flip (traffic signs have specific orientations)

### Key Functions

```python
preprocess_image(image_path)
# - Loads and preprocesses a single image
# - Returns normalized array ready for prediction

predict(image_path)
# - Preprocesses image and runs model inference
# - Returns prediction with confidence scores
```

## Creating and Testing the Model

### Model Architecture

The classifier uses **transfer learning** with MobileNetV2 as the base model. MobileNetV2 is a lightweight deep learning architecture optimized for mobile and embedded vision applications.

#### Architecture Components:

1. **Base Model: MobileNetV2**
   - Pre-trained on ImageNet dataset
   - Frozen weights for feature extraction
   - Input shape: 224×224×3

2. **Custom Top Layers:**
   - Global Average Pooling 2D
   - Dropout (0.3) - Regularization
   - Dense (256 neurons, ReLU activation)
   - Dropout (0.3) - Regularization
   - Dense (128 neurons, ReLU activation)
   - Dropout (0.2) - Regularization
   - Dense (43 neurons, Softmax activation) - Output layer

3. **Compilation:**
   - Optimizer: Adam (learning rate: 0.001)
   - Loss: Categorical Cross-Entropy
   - Metrics: Accuracy

### Training Process

```python
# Initialize classifier
classifier = TrafficSignClassifier(num_classes=43, img_size=(224, 224))

# Build model with transfer learning
classifier.build_model(use_pretrained=True)

# Train on dataset
history = classifier.train_model(
    train_data_dir='path/to/train',
    val_data_dir='path/to/validation',
    epochs=20,
    batch_size=32
)

# Save trained model
classifier.save_model('models/traffic_sign_model.h5')
```

### Callbacks and Training Features:

- **Early Stopping**: Stops training when validation loss stops improving
- **Learning Rate Reduction**: Reduces learning rate when loss plateaus
- **Best Model Restoration**: Saves best performing model

### Traffic Sign Classes (43 Categories)

The model recognizes the following traffic sign types:

#### Speed Limits:
- Speed limit (20km/h, 30km/h, 50km/h, 60km/h, 70km/h, 80km/h, 100km/h, 120km/h)
- End of speed limit (80km/h)
- End of all speed and passing limits

#### Warning Signs:
- General caution
- Dangerous curve (left/right)
- Double curve
- Bumpy road
- Slippery road
- Road narrows on the right
- Road work
- Traffic signals
- Pedestrians
- Children crossing
- Bicycles crossing
- Beware of ice/snow
- Wild animals crossing

#### Prohibitory Signs:
- No passing
- No passing for vehicles over 3.5 metric tons
- No vehicles
- Vehicles over 3.5 metric tons prohibited
- No entry
- End of no passing
- End of no passing by vehicles over 3.5 metric tons

#### Mandatory Signs:
- Right-of-way at the next intersection
- Priority road
- Yield
- Stop
- Turn right ahead
- Turn left ahead
- Ahead only
- Go straight or right
- Go straight or left
- Keep right
- Keep left
- Roundabout mandatory

## Creating Model for Test Set

### Prediction Pipeline

The application provides a complete pipeline for classifying traffic signs:

1. **Upload Interface**
   - User-friendly web form
   - Drag-and-drop or click to upload
   - Real-time image preview
   - File type validation

2. **Image Processing**
   - Automatic preprocessing
   - Resizing and normalization
   - Format conversion

3. **Model Inference**
   - Load uploaded image
   - Preprocess for model input
   - Run prediction through neural network
   - Calculate confidence scores

4. **Result Display**
   - Primary prediction with confidence percentage
   - Top 3 most likely classes
   - Visual confidence bar
   - Color-coded results

### Prediction Function

```python
def predict(self, image_path):
    """
    Predict the traffic sign class for an image.
    
    Returns:
        dict: {
            'predicted_class': str,
            'confidence': float,
            'top_3': [
                {'class': str, 'confidence': float},
                ...
            ]
        }
    """
```

### Example Usage

```python
from traffic_sign_classifier import TrafficSignClassifier

# Initialize classifier
classifier = TrafficSignClassifier()
classifier.build_model(use_pretrained=True)

# Or load pre-trained model
classifier.load_model('models/traffic_sign_model.h5')

# Make prediction
result = classifier.predict('path/to/traffic_sign.jpg')

print(f"Predicted: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.2%}")
print("\nTop 3 Predictions:")
for pred in result['top_3']:
    print(f"  - {pred['class']}: {pred['confidence']:.2%}")
```

## Usage

### Running the Application

1. **Start the Flask server:**
```bash
# For development (with debug mode)
export FLASK_ENV=development
export SECRET_KEY='dev-secret-key'
python app.py

# For production
export FLASK_ENV=production
export SECRET_KEY='your-secure-production-key'
python app.py
```

2. **Access the web interface:**
   - Navigate to `http://localhost:5000`
   - Upload a traffic sign image
   - View classification results

### Expected Output

When you upload a traffic sign image, you'll receive:

- **Predicted Class**: The most likely traffic sign type
- **Confidence Score**: Percentage confidence (0-100%)
- **Top 3 Predictions**: Alternative classifications with probabilities
- **Visual Confidence Bar**: Graphical representation of confidence
- **Uploaded Image Display**: Preview of the classified image

## Creating the Flask App

The Flask application (`app.py`) provides a web interface for the classifier:

### Features

1. **File Upload Handling**
   - Accepts PNG, JPG, and JPEG formats
   - File size limit: 16MB
   - Secure filename processing
   - Input validation

2. **Image Processing Pipeline**
   - Validate uploaded file
   - Save to uploads directory
   - Perform classification
   - Generate results

3. **Result Display**
   - Show uploaded image
   - Display prediction with confidence
   - Show top 3 predictions
   - Provide confidence visualization

### Routes

- `/` - Home page with upload form
- `/upload` (POST) - Handle file upload and classification
- `/about` - Information about the application and methodology

## How It Works

1. **Upload Phase**: User uploads a traffic sign image through the web interface
2. **Preprocessing**: System automatically resizes image to 224×224 and normalizes
3. **Feature Extraction**: MobileNetV2 base model extracts visual features from the image
4. **Classification**: Custom dense layers process features and classify the sign type
5. **Confidence Scoring**: Softmax layer outputs probability distribution across 43 classes
6. **Result Display**: Top predictions shown with confidence percentages and visual indicators

## Technical Details

### Deep Learning Architecture

**Transfer Learning with MobileNetV2:**

MobileNetV2 is a convolutional neural network architecture designed for mobile and resource-constrained environments. Key characteristics:

- **Inverted Residual Structure**: Efficient feature extraction
- **Linear Bottlenecks**: Prevents information loss
- **Lightweight Design**: Suitable for real-time applications
- **Pre-trained Weights**: Leverages ImageNet knowledge

**Why Transfer Learning?**
- Reduces training time significantly
- Requires less training data
- Leverages features learned from millions of images
- Improves generalization and accuracy

### Model Performance

The model architecture is designed for:
- **Accuracy**: High classification precision
- **Speed**: Fast inference time (<1 second per image)
- **Robustness**: Handles various lighting and angles
- **Reliability**: Confidence scores indicate prediction quality

## Applications

- **Autonomous Vehicles**: Real-time traffic sign recognition for self-driving cars
- **Driver Assistance Systems**: Alert drivers about upcoming traffic signs
- **Traffic Management**: Automated sign inventory and monitoring systems
- **Navigation Applications**: Enhanced routing with traffic sign awareness
- **Road Safety Research**: Analysis of sign visibility, placement, and effectiveness
- **Educational Tools**: Teaching traffic rules and AI/ML concepts
- **Smart City Infrastructure**: Integration with intelligent transportation systems

## Project Structure

```
Project4/
├── app.py                          # Flask web application
├── traffic_sign_classifier.py      # Classification model and training
├── requirements.txt                # Python dependencies
├── README.md                       # This documentation
├── IMPLEMENTATION_SUMMARY.md       # Implementation summary
├── templates/                      # HTML templates
│   ├── base.html                  # Base template with styling
│   ├── index.html                 # Home page (upload form)
│   ├── result.html                # Results display page
│   └── about.html                 # About/documentation page
├── static/
│   └── uploads/                   # Uploaded images storage
└── models/
    └── traffic_sign_model.h5      # Trained model (if available)
```

## Limitations and Considerations

- Image quality and lighting conditions can affect accuracy
- Model trained on GTSRB (German signs) may need retraining for other countries
- Occluded or damaged signs may be harder to classify
- Should be used as part of a comprehensive system, not in isolation
- Real-world deployment requires extensive testing and validation

## Future Enhancements

- Support for additional international traffic sign standards (US, UK, etc.)
- Real-time video stream processing for continuous monitoring
- Mobile application development (iOS/Android)
- Multi-sign detection in single images
- Weather and lighting condition adaptations
- Geolocation-based sign database integration
- API endpoint for third-party integrations
- Model quantization for edge device deployment
- Explainable AI features (attention maps, gradient visualization)

## Download The Project Files

All project files are available in the repository:
- Source code: `app.py`, `traffic_sign_classifier.py`
- Configuration: `requirements.txt`
- Templates: HTML files in `templates/` directory
- Documentation: `README.md`, `IMPLEMENTATION_SUMMARY.md`

## License

This project is part of the DataScienceProjects repository.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For questions or issues, please open an issue in the GitHub repository.

---

**Developed by Emmanuel Odenyire** as part of the Data Science Projects series.
