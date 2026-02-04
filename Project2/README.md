# Project 2: Dog Breed Prediction Flask App

## Introduction

The Dog Breed Prediction Flask App is an intelligent web application that uses deep learning and computer vision to automatically identify dog breeds from photographs. Built with Flask and TensorFlow, this application leverages transfer learning with the MobileNetV2 architecture to provide accurate and fast breed classification.

### Key Features

- 🤖 **AI-Powered Classification**: Uses deep learning with MobileNetV2 architecture
- ⚡ **Fast Predictions**: Get results in seconds with high confidence scores
- 🎯 **Top-3 Predictions**: Shows the three most likely breeds with confidence percentages
- 🌐 **Web Interface**: Easy-to-use Flask-based web application with modern UI
- 📱 **Responsive Design**: Works seamlessly on desktop and mobile devices
- 🔒 **Secure Upload**: File validation and size limits for security

## Importing The Data and Libraries

### Required Libraries

The project uses the following Python libraries:

```python
# Web Framework
Flask==2.3.2
Werkzeug>=3.0.3

# Deep Learning & Machine Learning
tensorflow>=2.13.0
numpy>=1.24.0
scikit-learn>=1.3.0

# Image Processing
Pillow>=10.3.0
opencv-python>=4.8.1.78
matplotlib>=3.7.0
```

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/eodenyire/DataScienceProjects.git
cd DataScienceProjects/Project2
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Create necessary directories:**
```bash
mkdir -p static/uploads models image_data
```

### Dataset

For training a custom model, organize your dataset as follows:

```
image_data/
├── train/
│   ├── beagle/
│   ├── boxer/
│   ├── bulldog/
│   ├── chihuahua/
│   ├── golden_retriever/
│   ├── german_shepherd/
│   ├── labrador/
│   ├── poodle/
│   ├── rottweiler/
│   └── yorkshire_terrier/
└── val/
    ├── beagle/
    ├── boxer/
    └── ...
```

**Note:** The application includes a pre-built model structure. For production use, you should train the model with your own dataset or use a publicly available dog breed dataset.

## Data Preprocessing

The application handles data preprocessing automatically through several stages:

### 1. Image Loading and Validation

```python
def preprocess_image(image_path):
    """
    Preprocess the uploaded image for prediction.
    """
    # Load image
    img = Image.open(image_path)
    
    # Convert to RGB if necessary
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Convert to numpy array
    img_array = np.array(img)
    
    return img_array
```

### 2. Image Resizing and Normalization

The model automatically:
- Resizes images to 224×224 pixels (MobileNetV2 input size)
- Normalizes pixel values using MobileNetV2's preprocess_input function
- Converts images to the appropriate tensor format

### 3. Data Augmentation (Training Only)

For model training, data augmentation is applied:
- Random rotation (±20 degrees)
- Width and height shifts (±20%)
- Horizontal flipping
- Zoom (±20%)
- Shear transformations

## Build and Train Model

### Model Architecture

The application uses **Transfer Learning** with MobileNetV2:

```python
class DogBreedClassifier:
    def build_model(self):
        # Load pre-trained MobileNetV2
        base_model = MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze base model
        base_model.trainable = False
        
        # Add custom layers
        inputs = keras.Input(shape=(224, 224, 3))
        x = preprocess_input(inputs)
        x = base_model(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = keras.Model(inputs, outputs)
        
        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
```

### Model Components

1. **Base Model**: MobileNetV2 (pre-trained on ImageNet)
   - Efficient architecture for mobile/embedded devices
   - Frozen layers to preserve learned features

2. **Custom Layers**:
   - Global Average Pooling: Reduces spatial dimensions
   - Dropout (0.2): Prevents overfitting
   - Dense (128 units): Feature learning layer
   - Output Layer: Softmax activation for multi-class classification

3. **Optimizer**: Adam with learning rate 0.001
4. **Loss Function**: Categorical cross-entropy
5. **Metrics**: Accuracy

### Training the Model

To train your own model:

```python
from dog_breed_model import DogBreedClassifier

# Initialize classifier
classifier = DogBreedClassifier(num_classes=10)

# Build model
classifier.build_model()

# Create data generators
train_gen, val_gen = classifier.create_sample_data_generators(
    'image_data/train',
    'image_data/val',
    batch_size=32
)

# Train model
history = classifier.train(
    train_gen, 
    val_gen, 
    epochs=20
)

# Save trained model
classifier.save_model('models/dog_breed_model.h5')
```

### Creating a Sample Model

For demonstration purposes, run:

```bash
python dog_breed_model.py
```

This creates a sample model structure with 10 breed classes.

## Testing the Model

### Loading and Testing

```python
from dog_breed_model import DogBreedClassifier
import numpy as np
from PIL import Image

# Initialize and load model
classifier = DogBreedClassifier(num_classes=10)
classifier.load_model('models/dog_breed_model.h5')

# Load test image
img = Image.open('test_dog.jpg')
img_array = np.array(img)

# Make prediction
breed, confidence, top_3 = classifier.predict(img_array)

print(f"Predicted Breed: {breed}")
print(f"Confidence: {confidence * 100:.2f}%")
print("\nTop 3 Predictions:")
for i, (breed_name, conf) in enumerate(top_3, 1):
    print(f"{i}. {breed_name}: {conf * 100:.2f}%")
```

### Model Performance

The model provides:
- **Primary Prediction**: Most likely breed with confidence score
- **Top-3 Predictions**: Three most probable breeds
- **Confidence Scores**: Percentage confidence for each prediction

### Example Output

```
Predicted Breed: Golden Retriever
Confidence: 87.45%

Top 3 Predictions:
1. Golden Retriever: 87.45%
2. Labrador: 8.32%
3. Golden Doodle: 2.15%
```

## Creating the Flask App

### Application Structure

```
Project2/
├── app.py                      # Flask application
├── dog_breed_model.py          # Model training and prediction
├── requirements.txt            # Python dependencies
├── README.md                   # Documentation
├── .gitignore                 # Git ignore rules
├── templates/                  # HTML templates
│   ├── base.html              # Base template
│   ├── index.html             # Upload page
│   ├── result.html            # Results page
│   └── about.html             # About page
├── static/
│   ├── uploads/               # Uploaded images
│   └── css/                   # CSS files (optional)
└── models/
    └── dog_breed_model.h5     # Trained model
```

### Flask Routes

1. **Home Route** (`/`):
   - Displays upload form
   - Shows supported breeds
   - Provides instructions

2. **Prediction Route** (`/predict`):
   - Handles file upload
   - Validates file type and size
   - Processes image
   - Returns prediction results

3. **About Route** (`/about`):
   - Information about the application
   - Technical details
   - How it works

### Key Functions

#### File Upload Handling

```python
def allowed_file(filename):
    """Check if file has allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}
```

#### Image Preprocessing

```python
def preprocess_image(image_path):
    """Preprocess image for prediction."""
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return np.array(img)
```

#### Prediction Handler

```python
@app.route('/predict', methods=['POST'])
def predict():
    """Handle file upload and make prediction."""
    # Validate upload
    # Save file
    # Preprocess image
    # Make prediction
    # Return results
```

## Running the App in System

### Development Mode

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

### Production Mode

1. **Set environment variables:**
```bash
export FLASK_ENV=production
export SECRET_KEY='your-secure-secret-key-here'
```

2. **Install production server:**
```bash
pip install gunicorn
```

3. **Run with gunicorn:**
```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Docker Deployment (Optional)

Create a `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

Build and run:

```bash
docker build -t dog-breed-app .
docker run -p 5000:5000 dog-breed-app
```

## Download The Projects Files

### Repository Structure

All project files are available in the GitHub repository:

```
DataScienceProjects/
└── Project2/
    ├── app.py                      # Main Flask application
    ├── dog_breed_model.py          # Model training script
    ├── requirements.txt            # Dependencies
    ├── README.md                   # This file
    ├── .gitignore                 # Git ignore rules
    ├── templates/                  # HTML templates
    ├── static/                     # Static files
    └── models/                     # Model files
```

### Cloning the Repository

```bash
# Clone the repository
git clone https://github.com/eodenyire/DataScienceProjects.git

# Navigate to Project2
cd DataScienceProjects/Project2

# Install dependencies
pip install -r requirements.txt

# Create sample model
python dog_breed_model.py

# Run the application
python app.py
```

### Individual Files

Download specific files:

- **Main Application**: `app.py`
- **Model Script**: `dog_breed_model.py`
- **Dependencies**: `requirements.txt`
- **Templates**: `templates/` directory
- **Documentation**: `README.md`

## Usage Guide

### Step-by-Step Usage

1. **Start the Application**
   ```bash
   python app.py
   ```

2. **Open Web Browser**
   - Navigate to `http://localhost:5000`

3. **Upload Dog Image**
   - Click "Choose Image" button
   - Select a clear image of a dog
   - Supported formats: PNG, JPG, JPEG
   - Maximum size: 16MB

4. **Get Prediction**
   - Click "Predict Breed" button
   - Wait for processing (few seconds)

5. **View Results**
   - See predicted breed
   - Check confidence score
   - Review top-3 alternatives

6. **Try Another Image**
   - Click "Try Another Image" to classify more dogs

### Best Practices for Accurate Results

- Use clear, well-lit images
- Ensure the dog is the main subject
- Avoid heavily cropped or zoomed images
- Use images showing the full dog or at least the face
- Higher resolution images typically give better results

## Supported Breeds

The current model supports 10 popular dog breeds:

1. **Beagle** - Small to medium-sized hound
2. **Boxer** - Medium to large, short-haired breed
3. **Bulldog** - Muscular, hefty breed with distinctive face
4. **Chihuahua** - Tiny breed with large personality
5. **Golden Retriever** - Large, friendly, intelligent breed
6. **German Shepherd** - Large, versatile working dog
7. **Labrador** - Friendly, outgoing, active companion
8. **Poodle** - Intelligent, active, elegant breed
9. **Rottweiler** - Large, robust, powerful guard dog
10. **Yorkshire Terrier** - Small, toy-sized terrier

## Technical Specifications

### System Requirements

- **Python**: 3.8 or higher
- **RAM**: 4GB minimum (8GB recommended)
- **Storage**: 2GB for dependencies and model
- **CPU**: Multi-core processor recommended
- **GPU**: Optional (for training only)

### Model Specifications

- **Architecture**: MobileNetV2 (Transfer Learning)
- **Input Size**: 224×224×3 (RGB images)
- **Parameters**: ~3.5M trainable parameters
- **Model Size**: ~15MB
- **Inference Time**: <1 second per image (CPU)

### Performance Metrics

- **Top-1 Accuracy**: Varies with training data quality
- **Top-3 Accuracy**: Higher than top-1
- **Inference Speed**: Fast (optimized for CPU)
- **Memory Usage**: Low (~200MB during prediction)

## Troubleshooting

### Common Issues

1. **Model Not Found**
   ```
   Solution: Run `python dog_breed_model.py` to create sample model
   ```

2. **TensorFlow Import Error**
   ```
   Solution: Install tensorflow: `pip install tensorflow>=2.13.0`
   ```

3. **Upload Folder Permission Error**
   ```
   Solution: Create folder manually: `mkdir -p static/uploads`
   ```

4. **Large File Upload Error**
   ```
   Solution: File size limit is 16MB, compress your image
   ```

5. **Port Already in Use**
   ```
   Solution: Change port in app.py or kill process using port 5000
   ```

## Applications

This dog breed prediction system can be used for:

- **Pet Adoption Centers**: Identify breed of rescued dogs
- **Veterinary Clinics**: Breed-specific care recommendations
- **Dog Shows**: Breed verification and registration
- **Educational**: Learning about different dog breeds
- **Mobile Apps**: Integration into dog-related applications
- **Social Media**: Fun breed identification for pet photos
- **Pet Insurance**: Breed identification for policy quotes

## Future Enhancements

Planned improvements for future versions:

- [ ] Expand to 120+ dog breeds (full dataset)
- [ ] Mixed breed detection and percentage estimation
- [ ] Batch processing for multiple images
- [ ] Real-time video feed prediction
- [ ] Breed information and characteristics display
- [ ] Similar breeds comparison
- [ ] REST API for third-party integration
- [ ] Mobile application (iOS/Android)
- [ ] User accounts and history tracking
- [ ] Social sharing features
- [ ] Advanced filtering and search

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

- **TensorFlow Team**: For the excellent deep learning framework
- **MobileNetV2**: For the efficient model architecture
- **Flask**: For the lightweight web framework
- **Dataset Contributors**: For dog breed image datasets

## Author

Developed as part of the Data Science Projects series.

## Contact & Support

For questions, issues, or suggestions:
- Open an issue on GitHub
- Check existing documentation
- Review troubleshooting section

---

**Note**: This is an educational project demonstrating the application of transfer learning and web development for image classification. For production use, consider training with larger datasets and implementing additional security measures.
