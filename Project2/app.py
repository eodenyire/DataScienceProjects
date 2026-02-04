"""
Flask Web Application for Dog Breed Prediction
"""

import os
import numpy as np
from flask import Flask, render_template, request, flash, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image
import tensorflow as tf
from dog_breed_model import DogBreedClassifier

app = Flask(__name__)
# Use environment variable for SECRET_KEY in production
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the model
classifier = DogBreedClassifier(num_classes=10)
MODEL_PATH = 'models/dog_breed_model.h5'

# Try to load the model, create sample if not exists
if os.path.exists(MODEL_PATH):
    try:
        classifier.load_model(MODEL_PATH)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"⚠ Error loading model: {e}")
        print("Please run: python dog_breed_model.py to create the model")
        classifier = None
else:
    print("⚠ Model not found at:", MODEL_PATH)
    print("Please run: python dog_breed_model.py to create the model")
    classifier = None


def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def preprocess_image(image_path):
    """
    Preprocess the uploaded image for prediction.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        numpy.ndarray: Preprocessed image array
    """
    # Load image
    img = Image.open(image_path)
    
    # Convert to RGB if necessary
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Convert to numpy array
    img_array = np.array(img)
    
    return img_array


@app.route('/')
def index():
    """Render the home page."""
    breeds = classifier.class_names if classifier else []
    return render_template('index.html', breeds=breeds)


@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle file upload and perform breed prediction.
    """
    # Check if model is loaded
    if classifier is None:
        flash('Model not loaded. Please run: python dog_breed_model.py', 'error')
        return redirect(url_for('index'))
    
    # Check if file was uploaded
    if 'file' not in request.files:
        flash('No file uploaded', 'error')
        return redirect(url_for('index'))
    
    file = request.files['file']
    
    # Check if filename is empty
    if file.filename == '':
        flash('No file selected', 'error')
        return redirect(url_for('index'))
    
    # Check if file is allowed
    if not allowed_file(file.filename):
        flash('Invalid file type. Only PNG, JPG, and JPEG are allowed.', 'error')
        return redirect(url_for('index'))
    
    try:
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Preprocess image
        img_array = preprocess_image(filepath)
        
        # Make prediction
        predicted_breed, confidence, top_predictions = classifier.predict(img_array)
        
        # Prepare results
        result = {
            'filename': filename,
            'image_path': filepath,
            'predicted_breed': predicted_breed.replace('_', ' ').title(),
            'confidence': round(confidence * 100, 2),
            'top_predictions': [
                {
                    'breed': breed.replace('_', ' ').title(),
                    'confidence': round(conf * 100, 2)
                }
                for breed, conf in top_predictions
            ]
        }
        
        return render_template('result.html', result=result)
    
    except Exception as e:
        flash(f'Error processing image: {str(e)}', 'error')
        return redirect(url_for('index'))


@app.route('/about')
def about():
    """Render the about page."""
    return render_template('about.html')


if __name__ == '__main__':
    # Only enable debug mode in development
    # In production, set FLASK_ENV=production and use a proper WSGI server
    debug_mode = os.environ.get('FLASK_ENV', 'production') == 'development'
    app.run(debug=debug_mode, host='0.0.0.0', port=5000)
