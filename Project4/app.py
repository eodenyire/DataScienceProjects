"""
Flask Web Application for Traffic Sign Classification
"""

import os
from flask import Flask, render_template, request, flash, redirect, url_for
from werkzeug.utils import secure_filename
from traffic_sign_classifier import TrafficSignClassifier
import cv2
from PIL import Image

app = Flask(__name__)
# Use environment variable for SECRET_KEY in production
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize the classifier
classifier = TrafficSignClassifier()

# Try to load pre-trained model or build a new one
MODEL_PATH = 'models/traffic_sign_model.h5'
if os.path.exists(MODEL_PATH):
    try:
        classifier.load_model(MODEL_PATH)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"⚠ Error loading model: {e}")
        print("Building new model...")
        classifier.build_model(use_pretrained=False)
else:
    print("Building new model (no pre-trained model found)...")
    classifier.build_model(use_pretrained=False)


def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    """Render the home page."""
    return render_template('index.html')


@app.route('/about')
def about():
    """Render the about page."""
    return render_template('about.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Handle file upload and perform traffic sign classification.
    """
    # Check if file was uploaded
    if 'file' not in request.files:
        flash('No file uploaded', 'error')
        return redirect(url_for('index'))
    
    file = request.files['file']
    
    # Check if file was selected
    if file.filename == '':
        flash('No file selected', 'error')
        return redirect(url_for('index'))
    
    # Validate file type
    if not allowed_file(file.filename):
        flash('Invalid file type. Please upload a PNG, JPG, or JPEG image.', 'error')
        return redirect(url_for('index'))
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Perform classification
        result = classifier.predict(filepath)
        
        # Prepare data for result page
        result_data = {
            'uploaded_image': filename,
            'predicted_class': result['predicted_class'],
            'confidence': result['confidence'],
            'top_3': result['top_3']
        }
        
        return render_template('result.html', **result_data)
    
    except Exception as e:
        flash(f'Error processing image: {str(e)}', 'error')
        return redirect(url_for('index'))


if __name__ == '__main__':
    # Get port from environment variable or use default
    port = int(os.environ.get('PORT', 5000))
    
    # Run the application
    app.run(
        host='0.0.0.0',
        port=port,
        debug=os.environ.get('FLASK_ENV') == 'development'
    )
