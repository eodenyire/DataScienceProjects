"""
Flask Web Application for Vehicle Detection and Counting

This application provides a web interface for detecting and counting vehicles
in uploaded images using computer vision techniques.

Author: Emmanuel Odenyire
Email: eodenyire@gmail.com
"""

import os
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from vehicle_detector import VehicleDetector
import secrets

# Initialize Flask app
app = Flask(__name__)

# Configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', secrets.token_hex(16))
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp'}

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize vehicle detector
vehicle_detector = VehicleDetector(min_contour_area=500)


def allowed_file(filename):
    """
    Check if the uploaded file has an allowed extension.
    
    Args:
        filename: Name of the file
        
    Returns:
        Boolean indicating if file is allowed
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def save_processed_image(image, filename):
    """
    Save processed image to upload folder.
    
    Args:
        image: OpenCV image array
        filename: Name for the saved file
        
    Returns:
        Path to saved file
    """
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    cv2.imwrite(filepath, image)
    return filepath


@app.route('/')
def index():
    """
    Home page route - displays upload form.
    """
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Handle file upload and vehicle detection.
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
    
    # Check if file is allowed
    if not allowed_file(file.filename):
        flash('Invalid file type. Please upload PNG, JPG, JPEG, or BMP images.', 'error')
        return redirect(url_for('index'))
    
    try:
        # Secure the filename
        filename = secure_filename(file.filename)
        
        # Save uploaded file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process image for vehicle detection
        processed_image, vehicle_count, details = vehicle_detector.detect_vehicles_in_image(filepath)
        
        # Save processed image
        processed_filename = f'processed_{filename}'
        save_processed_image(processed_image, processed_filename)
        
        # Prepare result data
        result_data = {
            'original_image': filename,
            'processed_image': processed_filename,
            'vehicle_count': vehicle_count,
            'details': details
        }
        
        return render_template('result.html', result=result_data)
    
    except Exception as e:
        flash(f'Error processing image: {str(e)}', 'error')
        return redirect(url_for('index'))


@app.route('/about')
def about():
    """
    About page route - displays information about the application.
    """
    return render_template('about.html')


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """
    Serve uploaded files.
    
    Args:
        filename: Name of the file to serve
        
    Returns:
        File from upload directory
    """
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.errorhandler(413)
def too_large(e):
    """
    Handle file too large error.
    """
    flash('File is too large. Maximum size is 16MB.', 'error')
    return redirect(url_for('index'))


@app.errorhandler(404)
def page_not_found(e):
    """
    Handle 404 errors.
    """
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_error(e):
    """
    Handle 500 errors.
    """
    return render_template('500.html'), 500


if __name__ == '__main__':
    # Get configuration from environment
    debug = os.environ.get('FLASK_ENV') == 'development'
    port = int(os.environ.get('PORT', 5000))
    
    print("=" * 60)
    print("Vehicle Detection and Counting Flask Application")
    print("=" * 60)
    print(f"Environment: {'Development' if debug else 'Production'}")
    print(f"Server: http://localhost:{port}")
    print("=" * 60)
    
    # Run the app
    app.run(host='0.0.0.0', port=port, debug=debug)
