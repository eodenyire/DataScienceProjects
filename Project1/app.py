"""
Flask Web Application for Pan Card Tampering Detection
"""

import os
from flask import Flask, render_template, request, flash, redirect, url_for
from werkzeug.utils import secure_filename
import cv2
from pan_card_detector import detect_tampering, compare_images

app = Flask(__name__)
# Use environment variable for SECRET_KEY in production
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def save_processed_images(diff, thresh, contours, upload_folder):
    """
    Save the processed images (difference, threshold, and contours).
    
    Args:
        diff: Difference image
        thresh: Threshold image
        contours: Image with contours drawn
        upload_folder: Folder to save images
    
    Returns:
        tuple: Paths to saved images (diff_path, thresh_path, contours_path)
    """
    diff_path = os.path.join(upload_folder, 'difference.jpg')
    thresh_path = os.path.join(upload_folder, 'threshold.jpg')
    contours_path = os.path.join(upload_folder, 'contours.jpg')
    
    cv2.imwrite(diff_path, diff)
    cv2.imwrite(thresh_path, thresh)
    cv2.imwrite(contours_path, contours)
    
    return diff_path, thresh_path, contours_path


@app.route('/')
def index():
    """Render the home page."""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Handle file upload and perform tampering detection.
    """
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
    
    # Save the uploaded file
    filename = secure_filename(file.filename)
    user_image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(user_image_path)
    
    # Path to original Pan Card image (you should have this in image_data folder)
    original_image_path = 'image_data/original_pan_card.jpg'
    
    # Check if original image exists
    if not os.path.exists(original_image_path):
        flash('Original Pan Card image not found. Please add it to image_data folder.', 'error')
        return redirect(url_for('index'))
    
    try:
        # Perform tampering detection
        similarity_score, diff, thresh, contours_image = detect_tampering(
            original_image_path, 
            user_image_path
        )
        
        # Save processed images
        diff_path, thresh_path, contours_path = save_processed_images(
            diff, thresh, contours_image, app.config['UPLOAD_FOLDER']
        )
        
        # Determine if tampered
        is_tampered = similarity_score < 0.95
        
        # Prepare results
        result = {
            'similarity': round(similarity_score * 100, 2),
            'is_tampered': is_tampered,
            'original_image': original_image_path,
            'uploaded_image': user_image_path,
            'difference_image': diff_path,
            'threshold_image': thresh_path,
            'contours_image': contours_path
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
    app.run(debug=True, host='0.0.0.0', port=5000)
