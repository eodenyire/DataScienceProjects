"""
Flask Web Application for Image Watermarking

This application allows users to add text or image watermarks to their images.
"""

import os
from flask import Flask, render_template, request, flash, redirect, url_for, send_file
from werkzeug.utils import secure_filename
from image_watermarker import ImageWatermarker
import cv2

app = Flask(__name__)
# Use environment variable for SECRET_KEY in production
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['LOGO_FOLDER'] = 'static/logos'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Ensure folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['LOGO_FOLDER'], exist_ok=True)

# Initialize watermarker
watermarker = ImageWatermarker()


def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def get_available_logos():
    """Get list of available logo files in the logo folder."""
    logos = []
    if os.path.exists(app.config['LOGO_FOLDER']):
        for filename in os.listdir(app.config['LOGO_FOLDER']):
            if allowed_file(filename):
                logos.append(filename)
    return logos


@app.route('/')
def index():
    """Render the home page."""
    logos = get_available_logos()
    return render_template('index.html', logos=logos)


@app.route('/watermark/text', methods=['POST'])
def watermark_text():
    """
    Handle text watermarking request.
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
    
    # Get watermark parameters
    text = request.form.get('watermark_text', 'Watermark')
    position = request.form.get('position', 'bottom-right')
    font_size = int(request.form.get('font_size', 36))
    opacity = float(request.form.get('opacity', 0.5))
    
    # Get color
    color_hex = request.form.get('color', '#FFFFFF')
    # Convert hex to RGB
    color_hex = color_hex.lstrip('#')
    color = tuple(int(color_hex[i:i+2], 16) for i in (0, 2, 4))
    
    try:
        # Save the uploaded file
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], f'input_{filename}')
        file.save(input_path)
        
        # Create watermarked image
        output_filename = f'watermarked_{filename}'
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
        
        watermarker.add_text_watermark(
            input_path, 
            text, 
            output_path,
            position=position,
            font_size=font_size,
            color=color,
            opacity=opacity
        )
        
        # Prepare result
        result = {
            'type': 'text',
            'original_image': input_path,
            'watermarked_image': output_path,
            'watermark_text': text,
            'position': position,
            'font_size': font_size,
            'opacity': opacity
        }
        
        return render_template('result.html', result=result)
    
    except Exception as e:
        flash(f'Error processing image: {str(e)}', 'error')
        return redirect(url_for('index'))


@app.route('/watermark/image', methods=['POST'])
def watermark_image():
    """
    Handle image watermarking request.
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
    
    # Get watermark parameters
    position = request.form.get('position', 'bottom-right')
    size_ratio = float(request.form.get('size_ratio', 0.15))
    opacity = float(request.form.get('opacity', 0.5))
    
    # Handle logo upload or selection
    logo_path = None
    
    # Check if user uploaded a logo
    if 'logo_file' in request.files and request.files['logo_file'].filename != '':
        logo_file = request.files['logo_file']
        if allowed_file(logo_file.filename):
            logo_filename = secure_filename(logo_file.filename)
            logo_path = os.path.join(app.config['LOGO_FOLDER'], logo_filename)
            logo_file.save(logo_path)
    
    # If no logo uploaded, check if existing logo selected
    if logo_path is None:
        selected_logo = request.form.get('selected_logo', '')
        if selected_logo:
            logo_path = os.path.join(app.config['LOGO_FOLDER'], selected_logo)
            if not os.path.exists(logo_path):
                flash('Selected logo not found', 'error')
                return redirect(url_for('index'))
        else:
            flash('Please upload or select a logo', 'error')
            return redirect(url_for('index'))
    
    try:
        # Save the uploaded file
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], f'input_{filename}')
        file.save(input_path)
        
        # Create watermarked image
        output_filename = f'watermarked_{filename}'
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
        
        watermarker.add_image_watermark(
            input_path,
            logo_path,
            output_path,
            position=position,
            size_ratio=size_ratio,
            opacity=opacity
        )
        
        # Prepare result
        result = {
            'type': 'image',
            'original_image': input_path,
            'watermarked_image': output_path,
            'logo_image': logo_path,
            'position': position,
            'size_ratio': size_ratio,
            'opacity': opacity
        }
        
        return render_template('result.html', result=result)
    
    except Exception as e:
        flash(f'Error processing image: {str(e)}', 'error')
        return redirect(url_for('index'))


@app.route('/download/<path:filename>')
def download_file(filename):
    """
    Allow users to download the watermarked image.
    """
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True)
        else:
            flash('File not found', 'error')
            return redirect(url_for('index'))
    except Exception as e:
        flash(f'Error downloading file: {str(e)}', 'error')
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
