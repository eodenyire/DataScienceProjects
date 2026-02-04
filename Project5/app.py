"""
Flask Web Application for Text Extraction from Images

This application provides a web interface for extracting text from images
using Tesseract OCR with preprocessing options.

Author: Emmanuel Odenyire
Email: eodenyire@gmail.com
"""

import os
from flask import Flask, render_template, request, flash, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from text_extractor import TextExtractor

app = Flask(__name__)
# Use environment variable for SECRET_KEY in production
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'tif', 'webp'}

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize text extractor
text_extractor = TextExtractor()


def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    """Render the home page with upload form."""
    return render_template('index.html')


@app.route('/extract', methods=['POST'])
def extract():
    """
    Handle file upload and perform text extraction.
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
        flash('Invalid file type. Please upload an image file.', 'error')
        return redirect(url_for('index'))
    
    # Get extraction options
    preprocess = request.form.get('preprocess') == 'on'
    show_confidence = request.form.get('show_confidence') == 'on'
    language = request.form.get('language', 'eng')
    
    # Save the uploaded file
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    try:
        # Save preprocessed image if requested
        preprocessed_path = None
        if preprocess:
            preprocessed_filename = f"preprocessed_{filename}"
            preprocessed_path = os.path.join(app.config['UPLOAD_FOLDER'], preprocessed_filename)
            text_extractor.save_preprocessed_image(filepath, preprocessed_path)
        
        # Extract text
        if show_confidence:
            result = text_extractor.extract_text_with_confidence(
                filepath, 
                preprocess=preprocess,
                language=language
            )
            extracted_text = result['text']
            word_count = len(result['words'])
            avg_confidence = round(result['average_confidence'], 2)
            words_data = result['words']
        else:
            extracted_text = text_extractor.extract_text(
                filepath, 
                preprocess=preprocess,
                language=language
            )
            word_count = len(extracted_text.split())
            avg_confidence = None
            words_data = None
        
        # Count characters and lines
        char_count = len(extracted_text)
        line_count = len(extracted_text.split('\n'))
        
        # Prepare result data
        result_data = {
            'extracted_text': extracted_text,
            'original_image': filepath,
            'preprocessed_image': preprocessed_path,
            'filename': filename,
            'word_count': word_count,
            'char_count': char_count,
            'line_count': line_count,
            'average_confidence': avg_confidence,
            'words_data': words_data,
            'show_confidence': show_confidence,
            'language': language,
            'preprocessed': preprocess
        }
        
        return render_template('result.html', result=result_data)
    
    except Exception as e:
        flash(f'Error extracting text: {str(e)}', 'error')
        return redirect(url_for('index'))


@app.route('/download/<filename>')
def download(filename):
    """
    Serve uploaded or processed files for download.
    """
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)


@app.route('/about')
def about():
    """Render the about page."""
    return render_template('about.html')


if __name__ == '__main__':
    # Only enable debug mode in development
    # In production, set FLASK_ENV=production and use a proper WSGI server
    debug_mode = os.environ.get('FLASK_ENV', 'production') == 'development'
    app.run(debug=debug_mode, host='0.0.0.0', port=5000)
