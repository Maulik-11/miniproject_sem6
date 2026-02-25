"""
MemeShield - AI-Powered Hate Speech Detection in Memes
Main Flask Application
"""

import os
import uuid
from datetime import datetime
from flask import Flask, render_template, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
import mysql.connector
from mysql.connector import Error

from config import Config
from utils.ocr import extract_text_from_image
from utils.preprocess import preprocess_text, preprocess_image
from utils.cnn_model import CNNFeatureExtractor
from utils.fusion_model import MemeClassifier

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize models (lazy loading for better performance)
cnn_extractor = None
meme_classifier = None


def get_db_connection():
    """Create and return a MySQL database connection."""
    try:
        connection = mysql.connector.connect(
            host=Config.MYSQL_HOST,
            user=Config.MYSQL_USER,
            password=Config.MYSQL_PASSWORD,
            database=Config.MYSQL_DATABASE,
            port=Config.MYSQL_PORT
        )
        return connection
    except Error as e:
        print(f"Error connecting to MySQL: {e}")
        return None


def init_database():
    """Initialize the database and create tables if they don't exist."""
    try:
        # First connect without database to create it if needed
        connection = mysql.connector.connect(
            host=Config.MYSQL_HOST,
            user=Config.MYSQL_USER,
            password=Config.MYSQL_PASSWORD,
            port=Config.MYSQL_PORT
        )
        cursor = connection.cursor()
        
        # Create database if not exists
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {Config.MYSQL_DATABASE}")
        cursor.execute(f"USE {Config.MYSQL_DATABASE}")
        
        # Create memes table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memes (
                id INT AUTO_INCREMENT PRIMARY KEY,
                filename VARCHAR(255) NOT NULL,
                original_filename VARCHAR(255),
                extracted_text TEXT,
                prediction VARCHAR(50) NOT NULL,
                confidence FLOAT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_prediction (prediction),
                INDEX idx_timestamp (timestamp)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """)
        
        connection.commit()
        cursor.close()
        connection.close()
        print("Database initialized successfully!")
        return True
    except Error as e:
        print(f"Error initializing database: {e}")
        return False


def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS


def get_models():
    """Lazy load and return the ML models."""
    global cnn_extractor, meme_classifier
    
    if cnn_extractor is None:
        cnn_extractor = CNNFeatureExtractor(model_name=Config.CNN_MODEL_NAME)
    
    if meme_classifier is None:
        meme_classifier = MemeClassifier()
    
    return cnn_extractor, meme_classifier


def save_to_database(filename, original_filename, extracted_text, prediction, confidence):
    """Save prediction result to MySQL database."""
    connection = get_db_connection()
    if connection is None:
        return False
    
    try:
        cursor = connection.cursor()
        query = """
            INSERT INTO memes (filename, original_filename, extracted_text, prediction, confidence)
            VALUES (%s, %s, %s, %s, %s)
        """
        cursor.execute(query, (filename, original_filename, extracted_text, prediction, confidence))
        connection.commit()
        cursor.close()
        connection.close()
        return True
    except Error as e:
        print(f"Error saving to database: {e}")
        return False


# ==================== ROUTES ====================

@app.route('/')
def index():
    """Home page with meme upload form."""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle meme upload and trigger prediction pipeline."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed: PNG, JPG, JPEG, GIF, WEBP'}), 400
    
    try:
        # Generate unique filename
        original_filename = secure_filename(file.filename)
        ext = original_filename.rsplit('.', 1)[1].lower()
        unique_filename = f"{uuid.uuid4().hex}.{ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        # Save file
        file.save(filepath)
        
        return jsonify({
            'success': True,
            'filename': unique_filename,
            'original_filename': original_filename,
            'filepath': filepath
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict', methods=['POST'])
def predict():
    """Run the AI pipeline on an uploaded meme."""
    data = request.get_json()
    
    if not data or 'filename' not in data:
        return jsonify({'error': 'No filename provided'}), 400
    
    filename = data['filename']
    original_filename = data.get('original_filename', filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
    
    try:
        # Step 1: Extract text using OCR
        extracted_text = extract_text_from_image(filepath)
        
        # Step 2: Preprocess text
        processed_text = preprocess_text(extracted_text)
        
        # Step 3: Preprocess image
        processed_image = preprocess_image(filepath)
        
        # Step 4: Get models and extract features
        cnn_ext, classifier = get_models()
        
        # Step 5: Extract image features using CNN
        image_features = cnn_ext.extract_features(processed_image)
        
        # Step 6: Run multimodal fusion and get prediction
        prediction, confidence = classifier.predict(processed_text, image_features)
        
        # Step 7: Save to database
        save_to_database(filename, original_filename, extracted_text, prediction, confidence)
        
        return jsonify({
            'success': True,
            'filename': filename,
            'extracted_text': extracted_text,
            'processed_text': processed_text,
            'prediction': prediction,
            'confidence': round(confidence * 100, 2),
            'label': 'Hate Speech' if prediction == 'hate' else 'Non-Hate Speech'
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/history')
def history():
    """Display prediction history from database."""
    connection = get_db_connection()
    
    if connection is None:
        return render_template('history.html', memes=[], error="Database connection failed")
    
    try:
        cursor = connection.cursor(dictionary=True)
        cursor.execute("""
            SELECT id, filename, original_filename, extracted_text, 
                   prediction, confidence, timestamp
            FROM memes
            ORDER BY timestamp DESC
            LIMIT 100
        """)
        memes = cursor.fetchall()
        cursor.close()
        connection.close()
        
        return render_template('history.html', memes=memes)
    
    except Error as e:
        return render_template('history.html', memes=[], error=str(e))


@app.route('/api/history')
def api_history():
    """API endpoint to get prediction history as JSON."""
    connection = get_db_connection()
    
    if connection is None:
        return jsonify({'error': 'Database connection failed'}), 500
    
    try:
        cursor = connection.cursor(dictionary=True)
        cursor.execute("""
            SELECT id, filename, original_filename, extracted_text, 
                   prediction, confidence, timestamp
            FROM memes
            ORDER BY timestamp DESC
            LIMIT 100
        """)
        memes = cursor.fetchall()
        cursor.close()
        connection.close()
        
        # Convert datetime to string for JSON serialization
        for meme in memes:
            if meme['timestamp']:
                meme['timestamp'] = meme['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        
        return jsonify({'success': True, 'data': memes})
    
    except Error as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats')
def api_stats():
    """API endpoint to get prediction statistics."""
    connection = get_db_connection()
    
    if connection is None:
        return jsonify({'error': 'Database connection failed'}), 500
    
    try:
        cursor = connection.cursor(dictionary=True)
        
        # Get total counts
        cursor.execute("SELECT COUNT(*) as total FROM memes")
        total = cursor.fetchone()['total']
        
        # Get hate speech count
        cursor.execute("SELECT COUNT(*) as hate_count FROM memes WHERE prediction = 'hate'")
        hate_count = cursor.fetchone()['hate_count']
        
        # Get average confidence
        cursor.execute("SELECT AVG(confidence) as avg_confidence FROM memes")
        avg_confidence = cursor.fetchone()['avg_confidence'] or 0
        
        cursor.close()
        connection.close()
        
        return jsonify({
            'success': True,
            'total_analyzed': total,
            'hate_speech_count': hate_count,
            'non_hate_count': total - hate_count,
            'average_confidence': round(avg_confidence * 100, 2)
        })
    
    except Error as e:
        return jsonify({'error': str(e)}), 500


@app.route('/delete/<int:meme_id>', methods=['DELETE'])
def delete_meme(meme_id):
    """Delete a meme record from the database."""
    connection = get_db_connection()
    
    if connection is None:
        return jsonify({'error': 'Database connection failed'}), 500
    
    try:
        cursor = connection.cursor(dictionary=True)
        
        # Get filename to delete the file
        cursor.execute("SELECT filename FROM memes WHERE id = %s", (meme_id,))
        result = cursor.fetchone()
        
        if result:
            # Delete file from uploads folder
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], result['filename'])
            if os.path.exists(filepath):
                os.remove(filepath)
            
            # Delete from database
            cursor.execute("DELETE FROM memes WHERE id = %s", (meme_id,))
            connection.commit()
        
        cursor.close()
        connection.close()
        
        return jsonify({'success': True})
    
    except Error as e:
        return jsonify({'error': str(e)}), 500


# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    # Initialize database on startup
    init_database()
    
    # Run the Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=Config.DEBUG
    )
