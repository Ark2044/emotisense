import os
import logging
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from dotenv import load_dotenv
import google.generativeai as genai

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='app.log'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    logger.error("GOOGLE_API_KEY environment variable is not set")
    raise ValueError("GOOGLE_API_KEY environment variable is not set")

# Configure Generative AI
genai.configure(api_key=GOOGLE_API_KEY)

# Load the emotion recognition model with error handling
try:
    emotion_model = load_model('model_1.h5')
    emotion_model.load_weights('model_weights1.h5')
    logger.info("Emotion recognition model loaded successfully")
except Exception as e:
    logger.error(f"Error loading emotion model: {e}")
    raise

# Class labels for the emotion recognition model
CLASS_LABELS = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

# Meme mapping for emotions
EMOTION_MEME_MAPPING = {
    'happy': 'happy_meme.jpg',
    'sad': 'sad_meme.jpg',
    'angry': 'angry_meme.jpg',
    'fearful': 'fear_meme.jpg',
    'disgusted': 'disgust_meme.jpg',
    'surprised': 'surprise_meme.jpg',
    'neutral': 'neutral_meme.jpg'
}

# Initialize Flask app
app = Flask(__name__)
app.config.update(
    UPLOAD_FOLDER='uploads',
    MEME_FOLDER='memes',
    MAX_CONTENT_LENGTH=16 * 1024 * 1024,  # 16MB max file size
)

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MEME_FOLDER'], exist_ok=True)

def preprocess_image(filepath):
    """
    Preprocess the uploaded image for emotion detection.
    
    Args:
        filepath (str): Path to the uploaded image file
    
    Returns:
        np.array: Preprocessed image array
    """
    try:
        # Read image in grayscale
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Could not read the image")

        # Resize and normalize
        img_resized = cv2.resize(img, (48, 48))
        processed_frame = np.expand_dims(img_resized, axis=-1)
        processed_frame = np.expand_dims(processed_frame, axis=0)
        processed_frame = processed_frame / 255.0

        return processed_frame
    except Exception as e:
        logger.error(f"Image preprocessing error: {e}")
        raise

@app.route('/')
def index():
    """Render the main emotion detection page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict emotion from uploaded image.
    
    Returns:
        Redirect to results page with detected emotion
    """
    # Validate file upload
    if 'image' not in request.files:
        logger.warning("No image file uploaded")
        return "No image file uploaded", 400

    file = request.files['image']
    if file.filename == '':
        logger.warning("No image selected")
        return "No image selected", 400

    # Save the file with a unique filename
    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        # Preprocess and predict
        processed_frame = preprocess_image(filepath)
        predictions = emotion_model.predict(processed_frame)
        predicted_class = np.argmax(predictions, axis=-1)[0]
        predicted_label = CLASS_LABELS[predicted_class]

        # Log the prediction
        logger.info(f"Emotion detected: {predicted_label}")

        # Redirect to result page with the predicted emotion
        return redirect(url_for('result', emotion=predicted_label))

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return "Error processing image", 500

@app.route('/result')
def result():
    """
    Render the results page with detected emotion and meme.
    
    Returns:
        Rendered result template
    """
    emotion = request.args.get('emotion', 'neutral')
    
    # Get corresponding meme or default
    meme_filename = EMOTION_MEME_MAPPING.get(emotion, 'neutral_meme.jpg')
    
    return render_template('result.html', emotion=emotion, meme=meme_filename)

@app.route('/memes/<filename>')
def serve_meme(filename):
    """
    Serve meme images.
    
    Args:
        filename (str): Name of the meme file
    
    Returns:
        Meme image file
    """
    return send_from_directory(app.config['MEME_FOLDER'], filename)

@app.route('/text_query', methods=['POST'])
def text_query():
    """
    Handle text-based AI queries.
    
    Returns:
        JSON response with AI-generated text
    """
    user_input = request.form.get('query', '').strip()
    
    if not user_input:
        return jsonify({'response': "Please provide a query."})

    try:
        # Use Gemini-1.5 Flash for quick responses
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Add context about being an emotion AI companion
        context = (
            "You are an empathetic AI companion helping a user understand their emotions. "
            "Provide supportive, insightful responses that help the user reflect on their feelings."
        )
        full_prompt = f"{context}\n\nUser's message: {user_input}"
        
        response = model.generate_content(full_prompt)
        
        # Log the query and response
        logger.info(f"Text Query - Input: {user_input}")
        
        return jsonify({'response': response.text})
    
    except Exception as e:
        logger.error(f"Text query error: {e}")
        return jsonify({'response': "I'm having trouble processing your request. Please try again."})

@app.route('/voice_query', methods=['POST'])
def voice_query():
    """
    Placeholder for voice query functionality.
    
    Returns:
        JSON response with a mock AI response
    """
    try:
        # In a real implementation, you'd process voice input here
        mock_response = "I heard your voice query. How are you feeling today?"
        return jsonify({'response': mock_response})
    except Exception as e:
        logger.error(f"Voice query error: {e}")
        return jsonify({'response': "Error processing voice query"})

# Error handlers
@app.errorhandler(404)
def page_not_found(e):
    """Custom 404 error handler."""
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    """Custom 500 error handler."""
    return render_template('500.html'), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)