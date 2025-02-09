from flask import Flask, render_template, request, jsonify, send_from_directory
from sentiment_evaluator import SentimentEvaluator
import os
import logging


app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model")
evaluator = SentimentEvaluator(MODEL_PATH)

#logging configuration
logging.basicConfig(level=logging.DEBUG)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data.get('text', '')
        if not text:
            return jsonify({'error': 'No text provided'}), 400
            
        logging.debug(f"Received text: {text}")
        sentiment, scores = evaluator.predict_sentiment(text, return_scores=True)
        logging.debug(f"Prediction result - Sentiment: {sentiment}, Scores: {scores}")
        
        response = {'sentiment': sentiment, 'scores': scores}
        return jsonify(response)
    except Exception as e:
        logging.error(f"Error in predict route: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Get port from environment variable for Render
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port) 