# Flask app
from flask import Flask, request, jsonify
from predict import CyberbullyingPredictor
import logging
import os

app = Flask(__name__)

# Initialize predictor
MODEL_PATH = os.getenv('MODEL_PATH', 'models/model_d=5_msl=50.bin')
predictor = CyberbullyingPredictor(MODEL_PATH)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'}), 200


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict cyberbullying risk

    Expected JSON:
    {
        "age_group": "13-17",
        "gender": "Female",
        "daily_internet_hours": 6,
        "primary_activity": "Gaming",
        "uses_facebook": 1,
        "num_social_media_accounts": 3,
        "exposed_to_bad_language": 1,
        "learned_bad_words": 1,
        "received_school_education": 0,
        "awareness_level": "Low"
    }
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        # Validate required fields
        required_fields = [
            'age_group', 'gender', 'daily_internet_hours', 'primary_activity',
            'uses_facebook', 'num_social_media_accounts', 'exposed_to_bad_language',
            'learned_bad_words', 'received_school_education', 'awareness_level'
        ]

        missing_fields = [f for f in required_fields if f not in data]
        if missing_fields:
            return jsonify({'error': f'Missing fields: {missing_fields}'}), 400

        # Make prediction
        result = predictor.predict(data)

        return jsonify({
            'prediction': result,
            'status': 'success'
        }), 200

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e), 'status': 'error'}), 500


@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """Batch predictions"""
    try:
        data = request.get_json()

        if not isinstance(data, list):
            return jsonify({'error': 'Expected list of students'}), 400

        results = predictor.predict_batch(data)

        return jsonify({
            'predictions': results,
            'count': len(results),
            'status': 'success'
        }), 200

    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({'error': str(e), 'status': 'error'}), 500


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)