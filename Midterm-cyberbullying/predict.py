# Predictive model
import pickle
import os
from typing import Dict, List


class CyberbullyingPredictor:
    def __init__(self, model_path='models/model_d=5_msl=50.bin'):
        """Load the trained model and vectorizer"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")

        with open(model_path, 'rb') as f:
            self.dv, self.rf = pickle.load(f)

    def predict(self, data: Dict) -> Dict:
        """
        Make prediction for a single student

        Args:
            data: Dictionary with keys:
                - age_group: '9-12' or '13-17'
                - gender: 'Male' or 'Female'
                - daily_internet_hours: int (0-8+)
                - primary_activity: 'Chatting', 'Studying', 'Gaming', 'Social Media', 'Videos'
                - uses_facebook: bool or 1/0
                - num_social_media_accounts: int
                - exposed_to_bad_language: bool or 1/0
                - learned_bad_words: bool or 1/0
                - received_school_education: bool or 1/0
                - awareness_level: 'Low', 'Medium', 'High'

        Returns:
            Dictionary with prediction and confidence
        """
        # Convert to format expected by DictVectorizer
        X = self.dv.transform([data])

        # Get prediction and probability
        prediction = self.rf.predict(X)[0]
        probability = self.rf.predict_proba(X)[0]

        return {
            'cyberbullying_risk': bool(prediction),
            'risk_probability': float(probability[1]),  # Probability of True
            'confidence': float(max(probability))
        }

    def predict_batch(self, data_list: List[Dict]) -> List[Dict]:
        """Make predictions for multiple students"""
        return [self.predict(data) for data in data_list]


# Example usage
if __name__ == '__main__':
    predictor = CyberbullyingPredictor()

    # Test prediction
    test_student = {
        'age_group': '13-17',
        'gender': 'Female',
        'daily_internet_hours': 6,
        'primary_activity': 'Gaming',
        'uses_facebook': 1,
        'num_social_media_accounts': 3,
        'exposed_to_bad_language': 1,
        'learned_bad_words': 1,
        'received_school_education': 0,
        'awareness_level': 'Low'
    }

    result = predictor.predict(test_student)
    print(result)