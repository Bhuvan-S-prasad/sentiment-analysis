import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
import string
import logging
import os

class SentimentEvaluator:
    def __init__(self, model_path):
        """
        Initialize the sentiment evaluator with a trained model.
        
        Args:
            model_path (str): Path to the saved model directory
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.label2id = {'Positive': 1, 'Negative': 0, 'Neutral': 2, 'Irrelevant': 3}
        self.id2label = {v: k for k, v in self.label2id.items()}
        
    def preprocess_text(self, text):
        """
        Preprocess the input text using the same preprocessing steps as training.
        
        Args:
            text (str): Input text to preprocess
            
        Returns:
            str: Preprocessed text
        """
        # Remove URLs
        text = re.sub(r'http\S+|www\.\S+', '', text)
        # Remove Twitter handles (mentions)
        text = re.sub(r'@\w+', '', text)
        # Remove hash '#' symbol from hashtags
        text = re.sub(r'#', '', text)
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        # Remove HTML entities
        text = re.sub(r'&\w+;', '', text)
        # Convert text to lowercase
        text = text.lower()
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def predict_sentiment(self, text, return_scores=False):
        """
        Predict sentiment for the given text.
        
        Args:
            text (str): Input text to analyze
            return_scores (bool): If True, return prediction probabilities for all classes
            
        Returns:
            str: Predicted sentiment label
            dict (optional): Prediction probabilities for all classes if return_scores=True
        """
        try:
            # Preprocess the text
            cleaned_text = self.preprocess_text(text)
            logging.debug(f"Cleaned text: {cleaned_text}")
            
            # Tokenize
            inputs = self.tokenizer(
                cleaned_text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=128
            ).to(self.device)
            
            # Get prediction
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1)
                prediction = torch.argmax(logits, dim=1)
            
            predicted_label = self.id2label[prediction.item()]
            logging.debug(f"Predicted label: {predicted_label}")
            
            if return_scores:
                scores = {
                    label: probabilities[0][idx].item()
                    for label, idx in self.label2id.items()
                }
                logging.debug(f"Confidence scores: {scores}")
                return predicted_label, scores
            
            return predicted_label
        except Exception as e:
            logging.error(f"Error in predict_sentiment: {str(e)}")
            raise

if __name__ == "__main__":
    # Use relative path for testing
    model_path = os.path.join(os.path.dirname(__file__), "model")
    evaluator = SentimentEvaluator(model_path)
    test_text = "This game is absolutely amazing!"
    sentiment, scores = evaluator.predict_sentiment(test_text, return_scores=True)
    print(f"Text: {test_text}")
    print(f"Predicted Sentiment: {sentiment}")
    print("Confidence Scores:")
    for label, score in scores.items():
        print(f"  {label}: {score:.4f}") 