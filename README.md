# Sentiment Analyzer: Advanced Sentiment Analysis System

A high-performance real-time sentiment analysis platform leveraging deep learning and NLP techniques to classify text into multiple sentiment categories.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Training Process](#training-process)
- [Results](#results)
- [Skills and Technologies Used](#skills-and-technologies-used)
- [Conclusion](#conclusion)
- [Getting Started](#getting-started)

## Overview
The Sentiment Analyzer is designed to provide instant sentiment analysis of texts such as tweets, game reviews, and social media posts. It identifies four sentiment classes: **Positive**, **Negative**, **Neutral**, and **Irrelevant**. The system leverages advanced transformer architectures, particularly the DistilBERT model, fine-tuned for multi-class sentiment classification. Additionally, the project features:
- A web-based user interface powered by Flask.
- Real-time results with confidence scores.
- A robust, reproducible training pipeline using PyTorch and Hugging Face's Transformers.

## Dataset
The dataset used in this project is a custom curated collection of social media texts and game reviews. It is divided into:
- **Training Dataset:** A larger dataset containing tweets and game review sentences with sentiment annotations.
- **Validation Dataset:** A smaller set reserved for evaluating the model's performance.
  
The data comes with the following characteristics:
- **Columns:** id, game, sentiment, and text.
- **Labels:** Mapped as follows:
  - Positive: 1
  - Negative: 0
  - Neutral: 2
  - Irrelevant: 3
- Missing texts have been handled by filling with "no text provided" to ensure consistency during preprocessing.

## Data Preprocessing
Data preprocessing is a critical step in ensuring quality data for model training. The following preprocessing steps were applied:
- **URL Removal:** Eliminates any URLs or web links.
- **Twitter Handle Removal:** Strips out any mentions (e.g., `@username`).
- **Hashtag Cleaning:** Removes the '#' characters from hashtags.
- **Number Removal:** Eliminates digits that do not contribute to sentiment.
- **HTML Entity Handling:** Removes HTML entities.
- **Case Normalization:** Converts all text to lowercase.
- **Punctuation Removal:** Strips out punctuation symbols.
- **Whitespace Normalization:** Cleans multiple spaces into a single space and trims the text.
- **Tokenization:** The cleaned text is then tokenized using the DistilBERT tokenizer. All texts are padded and truncated to a maximum sequence length of 128 tokens.

## Training Process
The model training process is built using the Hugging Face Transformers library with the following key components:

- **Model:** `distilbert-base-uncased` fine-tuned for sequence classification.
- **Loss Function:** Cross-entropy loss with custom class weights to address data imbalance.
- **Optimizer and Learning Rate:** A learning rate of 2e-5 is used with AdamW, incorporating weight decay.
- **Batch Sizes:** 16 samples per device for both training and evaluation.
- **Number of Epochs:** The model is trained for up to 3 epochs, with early stopping based on F1 score performance.
- **Evaluation Metrics:** Accuracy, F1 score (weighted), ROUGE, and BLEU are computed using the `evaluate` library.
- **Hardware Acceleration:** Supports GPU acceleration if available.
- **Training Pipeline:** 
  - Data is loaded from CSV files.
  - Preprocessing and tokenization are performed.
  - The data is cast into HuggingFace `Dataset` objects and formatted for PyTorch.
  - A custom training loop using a modified Trainer (with EarlyStoppingCallback) is executed.
  - Best model checkpoints and logs are saved for further evaluation and deployment.

## Results
After training, the following performance metrics were observed on the validation dataset:
- **Evaluation Loss:** ~0.2584
- **Accuracy:** ~94.60%
- **F1 Score:** ~94.61%
- **ROUGE Scores:** ROUGE-1 and ROUGE-L ~94.60%
- **BLEU Score:** 0.0 (reflecting that text-oriented metrics may vary)
  
The detailed classification report is as follows:
- **Irrelevant:** Precision 96%, Recall 91%, F1-Score 93%
- **Negative:** Precision 98%, Recall 94%, F1-Score 96%
- **Neutral:** Precision 91%, Recall 96%, F1-Score 94%
- **Positive:** Precision 94%, Recall 96%, F1-Score 95%

## Skills and Technologies Used

### Core ML & NLP
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.30.2-yellow?logo=huggingface)](https://huggingface.co/transformers/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2-FF9F1C?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![NLP](https://img.shields.io/badge/NLP-Advanced-4B32C3?logo=bookstack)](https://en.wikipedia.org/wiki/Natural_language_processing)

### Data Processing
[![pandas](https://img.shields.io/badge/pandas-1.5-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![NumPy](https://img.shields.io/badge/NumPy-1.24-013243?logo=numpy&logoColor=white)](https://numpy.org)

### Web & API
[![Flask](https://img.shields.io/badge/Flask-2.0.1-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![REST API](https://img.shields.io/badge/REST_API-FF6F00?logo=insomnia&logoColor=white)](https://en.wikipedia.org/wiki/Representational_state_transfer)
[![JavaScript](https://img.shields.io/badge/JavaScript-ES6+-F7DF1E?logo=javascript&logoColor=black)](https://developer.mozilla.org/en-US/docs/Web/JavaScript)
[![HTML5](https://img.shields.io/badge/HTML5-E34F26?logo=html5&logoColor=white)](https://developer.mozilla.org/en-US/docs/Glossary/HTML5)
[![CSS3](https://img.shields.io/badge/CSS3-1572B6?logo=css3&logoColor=white)](https://developer.mozilla.org/en-US/docs/Web/CSS)

### Deployment & DevOps
[![Docker](https://img.shields.io/badge/Docker-24.0-2496ED?logo=docker&logoColor=white)](https://docker.com)
[![Gunicorn](https://img.shields.io/badge/Gunicorn-20.1-499490?logo=gunicorn)](https://gunicorn.org)

### Development Tools
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white)](https://jupyter.org)
[![Git](https://img.shields.io/badge/Git-2.40-F05032?logo=git&logoColor=white)](https://git-scm.com)

## Conclusion
The Sentiment Analyzer project successfully demonstrates the use of modern NLP techniques and transformer architectures to perform sentiment classification on real-world data. With robust data preprocessing, an efficient training pipeline, and strong performance metrics, this system is well-suited for deployment in applications requiring rapid sentiment analysis. Future enhancements could include:
- Fine-tuning on larger and more diverse datasets.
- Integration of additional pre-trained language models.
- Expanding the web interface for broader accessibility.

## Getting Started
To get started with the project:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/sentiment-analysis.git
   cd sentiment-analysis
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Web Application:**
   ```bash
   flask run
   ```
   or use Gunicorn for a production-ready server:
   ```bash
   gunicorn app:app
   ```

4. **Training the Model:**
   Follow the instructions in `notebook/train_and_eval.ipynb` to preprocess data and train the model.

5. **Evaluate and Predict:**
   Use the provided scripts in the `notebook/` and `webapp/sentiment_evaluator.py` to evaluate performance and make predictions on new texts.

