# Sentiment Analysis on Movie Reviews

[![License](https://img.shields.io/badge/License-Apache_2.0-0D76A8?style=flat)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.7.12](https://img.shields.io/badge/Python-3.7.12-green.svg)](https://shields.io/)

## Project Overview

This project performs sentiment analysis on IMDB movie reviews using machine learning models. The objective is to classify each review as positive or negative and identify which algorithm is most effective for binary text classification. The workflow includes text preprocessing, TF-IDF vectorization, model training, and comparative evaluation across several supervised learning methods.

## Key Tasks

- Load and explore the IMDB Movie Review dataset (50,000 reviews)
- Apply data cleaning: remove HTML tags, normalize case, tokenize text
- Remove stopwords and apply stemming (Porter Stemmer)
- Convert text to numerical features using TF-IDF
- Split data into training (75%) and testing (25%) subsets
- Train models:
    - Logistic Regression
    - Linear Support Vector Classifier (LinearSVC)
    - K-Nearest Neighbors (KNN)
    - Multi-Layer Perceptron (MLP)
- Evaluate models and compare accuracy
- Generate confusion matrix for best-performing model
- Discuss strengths and limitations of approaches

## Setup

1. **Install dependencies**  
   Install directly:
   ```bash
   pip install numpy pandas scikit-learn matplotlib seaborn nltk
   ```

2. **Run the notebook**  
   Open `Sentiment_Analysis_IMDB.ipynb` in Jupyter, GCP or VS Code and run all cells.

## Data Sources

- **Dataset:** [IMDB Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)
    - 50,000 labeled movie reviews (balanced: 25k positive, 25k negative)
    - Common benchmark for text classification

## Outputs

- Cleaned and preprocessed review dataset
- TF-IDF vectorized text features
- Performance comparison across models (accuracy metrics and plots)
- Confusion matrix for Logistic Regression model
- Visualizations: sentiment distribution and word clouds

## Results
- **Best Model:** Logistic Regression (88.52% accuracy)
- **Key Findings:**
    - Linear models (Logistic Regression, LinearSVC) outperform KNN and MLP for this type of task
    - TF-IDF features effectively represent word importance
    - Future improvements could include embeddings (Word2Vec, GloVe) and Transformer-based models (BERT)

## Author

[Alex Chen Hsieh](https://www.linkedin.com/in/alex-chen-hsieh/)

---

*Project created as part of Introduction to Data Mining (CS 436) at Binghamton University, 2025*
