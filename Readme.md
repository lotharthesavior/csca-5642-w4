# Disaster Tweets Classification Project

## Overview

The goal is to build a machine learning model to classify X posts (tweets) as either describing real disasters (e.g., earthquakes, floods) or not (e.g., metaphorical uses like "my heart is ablaze"). This is a binary text classification task in Natural Language Processing (NLP).

## Problem Statement

The dataset consists of 7,613 training tweets and 3,263 test tweets, each with columns: `id`, `keyword` (optional), `location` (optional), `text`, and `target` (1 for disaster, 0 for non-disaster in the training set). The challenge is to predict the `target` for the test set, handling noisy text, missing values, and class imbalance (56.6% non-disaster, 43.4% disaster).

## Approach and Methodology

> Data can be downloaded from: https://www.kaggle.com/competitions/nlp-getting-started/data

The project is implemented in a Jupyter notebook (`project.ipynb`) with the following steps:

1. **Exploratory Data Analysis (EDA)**:
   - Visualized target distribution, tweet lengths, and common words using bar plots, histograms, and word clouds.
   - Identified class imbalance and missing values in `keyword` (61) and `location` (2,533).
2. **Data Cleaning**:
   - Removed URLs, mentions, hashtags, and special characters from text.
   - Converted text to lowercase, removed stop words, and applied stemming using NLTK.
   - Filled missing `keyword` values with "unknown".
3. **Feature Engineering**:
   - Tokenized text and padded sequences to a fixed length (50 words).
   - Used a randomly initialized embedding matrix (100-dimensional).
4. **Model Architecture**:
   - Built an LSTM-based sequential neural network using Keras.
   - Architecture: Embedding layer (trainable), LSTM (64–128 units), Dropout (0.2–0.3), Dense layers, and sigmoid output for binary classification.
5. **Training and Tuning**:
   - Trained the model for 20 epochs with class weights to address imbalance.
   - Performed hyperparameter tuning on LSTM units (64, 128) and dropout rates (0.2, 0.3).
6. **Evaluation**:
   - Evaluated using accuracy and F1-score on a validation set (20% of training data).
   - Visualized model performance with loss curves and confusion matrices.

## Results

- The best model (64 LSTM units, 0.2 dropout) achieved a validation accuracy of 0.76 and an F1-score of 0.70.
- Initial models failed to predict the minority class (F1=0.00), but enabling trainable embeddings and class weights resolved this issue.
- Challenges included short, noisy tweets and class imbalance, mitigated through preprocessing and class weighting.

## Outputs

- **Visualizations**: Target distribution, tweet length histogram, word cloud, loss curves, and confusion matrices.
- **Model Performance**: Hyperparameter tuning results and confusion matrices for the main and best models.
- **Submission File**: `submission.csv` with test set predictions.

## Future Improvements

- Use transformers (e.g., BERT) for better contextual understanding.
- Incorporate `keyword` features into the model.
- Apply oversampling to further address class imbalance.

