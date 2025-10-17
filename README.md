# BBC News Classification

A machine learning project that automatically classifies BBC news articles into categories using LSTM and Naive Bayes models.

## About

This project classifies news articles into 5 categories:
- Business
- Entertainment
- Politics
- Sport
- Tech

## Models Used

1. **LSTM Neural Network** - Deep learning approach
2. **Multinomial Naive Bayes** - Traditional ML with TF-IDF

## Installation

```bash
pip install pandas numpy matplotlib seaborn
pip install nltk tensorflow scikit-learn joblib
```

Download required NLTK data:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

## Usage

1. **Load data**: Place `BBC News Train.csv` in project directory
2. **Run preprocessing**: Cleans text, removes stopwords, applies stemming/lemmatization
3. **Train models**: Choose between LSTM or Naive Bayes
4. **Make predictions**: Input any news text to get category prediction

## Example Prediction

```python
test_text = ["Argentina take on France at Lusail Stadium in the FIFA World Cup..."]
prediction = nb.predict(test_text)
# Output: sport
```

## Results

- LSTM Model: Trained with early stopping, 10 epochs
- Naive Bayes: High accuracy with TF-IDF features
- Both models saved for future use (`NB_model.pkl`)

## Files

- `BBC News Train.csv` - Training dataset (1,490 articles)
- `_finalCleaned.csv` - Cleaned data
- `_finalProcessed.csv` - Preprocessed data
- `NB_model.pkl` - Saved Naive Bayes model

## Technologies

Python • TensorFlow • Keras • NLTK • Scikit-learn • Pandas

