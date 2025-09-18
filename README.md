# Sentiment Analysis with Transformers

This project provides a sentiment analysis and topic classification pipeline using pre-trained transformer models. It also includes keyword extraction functionality using KeyBERT.

## Features

- Sentiment analysis using a fine-tuned DistilBERT model (`distilbert-base-uncased-finetuned-sst-2-english`).
- Topic classification using zero-shot classification with candidate topics: politics, sports, entertainment, technology, health, finance, education.
- Keyword extraction from text using KeyBERT.
- GPU support if available for faster inference.

## Project Structure

- `project-root/model_training.py`: Contains the sentiment analysis and topic classification pipelines and functions.
- `project-root/keyword_extractor.py`: Implements keyword extraction using KeyBERT.
- `project-root/test.py`: (Not detailed here) Presumably contains testing or example usage code.
- `project-root/saved_model/`: Directory for saved models (if applicable).
- `results/`: Directory for output results (if applicable).

## Installation

1. Create a Python virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install required packages:

```bash
pip install transformers torch keybert
```

## Usage

Import the functions from the modules and use them as follows:

```python
from model_training import get_sentiment, get_topic
from keyword_extractor import extract_keywords

text = "The new technology in AI is transforming the world."

# Get sentiment
sentiment = get_sentiment(text)
print(f"Sentiment: {sentiment}")

# Get topic
topic, score = get_topic(text)
print(f"Topic: {topic} (score: {score})")

# Extract keywords
keywords = extract_keywords(text, sentiment['label'])
print(f"Keywords: {keywords}")
```

## Notes

- The sentiment analysis model uses the Hugging Face `distilbert-base-uncased-finetuned-sst-2-english` model.
- The topic classification uses zero-shot classification with predefined candidate topics.
- Keyword extraction uses KeyBERT, which extracts keywords based on BERT embeddings.

## License

This project is licensed under the MIT License.
