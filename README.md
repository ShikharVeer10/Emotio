# Emotio

Emotio is a project for sentiment analysis and topic classification using transformer-based models, with additional support for keyword extraction. It leverages state-of-the-art NLP models to analyze text, detect sentiment, classify topics, and extract key information.

## Features

- **Sentiment Analysis:** Uses a fine-tuned DistilBERT model (`distilbert-base-uncased-finetuned-sst-2-english`) to determine the sentiment of text.
- **Topic Classification:** Performs zero-shot classification across a range of candidate topics, such as politics, sports, entertainment, technology, health, finance, and education.
- **Keyword Extraction:** Utilizes KeyBERT for extracting relevant keywords from text.
- **GPU Support:** Accelerates inference if a GPU is available.

## Project Structure

- `model_training.py`: Core sentiment analysis and topic classification logic.
- `keyword_extractor.py`: Implements keyword extraction functionality.
- `test.py`: (Not detailed) Example usage or testing.
- `saved_model/`: Stores trained models (if any).
- `results/`: Output results (if any).

## Installation

1. Create a Python virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

2. Install dependencies:
    ```bash
    pip install transformers torch keybert
    ```

## Usage

```python
from model_training import get_sentiment, get_topic
from keyword_extractor import extract_keywords

text = "The new technology in AI is transforming the world."

# Sentiment Analysis
sentiment = get_sentiment(text)
print(f"Sentiment: {sentiment}")

# Topic Classification
topic, score = get_topic(text)
print(f"Topic: {topic} (score: {score})")

# Keyword Extraction
keywords = extract_keywords(text, sentiment['label'])
print(f"Keywords: {keywords}")
```

## Notes

- The sentiment analysis is powered by Hugging Face's DistilBERT model.
- Topic classification uses zero-shot learning on a set of predefined topics.
- Keyword extraction is performed by KeyBERT using BERT embeddings.

## License

This project is licensed under the MIT License.

---

For more information, view the [source code on GitHub](https://github.com/ShikharVeer10/Emotio).
