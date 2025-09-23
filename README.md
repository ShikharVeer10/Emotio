# Emotio — Project Root

Welcome to the `project-root` directory of Emotio!  
Emotio is a project for sentiment analysis and topic classification using transformer-based models, with additional support for keyword extraction. It leverages state-of-the-art NLP models to analyze text for sentiment, categorize topics, and extract relevant keywords.

---

## Features

- **Sentiment Analysis:** Fine-tuned DistilBERT model (`distilbert-base-uncased-finetuned-sst-2-english`) to determine sentiment.
- **Topic Classification:** Zero-shot classification across a range of candidate topics (e.g., politics, sports, entertainment, technology, health, finance, education).
- **Keyword Extraction:** Utilizes KeyBERT for extracting relevant keywords.
- **GPU Support:** Accelerates inference if a GPU is available.

---

## Directory Structure

- `model_training.py` — Core logic for sentiment analysis and topic classification.
- `keyword_extractor.py` — Implements keyword extraction.
- `local_analysis.py` — Local data analysis utilities.
- `test.py` — Example usage or testing script.
- `saved_model/` — Stores trained models (if any).
- `results/` — Output/results directory.
- Other files: `.gitignore`, `TODO.md`, etc.

---

## Installation

1. **Create a Python virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

2. **Install dependencies:**
    ```bash
    pip install transformers torch keybert
    ```

---

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

---

## Notes

- Sentiment analysis is powered by Hugging Face's DistilBERT model.
- Topic classification relies on zero-shot learning with a set of predefined topics.
- Keyword extraction uses KeyBERT and BERT embeddings.

---

## License

This project is licensed under the MIT License.

---

## Contributing & Maintenance

If you add new files, features, or reorganize the project-root, please update this README to keep it current.  
For more information, see the [main README](../README.md) or [source code on GitHub](https://github.com/ShikharVeer10/Emotio).

