# Emotio

Emotio is a Python sentiment-analysis project for general text and Twitter/X content. It combines pretrained transformer models with VADER, topic classification, keyword extraction, sarcasm and negation heuristics, and evaluation utilities.

## What is implemented

- **Baseline sentiment:** Cardiff NLP's `twitter-roberta-base-sentiment-latest`
- **Topic classification:** zero-shot classification with a built-in list of 36 topic labels
- **Keyword extraction:** KeyBERT
- **Advanced analysis:** RoBERTa, VADER, and DistilBERT combined with cross-attention-style fusion
- **Sarcasm and negation analysis:** rule-based detectors used by the advanced analyzer
- **Weighted fusion:** optional RoBERTa, VADER, and BART-MNLI ensemble in `hybrid_fusion.py`
- **Twitter/X analysis:** recent English tweets retrieved with Tweepy and exported to CSV
- **Evaluation:** accuracy, macro/weighted F1, confusion matrices, latency, throughput, memory, class imbalance, reliability, sarcasm/mixed-sentiment, and concept-drift utilities
- **CPU/GPU support:** Transformers pipelines use GPU when available in the baseline utilities; analyzers accept a Transformers device ID

## Repository layout

```text
Emotio/
├── README.md
├── MANUSCRIPT_REVISION_GUIDE.md
├── test.py
└── project-root/
    ├── model_training.py                 # Baseline sentiment and topic pipelines
    ├── keyword_extractor.py              # KeyBERT extraction
    ├── advanced_sentiment_model.py       # Sarcasm-aware advanced analyzer
    ├── hybrid_fusion.py                  # Weighted multi-model fusion
    ├── local_analysis.py                 # Example local text analysis
    ├── twitter_analysis.py               # Twitter/X API analysis
    ├── model_evaluation.py               # Local and Twitter evaluation
    ├── enhanced_evaluation.py            # Reusable evaluation metrics
    ├── run_comprehensive_evaluation.py   # End-to-end evaluation runner
    ├── test_sentiment.py                 # Tweet preprocessing example
    ├── quick_test.py                     # Advanced analyzer test
    └── TODO.md
```

Generated CSV, JSON, log, and model files are ignored by `.gitignore`. Existing PNG files in `project-root/` are checked-in evaluation artifacts.

## Requirements and installation

Python 3.8+ is recommended. The repository does not currently include a `requirements.txt`, so install the runtime dependencies directly:

```bash
git clone https://github.com/ShikharVeer10/Emotio.git
cd Emotio
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install torch transformers keybert sentence-transformers \
    vaderSentiment scipy numpy pandas scikit-learn matplotlib seaborn psutil tweepy
```

The first use downloads the configured Hugging Face and KeyBERT models. Use a CUDA-compatible PyTorch installation if GPU inference is required.

## Quick start

Run commands from the repository root. The baseline module loads its models when imported:

```python
import sys
sys.path.insert(0, "project-root")

from model_training import get_sentiment, get_topic
from keyword_extractor import extract_keywords

text = "I love this new feature!"
sentiment = get_sentiment(text)
topic, topic_score = get_topic(text)
keywords = extract_keywords(text, sentiment)

print(sentiment)                    # {'label': ..., 'score': ...}
print(topic, topic_score)
print(keywords["keywords"])
```

For the advanced analyzer:

```python
import sys
sys.path.insert(0, "project-root")

from advanced_sentiment_model import AdvancedSentimentAnalyzer

analyzer = AdvancedSentimentAnalyzer(device=-1)  # -1 = CPU; 0 = first GPU
result = analyzer.analyze("Oh great, another meeting that could have been an email.")

print(result["predicted_class"])
print(result["confidence"])
print(result["sarcasm_analysis"])
print(result["negation_analysis"])
print(result["explanation"])
```

The advanced result also contains class probabilities, model contributions, an attention matrix, individual model predictions, and the generated explanation.

## Command-line scripts

From the repository root:

```bash
# Example local analysis
python project-root/local_analysis.py

# Advanced analyzer smoke test
python project-root/quick_test.py

# Baseline evaluation on the built-in local validation set
python project-root/model_evaluation.py

# Comprehensive metrics and a JSON report
python project-root/run_comprehensive_evaluation.py
```

`model_evaluation.py` writes `local_evaluation_results.csv` and confusion-matrix images. The comprehensive runner writes a timestamped `comprehensive_evaluation_*.json` report and additional plots.

## Twitter/X analysis

`twitter_analysis.py` uses Tweepy `Client.search_recent_tweets`, requests the recent English tweets matching the hard-coded query `AI`, and analyzes up to 100 tweets. Pass the bearer token as the first argument, or enter it when prompted:

```bash
python project-root/twitter_analysis.py YOUR_BEARER_TOKEN
```

The script removes URLs, leading retweet markers, mention symbols, and hashtag symbols before analysis. It writes a timestamped `twitter_analysis_results_*.csv` containing the original and cleaned text, sentiment, topic, scores, trigger terms, and timestamp. Run `model_evaluation.py` afterward to compare predictions with a VADER baseline when Twitter results are available.

## Weighted hybrid fusion

`hybrid_fusion.py` provides `MultiModelSentimentAnalyzer`, which can combine:

- Cardiff NLP Twitter RoBERTa
- VADER
- BART-MNLI zero-shot sentiment classification

The default model weights are RoBERTa `0.45`, VADER `0.25`, and BART-MNLI `0.30`. Confidence weighting is enabled by default:

```python
import sys
sys.path.insert(0, "project-root")

from hybrid_fusion import HybridFusionStrategy, MultiModelSentimentAnalyzer

strategy = HybridFusionStrategy(
    model_weights={"roberta": 0.45, "vader": 0.25, "bart_mnli": 0.30}
)
analyzer = MultiModelSentimentAnalyzer(device=-1, fusion_strategy=strategy)
result = analyzer.analyze("Great product, but terrible support.")
print(result.predicted_class, result.confidence)
```

## Limitations

- Models are loaded from external pretrained checkpoints and require network access on first use.
- Twitter/X analysis requires a valid bearer token and access to the recent-search endpoint.
- The project currently has no packaged dependency file, formal test runner, or `LICENSE` file.
- Evaluation examples use small, hand-written datasets and should not be treated as benchmark results.

## Research notes

`MANUSCRIPT_REVISION_GUIDE.md` contains the project's manuscript-writing and evaluation guidance. The implementation is intended for experimentation and analysis rather than a production API.

## Citation

```bibtex
@software{emotio,
  author = {Shikhar Veer},
  title = {Emotio: Hybrid Sentiment Analysis for Text and Twitter/X Content},
  year = {2024},
  url = {https://github.com/ShikharVeer10/Emotio}
}
```
