from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch

SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"

device = 0 if torch.cuda.is_available() else -1
print(f"Device set to: {'GPU' if device == 0 else 'CPU'}")

tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL)

sentiment_pipeline = pipeline(
    "sentiment-analysis", model=model, tokenizer=tokenizer, device=device
)

def get_sentiment(text):
    return sentiment_pipeline(text)[0]

from transformers import pipeline

topic_pipeline = pipeline("zero-shot-classification", device=device)
CANDIDATE_TOPICS = ["politics", "sports", "entertainment", "technology", "health", "finance", "education"]

def get_topic(text):
    result = topic_pipeline(text, candidate_labels=CANDIDATE_TOPICS)
    return result["labels"][0], result["scores"][0]
