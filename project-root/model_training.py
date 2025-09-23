from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch

# Twitter-specific sentiment model (better for social media)
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

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

CANDIDATE_TOPICS = [
    "politics", "sports", "entertainment", "technology", "health", "finance", "education",
    "business", "cryptocurrency", "blockchain", "ai", "machine learning", "customer service",
    "food", "weather", "events", "travel", "science", "research", "productivity", "marketing",
    "social media", "startups", "career", "promotion", "project", "app", "service", "feedback",
    "conference", "networking", "automation", "workflow", "community", "announcement"
]

def get_topic(text):
    result = topic_pipeline(text, candidate_labels=CANDIDATE_TOPICS)
    return result["labels"][0], result["scores"][0]
