import pandas as pd
from model_training import get_sentiment, get_topic
from keyword_extractor import extract_keywords
from keybert import KeyBERT


df = pd.read_csv("data/dataset.csv")

results = []

for text in df["text"]:
    sentiment = get_sentiment(text)
    keyword_info = extract_keywords(text, sentiment)
    topic_label, topic_score = get_topic(text)

    results.append({
        "text": text,
        "sentiment": keyword_info["sentiment_label"],
        "sentiment_score": keyword_info["sentiment_score"],
        "keywords": ", ".join(keyword_info["keywords"]),
        "topic": topic_label,
        "topic_score": topic_score
    })

results_df = pd.DataFrame(results)
results_df.to_csv("results_with_topics.csv", index=False)
print("✅ Analysis complete. Results saved to results_with_topics.csv")

