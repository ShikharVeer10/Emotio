from keybert import KeyBERT

kw_model = KeyBERT()

def extract_keywords(text, sentiment_label, top_n=5):
    if isinstance(sentiment_label, dict):
        label = sentiment_label.get("label")
        score = sentiment_label.get("score")
    else:
        label = sentiment_label
        score = None

    keywords = kw_model.extract_keywords(text, top_n=top_n)
    keyword_list = [kw[0] for kw in keywords]
    return {
        "sentiment_label": label,
        "sentiment_score": score,
        "keywords": keyword_list
    }
