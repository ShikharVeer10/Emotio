from model_training import get_sentiment, get_topic
from keyword_extractor import extract_keywords

POSITIVE_WORDS = [
    "love", "like", "enjoy", "congratulations", "fantastic", "great", "amazing", "awesome", "deserve", "happy",
    "excellent", "wonderful", "best", "outstanding", "brilliant", "success", "appreciate", "pleased", "delighted"
]
NEGATIVE_WORDS = [
    "hate", "worst", "disappointed", "bad", "awful", "terrible", "sad", "angry", "upset", "horrible", "problem",
    "issue", "crash", "fail", "broken", "annoyed", "unhappy", "poor", "complain", "dissatisfied", "crashing"
]

def find_sentiment_trigger(text, sentiment_label):
    words = text.lower().split()
    if sentiment_label == "POSITIVE":
        for w in POSITIVE_WORDS:
            if w in words or any(w in word for word in words):
                return w
    elif sentiment_label == "NEGATIVE":
        for w in NEGATIVE_WORDS:
            if w in words or any(w in word for word in words):
                return w
    return None

texts = [
    "I absolutely love the new features in this app! It makes my life so much easier.",
    "This is the worst update ever. The app keeps crashing and nothing works.",
    "The meeting is scheduled for 3 PM tomorrow.",
    "Great job on the project, the results are fantastic!",
    "I'm really disappointed with the customer service I received.",
    "The weather today is cloudy with a chance of rain.",
    "Congratulations on your promotion! You totally deserve it.",
    "The food was cold and tasteless, not coming back here again.",
    "The package arrived yesterday as expected."
]

for text in texts:
    sentiment = get_sentiment(text)
    keyword_info = extract_keywords(text, sentiment)
    topic_label, topic_score = get_topic(text)
    keywords = keyword_info['keywords']

    # Trigger for sentiment: best matching sentiment word, else fallback to top keyword
    trigger_sentiment = find_sentiment_trigger(text, keyword_info['sentiment_label'])
    if not trigger_sentiment:
        trigger_sentiment = keywords[0] if keywords else ''

    print(f"Text: {text}")
    print(f"Sentiment Label: {keyword_info['sentiment_label']}")
    print(f"Sentiment Score: {keyword_info['sentiment_score']}")
    print(f"Topic: {topic_label} (score: {topic_score:.2f})")
    if trigger_sentiment:
        print(f"Trigger Keyword for Sentiment ({keyword_info['sentiment_label']}): {trigger_sentiment}")
    else:
        print(f"No trigger keyword found for Sentiment ({keyword_info['sentiment_label']})")
    print("-" * 40)