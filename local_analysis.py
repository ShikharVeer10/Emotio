from model_training import get_sentiment, get_topic
from keyword_extractor import extract_keywords

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
    highlighted = keywords[0] if keywords else ''
    print(f"Text: {text}")
    print(f"Sentiment Label: {keyword_info['sentiment_label']}")
    print(f"Sentiment Score: {keyword_info['sentiment_score']}")
    print(f"Topic: {topic_label} (score: {topic_score:.2f})")
    print(f"Trigger Keyword: {highlighted}")
    print("-" * 40)
