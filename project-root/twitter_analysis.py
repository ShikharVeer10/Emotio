import tweepy
import sys
import re
import pandas as pd
import json
from datetime import datetime
from getpass import getpass
from model_training import get_sentiment
from keyword_extractor import extract_keywords
from model_training import get_topic
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def preprocess_tweet(text):
    """Clean tweet text for better sentiment analysis"""
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove RT and mentions at the beginning
    text = re.sub(r'^RT\s+@\w+:\s*', '', text)
    # Keep @mentions but remove the @ symbol for processing
    text = re.sub(r'@(\w+)', r'\1', text)
    # Remove excessive hashtags (keep the word)
    text = re.sub(r'#(\w+)', r'\1', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

POSITIVE_WORDS = [
    "love", "like", "enjoy", "congratulations", "fantastic", "great", "amazing", "awesome", "deserve", "happy",
    "excellent", "wonderful", "best", "outstanding", "brilliant", "success", "appreciate", "pleased", "delighted",
    "convincing", "impressive", "helpful", "positive", "enjoyable", "satisfying", "remarkable", "exciting", "fun",
    "supportive", "creative", "valuable", "inspiring", "efficient", "productive", "innovative", "transforming",
    "opportunities", "innovation", "transparent", "trust", "collective", "early", "lucky", "selected", "gain"
]
NEGATIVE_WORDS = [
    "hate", "worst", "disappointed", "bad", "awful", "terrible", "sad", "angry", "upset", "horrible", "problem",
    "issue", "crash", "fail", "broken", "annoyed", "unhappy", "poor", "complain", "dissatisfied", "crashing",
    "boring", "confusing", "difficult", "slow", "unreliable", "frustrating", "useless", "annoying", "negative",
    "shit", "rotting", "monopoly", "black", "box", "without"
]

analyzer = SentimentIntensityAnalyzer()

def find_sentiment_trigger(text, sentiment_label):
    """Find the word that most likely triggered the sentiment classification"""
    words = text.lower().split()
    
    # First, look for explicit sentiment words
    if sentiment_label.lower() in ["positive", "pos"]:
        # Look for positive words with exact or partial matches
        for word in words:
            for pos_word in POSITIVE_WORDS:
                if pos_word in word or word in pos_word:
                    return pos_word
    elif sentiment_label.lower() in ["negative", "neg"]:
        # Look for negative words with exact or partial matches
        for word in words:
            for neg_word in NEGATIVE_WORDS:
                if neg_word in word or word in neg_word:
                    return neg_word
    
    # If no explicit words found, use VADER to find strongest sentiment word
    max_score = 0
    trigger = None
    for word in words:
        # Skip very short words and common words
        if len(word) < 3 or word in ['the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'boy', 'did', 'has', 'let', 'put', 'say', 'she', 'too', 'use']:
            continue
            
        score = analyzer.polarity_scores(word)
        
        if sentiment_label.lower() in ["positive", "pos"] and score["pos"] > max_score and score["pos"] > 0.3:
            max_score = score["pos"]
            trigger = word
        elif sentiment_label.lower() in ["negative", "neg"] and score["neg"] > max_score and score["neg"] > 0.3:
            max_score = score["neg"]
            trigger = word
        elif sentiment_label.lower() == "neutral" and abs(score["compound"]) < 0.1:
            # For neutral, look for words with balanced or low sentiment
            if abs(score["compound"]) < 0.1:
                trigger = word
    
    return trigger

if len(sys.argv) > 1:
    BEARER_TOKEN = sys.argv[1]
else:
    BEARER_TOKEN = getpass("Enter your Twitter/X Bearer Token: ")

client = tweepy.Client(bearer_token=BEARER_TOKEN)

def fetch_tweets_v2(query, count=10):
    response = client.search_recent_tweets(query=query, max_results=min(count,100), tweet_fields=["lang"])
    if response.data:
        return [tweet.text for tweet in response.data if tweet.lang == "en"]
    return []

query = "AI"  
num_tweets = 20  # Increased for better evaluation
texts = fetch_tweets_v2(query, num_tweets)

# Store results for evaluation
results = []

for text in texts:
    # Preprocess the tweet for better analysis
    cleaned_text = preprocess_tweet(text)
    
    sentiment = get_sentiment(cleaned_text)
    keyword_info = extract_keywords(cleaned_text, sentiment)
    topic_label, topic_score = get_topic(cleaned_text)
    keywords = keyword_info['keywords']

    trigger_topic = keywords[0] if keywords else ''
    trigger_sentiment = find_sentiment_trigger(cleaned_text, keyword_info['sentiment_label'])
    
    # Store result for evaluation
    result = {
        'original_text': text,
        'cleaned_text': cleaned_text,
        'predicted_sentiment': keyword_info['sentiment_label'],
        'sentiment_score': keyword_info['sentiment_score'],
        'topic': topic_label,
        'topic_score': topic_score,
        'sentiment_trigger': trigger_sentiment,
        'topic_trigger': trigger_topic,
        'timestamp': datetime.now().isoformat()
    }
    results.append(result)
    
    print(f"Original Text: {text}")
    print(f"Cleaned Text: {cleaned_text}")
    print(f"Sentiment Label: {keyword_info['sentiment_label']}")
    print(f"Sentiment Score: {keyword_info['sentiment_score']}")
    print(f"Topic: {topic_label} (score: {topic_score:.2f})")
    if trigger_sentiment:
        print(f"Trigger Keyword for Sentiment ({keyword_info['sentiment_label']}): {trigger_sentiment}")
    else:
        print(f"No trigger keyword found for Sentiment ({keyword_info['sentiment_label']})")
    if trigger_topic:
        print(f"Trigger Keyword for Topic ({topic_label}): {trigger_topic}")
    else:
        print(f"No trigger keyword found for Topic ({topic_label})")
    print("-" * 60)

# Save results to CSV for evaluation
df_results = pd.DataFrame(results)
results_file = f"twitter_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
df_results.to_csv(results_file, index=False)
print(f"\n📊 Results saved to: {results_file}")
print(f"📈 Total tweets analyzed: {len(results)}")

# Print summary statistics
sentiment_counts = df_results['predicted_sentiment'].value_counts()
print(f"\n📊 Sentiment Distribution:")
for sentiment, count in sentiment_counts.items():
    print(f"  {sentiment}: {count} ({count/len(results)*100:.1f}%)")

print(f"\n✅ To evaluate model performance, you can now manually label the results in '{results_file}' and run the evaluation script.")