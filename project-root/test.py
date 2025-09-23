import re
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

# Test the problematic tweets
test_tweets = [
    "https://t.co/8ws8y4o1P5\nThX🌐#BOSCh #ecb #Pfizer #Chevron\nmy @enilev🪔mom\nMr @BetaMoroney sir\n\n#bigdata #Linkedin \n#News #ML #finance #web3\n#uk #usa #eUrope #health #ai\n#france #frenchTECh #healthTech #ia #NVIDIA #Medical #banks #digitalhealth #SOciaLmeDia\nhttps://t.co/wyDcK8wES3",
    
    "RT @StefannyMusk: 1️⃣0️⃣0️⃣0️⃣Dollars in 2️⃣8️⃣Days\n\n🐬Repost and Follow\n@sentra_claus\n@AI_PTIQ\n@fhinjaai\n@drquantora\n@aitradonx\n\n[AD • NFA…",
    
    "RT @okwunjo96049: 🌱 Getting into decentralized AI has never been easier.\n\nWith Space Acres by @AutonomysNet, you don't need tokens or barri…",
    
    "RT @ZakaZ96: I don't know if it's the result of the AI shit rotting YouTube away, but stuff like this has now gotten me feeling \"well, I'm…"
]

print("Testing improved sentiment analysis:\n")

for i, text in enumerate(test_tweets, 1):
    # Preprocess the tweet for better analysis
    cleaned_text = preprocess_tweet(text)
    
    sentiment = get_sentiment(cleaned_text)
    keyword_info = extract_keywords(cleaned_text, sentiment)
    topic_label, topic_score = get_topic(cleaned_text)
    keywords = keyword_info['keywords']

    trigger_topic = keywords[0] if keywords else ''
    trigger_sentiment = find_sentiment_trigger(cleaned_text, keyword_info['sentiment_label'])
    
    print(f"Tweet {i}:")
    print(f"Original: {text[:100]}...")
    print(f"Cleaned: {cleaned_text}")
    print(f"Sentiment: {keyword_info['sentiment_label']} (score: {keyword_info['sentiment_score']:.3f})")
    print(f"Topic: {topic_label} (score: {topic_score:.2f})")
    print(f"Sentiment Trigger: {trigger_sentiment}")
    print(f"Topic Trigger: {trigger_topic}")
    print("-" * 80)
