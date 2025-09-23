import re
from model_training import get_sentiment
from keyword_extractor import extract_keywords
from model_training import get_topic
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def preprocess_tweet(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^RT\s+@\w+:\s*', '', text)
    text = re.sub(r'@(\w+)', r'\1', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

test_tweets = [
    "@MovLog_AI You sure doing great as much i seen from you 😋. And good luck to your future learning and arts. 😊",
    
    "https://t.co/8ws8y4o1P5\nThX🌐#BOSCh #ecb #Pfizer #Chevron\nmy @enilev🪔mom\nMr @BetaMoroney sir\n\n#bigdata #Linkedin\n#News #ML #finance #web3\n#uk #usa #eUrope #health #ai\n#france #frenchTECh #healthTech #ia #NVIDIA #Medical #banks #digitalhealth #SOciaLmeDia",
    
    "@Oluwatise28 @Pilot3Bot @pilot3ai The Virgens now have a home! Follow @Pilot3Bot & @pilot3ai for alpha, market insights, and smart AI responses",
    
    "RT @StefannyMusk: 1️⃣0️⃣0️⃣0️⃣Dollars in 2️⃣8️⃣Days\n\n🐬Repost and Follow\n@sentra_claus\n@AI_PTIQ\n@fhinjaai\n@drquantora\n@aitradonx\n\n[AD • NFA…",
    
    "RT @okwunjo96049: 🌱 Getting into decentralized AI has never been easier.\n\nWith Space Acres by @AutonomysNet, you don't need tokens or barri…"
]

print("=== TESTING IMPROVED SENTIMENT ANALYSIS ===\n")

for i, text in enumerate(test_tweets, 1):
    print(f"TWEET {i}:")
    print(f"Original: {text}")
    cleaned_text = preprocess_tweet(text)
    print(f"Cleaned: {cleaned_text}")
    
    # Analyze sentiment
    sentiment = get_sentiment(cleaned_text)
    keyword_info = extract_keywords(cleaned_text, sentiment)
    topic_label, topic_score = get_topic(cleaned_text)
    
    print(f"Sentiment: {keyword_info['sentiment_label']} (Score: {keyword_info['sentiment_score']:.4f})")
    print(f"Topic: {topic_label} (Score: {topic_score:.2f})")
    print("-" * 80)
    print()