import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from model_training import get_sentiment, get_topic
from keyword_extractor import extract_keywords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import glob
import os

def load_latest_twitter_results():
    pattern = "twitter_analysis_results_*.csv"
    files = glob.glob(pattern)
    if files:
        latest_file = max(files, key=os.path.getctime)
        print(f"Loading Twitter results from: {latest_file}")
        return pd.read_csv(latest_file), latest_file
    return None, None

def map_sentiment_label(label):
    label_lower = label.lower()
    if label_lower in ["positive", "pos"]:
        return "POSITIVE"
    elif label_lower in ["negative", "neg"]:
        return "NEGATIVE"
    else:
        return "NEUTRAL"

def create_vader_baseline(texts):
    analyzer = SentimentIntensityAnalyzer()
    vader_labels = []
    
    for text in texts:
        score = analyzer.polarity_scores(text)
        compound = score['compound']
        
        if compound >= 0.05:
            vader_labels.append('POSITIVE')
        elif compound <= -0.05:
            vader_labels.append('NEGATIVE')
        else:
            vader_labels.append('NEUTRAL')
    
    return vader_labels

def get_test_data():
    local_analysis_texts = [
        ("I absolutely love the new features in this app! It makes my life so much easier.", "POSITIVE"),
        ("This is the worst update ever. The app keeps crashing and nothing works.", "NEGATIVE"),
        ("The meeting is scheduled for 3 PM tomorrow.", "NEUTRAL"),
        ("Great job on the project, the results are fantastic!", "POSITIVE"),
        ("I'm really disappointed with the customer service I received.", "NEGATIVE"),
        ("The weather today is cloudy with a chance of rain.", "NEUTRAL"),
        ("Congratulations on your promotion! You totally deserve it.", "POSITIVE"),
        ("The food was cold and tasteless, not coming back here again.", "NEGATIVE"),
        ("The package arrived yesterday as expected.", "NEUTRAL"),
        ("Amazing AI breakthrough! This will revolutionize the industry.", "POSITIVE"),
        ("AI is destroying jobs and privacy. This is terrible.", "NEGATIVE"),
        ("The AI conference will be held next month.", "NEUTRAL"),
        ("Love how AI makes everything so much smarter and efficient!", "POSITIVE"),
        ("Another AI failure. When will this technology actually work?", "NEGATIVE"),
        ("AI technology continues to develop at a steady pace.", "NEUTRAL")
    ]
    
    return local_analysis_texts

def plot_confusion_matrix(cm, labels, title, filename):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count'})
    plt.title(f'{title}\nConfusion Matrix')
    plt.xlabel('Predicted Sentiment')
    plt.ylabel('True Sentiment')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure instead of showing it
    print(f"Confusion matrix saved as: {filename}")

def evaluate_model_detailed(true_labels, predicted_labels, method_name, plot_filename):
    
    labels = ['POSITIVE', 'NEGATIVE', 'NEUTRAL']
    accuracy = accuracy_score(true_labels, predicted_labels)
    cm = confusion_matrix(true_labels, predicted_labels, labels=labels)
    
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, predicted_labels, labels=labels, average=None, zero_division=0
    )
    
    macro_f1 = f1_score(true_labels, predicted_labels, average='macro', zero_division=0)
    weighted_f1 = f1_score(true_labels, predicted_labels, average='weighted', zero_division=0)
    
    print(f"\n{'='*60}")
    print(f" {method_name.upper()} EVALUATION")
    print(f"{'='*60}")
    print(f" Overall Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    print(f" Macro F1-Score: {macro_f1:.3f}")
    print(f"  Weighted F1-Score: {weighted_f1:.3f}")
    
    print(f"\n Per-Class Performance:")
    print("-" * 60)
    print(f"{'Class':<12} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
    print("-" * 60)
    
    for i, label in enumerate(labels):
        print(f"{label:<12} {precision[i]:<10.3f} {recall[i]:<10.3f} {f1[i]:<10.3f} {support[i]:<10}")

    plot_confusion_matrix(cm, labels, method_name, plot_filename)
    
    print(f"\n Classification Report:")
    print(classification_report(true_labels, predicted_labels, labels=labels, zero_division=0))
    
    return accuracy, cm, {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'per_class_precision': precision,
        'per_class_recall': recall,
        'per_class_f1': f1
    }

def evaluate_twitter_data():
    df, filename = load_latest_twitter_results()
    
    if df is None:
        print(" No Twitter results found. Please run twitter_analysis.py first.")
        return None
    
    print(f"Evaluating {len(df)} Twitter predictions...")
    cleaned_texts = df['cleaned_text'].tolist()
    vader_labels = create_vader_baseline(cleaned_texts)
    predicted_labels = [map_sentiment_label(label) for label in df['predicted_sentiment'].tolist()]
    
    print(f"Using VADER sentiment as baseline comparison")
    
    accuracy, cm, metrics = evaluate_model_detailed(
        vader_labels, predicted_labels, 
        "Twitter RoBERTa vs VADER Baseline", 
        "confusion_matrix_twitter_vs_vader.png"
    )
    results_df = df.copy()
    results_df['vader_sentiment'] = vader_labels
    results_df['predicted_mapped'] = predicted_labels
    results_df['agreement'] = [v == p for v, p in zip(vader_labels, predicted_labels)]
    
    results_df.to_csv("twitter_evaluation_results.csv", index=False)
    print(f"Detailed Twitter evaluation saved to: twitter_evaluation_results.csv")
    
    agreements = results_df[results_df['agreement'] == True]
    disagreements = results_df[results_df['agreement'] == False]
    
    print(f"\n📈 Agreement with VADER: {len(agreements)}/{len(results_df)} ({len(agreements)/len(results_df)*100:.1f}%)")
    
    if len(disagreements) > 0:
        print(f"\nSample Disagreements")
        for idx, row in disagreements.head(3).iterrows():
            print(f"\nText: {row['cleaned_text'][:100]}...")
            print(f"VADER: {row['vader_sentiment']} | RoBERTa: {row['predicted_mapped']} (score: {row['sentiment_score']:.3f})")
    
    return results_df

def evaluate_sentiment_model():
    print("🧪 Testing on local validation data...")
    test_data = get_test_data()
    true_labels = []
    predicted_labels = []
    results = []
    
    print("Analyzing texts...")
    for idx, (text, true_label) in enumerate(test_data):
        sentiment = get_sentiment(text)
        keyword_info = extract_keywords(text, sentiment)
        predicted_label = map_sentiment_label(keyword_info['sentiment_label'])
        topic_label, topic_score = get_topic(text)
        
        true_labels.append(true_label)
        predicted_labels.append(predicted_label)
        
        results.append({
            "text": text,
            "true_label": true_label,
            "predicted_label": predicted_label,
            "sentiment_score": keyword_info['sentiment_score'],
            "keywords": ", ".join(keyword_info['keywords']),
            "topic": topic_label,
            "topic_score": topic_score,
            "correct": true_label == predicted_label
        })
        
        status = '✅' if true_label == predicted_label else '❌'
        print(f"  {idx+1:2d}. {true_label} → {predicted_label} {status}")
    accuracy, cm, metrics = evaluate_model_detailed(
        true_labels, predicted_labels, 
        "Local Test Data Evaluation", 
        "confusion_matrix_local_test.png"
    )
    
    results_df = pd.DataFrame(results)
    results_df.to_csv("local_evaluation_results.csv", index=False)
    print(f"Local test results saved to: local_evaluation_results.csv")
    
    return accuracy, cm, results_df

def analyze_errors(results_df):
    errors = results_df[results_df['correct'] == False]
    
    if len(errors) > 0:
        print(f"\n🔍 ERROR ANALYSIS ({len(errors)} misclassifications)")
        print("="*60)
        
        for idx, row in errors.iterrows():
            print(f"\n Text: {row['text']}")
            print(f"   True: {row['true_label']} | Predicted: {row['predicted_label']}")
            print(f"   Confidence: {row['sentiment_score']:.3f}")
            print(f"   Keywords: {row['keywords']}")
            print("-" * 40)
    else:
        print("\n🎉 Perfect classification! No errors found.")

if __name__ == "__main__":
    print("🚀 Starting Comprehensive Model Evaluation...\n")
    print("1️⃣ LOCAL TEST DATA EVALUATION")
    print("-" * 40)
    local_accuracy, local_cm, local_results = evaluate_sentiment_model()
    analyze_errors(local_results)
    print("\n\n2️⃣ TWITTER DATA EVALUATION")
    print("-" * 40)
    twitter_results = evaluate_twitter_data()
    
    print("\n EVALUATION COMPLETED!")
    print(f"All results and plots saved in current directory")
    print(f"Key files generated:")
    print(f"   • confusion_matrix_local_test.png")
    print(f"   • confusion_matrix_twitter_vs_vader.png")
    print(f"   • local_evaluation_results.csv")
    print(f"   • twitter_evaluation_results.csv (if Twitter data available)")
