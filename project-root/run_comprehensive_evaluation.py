"""
Comprehensive Model Evaluation Runner

This script runs all evaluations addressing reviewer concerns and generates
a complete report suitable for academic publication.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_training import get_sentiment, get_topic
from keyword_extractor import extract_keywords
from hybrid_fusion import MultiModelSentimentAnalyzer, HybridFusionStrategy
from enhanced_evaluation import (
    LatencyBenchmark,
    MemoryProfiler, 
    InterAnnotatorReliability,
    SarcasmMixedSentimentEvaluator,
    ClassImbalanceAnalyzer,
    ConceptDriftDetector,
    run_comprehensive_evaluation
)


def get_extended_test_data():
    """
    Extended test dataset with more samples and edge cases.
    
    Addresses Point 3: Larger evaluation dataset for better validation.
    """
    test_data = [
        # Standard Positive
        ("I absolutely love the new features in this app! It makes my life so much easier.", "POSITIVE"),
        ("Great job on the project, the results are fantastic!", "POSITIVE"),
        ("Congratulations on your promotion! You totally deserve it.", "POSITIVE"),
        ("Amazing AI breakthrough! This will revolutionize the industry.", "POSITIVE"),
        ("Love how AI makes everything so much smarter and efficient!", "POSITIVE"),
        ("This is the best product I've ever used, highly recommend!", "POSITIVE"),
        ("Excellent customer service, they solved my problem immediately.", "POSITIVE"),
        ("The new update is incredible, everything works perfectly now.", "POSITIVE"),
        ("I'm so happy with my purchase, exceeded all expectations.", "POSITIVE"),
        ("What a wonderful experience, will definitely come back.", "POSITIVE"),
        
        # Standard Negative
        ("This is the worst update ever. The app keeps crashing and nothing works.", "NEGATIVE"),
        ("I'm really disappointed with the customer service I received.", "NEGATIVE"),
        ("The food was cold and tasteless, not coming back here again.", "NEGATIVE"),
        ("AI is destroying jobs and privacy. This is terrible.", "NEGATIVE"),
        ("Another AI failure. When will this technology actually work?", "NEGATIVE"),
        ("Terrible experience, wasted my money on this garbage.", "NEGATIVE"),
        ("The product broke after one day, complete waste of money.", "NEGATIVE"),
        ("Worst customer support ever, they hung up on me twice.", "NEGATIVE"),
        ("I hate this new interface, it's confusing and slow.", "NEGATIVE"),
        ("What a disaster, nothing about this product works right.", "NEGATIVE"),
        
        # Standard Neutral
        ("The meeting is scheduled for 3 PM tomorrow.", "NEUTRAL"),
        ("The weather today is cloudy with a chance of rain.", "NEUTRAL"),
        ("The package arrived yesterday as expected.", "NEUTRAL"),
        ("The AI conference will be held next month.", "NEUTRAL"),
        ("AI technology continues to develop at a steady pace.", "NEUTRAL"),
        ("The office will be closed on Monday for maintenance.", "NEUTRAL"),
        ("Please submit your reports by end of day Friday.", "NEUTRAL"),
        ("The quarterly earnings will be announced next week.", "NEUTRAL"),
        ("Today's agenda includes three main discussion topics.", "NEUTRAL"),
        ("The document has been updated with the latest figures.", "NEUTRAL"),
        
        # Sarcasm cases (challenging - should be detected as NEGATIVE despite positive words)
        ("Oh great, another meeting that could have been an email.", "NEGATIVE"),
        ("Wow, thanks for nothing, really appreciate it.", "NEGATIVE"),
        ("Sure, because that worked so well last time.", "NEGATIVE"),
        ("What a surprise, the train is late again.", "NEGATIVE"),
        ("Yeah right, like that's ever going to happen.", "NEGATIVE"),
        
        # Mixed sentiment (challenging - both positive and negative elements)
        ("The movie was visually stunning but the plot was confusing.", "NEUTRAL"),
        ("I love the product quality but hate the customer service.", "NEUTRAL"),
        ("Great features but way too expensive for what it offers.", "NEUTRAL"),
        ("Nice design but terrible user experience.", "NEUTRAL"),
        ("Good intentions but poor execution.", "NEUTRAL"),
        
        # Subtle/Implicit sentiment
        ("After three months, I finally got a response.", "NEGATIVE"),  # Implicit frustration
        ("The team exceeded the original timeline by two weeks.", "NEUTRAL"),  # Could be neg/pos
        ("They did exactly what was asked, nothing more.", "NEUTRAL"),  # Subtle criticism
        ("The product functions as described in the manual.", "NEUTRAL"),  # Neutral-positive
        ("Results were within acceptable parameters.", "NEUTRAL"),  # Technical neutral
        
        # Edge cases with negation
        ("I don't hate this product at all.", "POSITIVE"),  # Double negative
        ("This is not the worst I've seen.", "NEUTRAL"),  # Negated negative
        ("It's not entirely useless.", "NEUTRAL"),  # Weak praise
        ("I can't say I'm disappointed.", "POSITIVE"),  # Negated negative = positive
        ("The results were not unexpected.", "NEUTRAL"),  # Double negative = neutral
    ]
    
    return test_data


def map_sentiment_label(label):
    """Normalize sentiment labels."""
    label_lower = label.lower()
    if label_lower in ["positive", "pos"]:
        return "POSITIVE"
    elif label_lower in ["negative", "neg"]:
        return "NEGATIVE"
    else:
        return "NEUTRAL"


def sentiment_wrapper(text):
    """Wrapper function for get_sentiment to normalize output."""
    result = get_sentiment(text)
    return map_sentiment_label(result['label'])


def run_complete_evaluation():
    """Run complete evaluation with all enhancements."""
    print("=" * 70)
    print(" EMOTIO - COMPREHENSIVE MODEL EVALUATION")
    print(" Addressing All Reviewer Concerns")
    print("=" * 70)
    print(f"\nTimestamp: {datetime.now().isoformat()}")
    
    # Load test data
    test_data = get_extended_test_data()
    texts = [t[0] for t in test_data]
    true_labels = [t[1] for t in test_data]
    
    print(f"\n📊 Dataset: {len(test_data)} samples")
    print(f"   - Positive: {sum(1 for l in true_labels if l == 'POSITIVE')}")
    print(f"   - Negative: {sum(1 for l in true_labels if l == 'NEGATIVE')}")
    print(f"   - Neutral: {sum(1 for l in true_labels if l == 'NEUTRAL')}")
    
    # Run comprehensive evaluation
    print("\n" + "=" * 70)
    print(" RUNNING EVALUATIONS...")
    print("=" * 70)
    
    report = run_comprehensive_evaluation(
        sentiment_func=sentiment_wrapper,
        test_texts=texts,
        true_labels=true_labels,
        output_dir="."
    )
    
    # Additional: Test hybrid fusion
    print("\n\n" + "=" * 70)
    print(" HYBRID FUSION EVALUATION")
    print("=" * 70)
    
    try:
        fusion_strategy = HybridFusionStrategy()
        analyzer = MultiModelSentimentAnalyzer(device=-1, fusion_strategy=fusion_strategy)
        
        print("\nTesting hybrid fusion on sample texts...")
        
        fusion_results = []
        for text, true_label in test_data[:10]:  # Test on subset
            try:
                result = analyzer.analyze(text, models=["roberta", "vader"])
                fusion_results.append({
                    "text": text[:50] + "...",
                    "true": true_label,
                    "predicted": result.predicted_class,
                    "confidence": result.confidence,
                    "correct": result.predicted_class == true_label
                })
            except Exception as e:
                print(f"Error on text: {text[:30]}... - {e}")
        
        if fusion_results:
            fusion_df = pd.DataFrame(fusion_results)
            accuracy = fusion_df["correct"].mean()
            print(f"\nHybrid Fusion Accuracy: {accuracy*100:.1f}%")
            
    except Exception as e:
        print(f"Hybrid fusion evaluation skipped: {e}")
    
    # Concept Drift Simulation
    print("\n\n" + "=" * 70)
    print(" CONCEPT DRIFT ANALYSIS")
    print("=" * 70)
    
    drift_detector = ConceptDriftDetector(window_size=10, drift_threshold=0.15)
    
    # Simulate temporal windows
    window_size = 10
    for i in range(0, len(test_data), window_size):
        window_data = test_data[i:i+window_size]
        if len(window_data) < window_size:
            break
        
        window_texts = [t[0] for t in window_data]
        window_true = [t[1] for t in window_data]
        window_pred = [sentiment_wrapper(t) for t in window_texts]
        
        drift_detector.add_window(window_pred, window_true)
    
    drift_result = drift_detector.detect_drift()
    print(f"\n   Drift Detected: {'Yes ⚠️' if drift_result['drift_detected'] else 'No ✓'}")
    print(f"   Baseline Accuracy: {drift_result['baseline_accuracy']*100:.1f}%")
    print(f"   Recent Accuracy: {drift_result['recent_accuracy']*100:.1f}%")
    print(f"   Accuracy Change: {-drift_result['accuracy_drop']*100:+.1f}%")
    print(f"\n   Recommendation: {drift_result['recommendation']}")
    
    # Save comprehensive report
    print("\n\n" + "=" * 70)
    print(" SAVING REPORTS")
    print("=" * 70)
    
    # Save as JSON
    import json
    
    report_filename = f"comprehensive_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # Convert non-serializable items
    def make_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        return obj
    
    serializable_report = make_serializable(report)
    
    with open(report_filename, 'w') as f:
        json.dump(serializable_report, f, indent=2, default=str)
    
    print(f"   ✓ Report saved: {report_filename}")
    
    print("\n" + "=" * 70)
    print(" EVALUATION COMPLETE")
    print("=" * 70)
    print("\nFiles generated:")
    print(f"   • {report_filename}")
    print("\nAddresses reviewer points: 2, 4, 5, 7, 8, 9, 10")
    
    return report


if __name__ == "__main__":
    run_complete_evaluation()
