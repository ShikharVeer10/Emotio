"""Quick test of the advanced sentiment model with clear output."""

import sys
import os
sys.stdout.reconfigure(encoding='utf-8')

from advanced_sentiment_model import AdvancedSentimentAnalyzer

print("=" * 70)
print("ADVANCED SENTIMENT MODEL - PERFORMANCE TEST")
print("=" * 70)

analyzer = AdvancedSentimentAnalyzer(device=-1)

test_cases = [
    # Sarcasm cases (5 cases)
    ("Oh great, another meeting that could have been an email", "NEGATIVE"),
    ("Wow, thanks for nothing, really appreciate it", "NEGATIVE"),
    ("Sure, because that worked so well last time", "NEGATIVE"),
    ("What a surprise, the train is late again", "NEGATIVE"),
    ("Yeah right, like that's ever going to happen", "NEGATIVE"),
    
    # Negation cases (4 cases)
    ("I don't hate this product at all", "POSITIVE"),
    ("This is not the worst I've seen", "NEUTRAL"),
    ("I can't say I'm disappointed", "POSITIVE"),
    ("It's not entirely useless", "NEUTRAL"),
    
    # Mixed sentiment (2 cases)
    ("The movie was visually stunning but the plot was confusing", "NEUTRAL"),
    ("I love the product quality but hate the customer service", "NEUTRAL"),
    
    # Standard cases (3 cases)
    ("I absolutely love this new feature!", "POSITIVE"),
    ("This is terrible, worst experience ever", "NEGATIVE"),
    ("The meeting is at 3 PM tomorrow", "NEUTRAL"),
]

print(f"\nTesting {len(test_cases)} cases...\n")

correct = 0
sarcasm_correct = 0
negation_correct = 0
mixed_correct = 0
standard_correct = 0

for i, (text, expected) in enumerate(test_cases):
    result = analyzer.analyze(text)
    predicted = result["predicted_class"]
    is_correct = predicted == expected
    
    if is_correct:
        correct += 1
        if i < 5:
            sarcasm_correct += 1
        elif i < 9:
            negation_correct += 1
        elif i < 11:
            mixed_correct += 1
        else:
            standard_correct += 1
    
    status = "PASS" if is_correct else "FAIL"
    print(f"[{status}] {text[:50]}...")
    print(f"       Expected: {expected} | Predicted: {predicted} | Conf: {result['confidence']:.2f}")
    
    if result["sarcasm_analysis"]["is_sarcastic"]:
        print(f"       Sarcasm detected!")
    if result["negation_analysis"]["has_negation"]:
        print(f"       Negation: {result['negation_analysis']['polarity_shift']}")
    print()

print("=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)
print(f"Overall Accuracy: {correct}/{len(test_cases)} ({correct/len(test_cases)*100:.1f}%)")
print(f"  - Sarcasm Cases: {sarcasm_correct}/5 ({sarcasm_correct/5*100:.1f}%)")
print(f"  - Negation Cases: {negation_correct}/4 ({negation_correct/4*100:.1f}%)")
print(f"  - Mixed Sentiment: {mixed_correct}/2 ({mixed_correct/2*100:.1f}%)")
print(f"  - Standard Cases: {standard_correct}/3 ({standard_correct/3*100:.1f}%)")
print("=" * 70)

# Compare with old model
print("\nCOMPARISON WITH ORIGINAL MODEL:")
print("-" * 40)
print("Original Model (from earlier evaluation):")
print("  - Overall: ~72% accuracy")
print("  - Sarcasm: 30% accuracy (3/10)")
print("  - Mixed Sentiment: 20% accuracy (2/10)")
print("-" * 40)
print(f"Advanced Model (this run):")
print(f"  - Overall: {correct/len(test_cases)*100:.1f}% accuracy")
print(f"  - Sarcasm: {sarcasm_correct/5*100:.1f}% accuracy ({sarcasm_correct}/5)")
print(f"  - Mixed: {mixed_correct/2*100:.1f}% accuracy ({mixed_correct}/2)")
print("=" * 70)
