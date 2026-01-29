"""
Hybrid Decision Fusion Strategy for Multi-Model Sentiment Analysis

This module implements a mathematically defined fusion strategy that combines
predictions from multiple sentiment analysis models (RoBERTa, VADER, BART-MNLI)
using weighted ensemble with confidence calibration.

Mathematical Formulation:
-------------------------
Let M = {m1, m2, ..., mn} be the set of n models.
For each model mi, let:
    - p_i(c) = probability assigned to class c
    - w_i = learnable weight for model mi (Σw_i = 1)
    - conf_i = confidence score of model mi

The final fused prediction is computed as:
    P_fused(c) = Σ(i=1 to n) [w_i * conf_i * p_i(c)] / Σ(i=1 to n) [w_i * conf_i]

Decision Rule:
    y_pred = argmax_c P_fused(c)

Confidence Calibration:
    conf_calibrated = σ(α * conf_raw + β)
    where σ is the sigmoid function, α and β are calibration parameters
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import torch
from scipy.special import softmax
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class SentimentClass(Enum):
    """Enumeration of sentiment classes."""
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"


@dataclass
class ModelPrediction:
    """Container for individual model predictions."""
    model_name: str
    class_probabilities: Dict[str, float]  # {class_name: probability}
    confidence: float
    raw_output: Optional[Dict] = None


@dataclass
class FusedPrediction:
    """Container for fused prediction results."""
    predicted_class: str
    fused_probabilities: Dict[str, float]
    confidence: float
    model_contributions: Dict[str, float]  # Weight contribution of each model
    individual_predictions: List[ModelPrediction] = field(default_factory=list)


class ConfidenceCalibrator:
    
    def __init__(self, alpha: float = 1.0, beta: float = 0.0):
        
        self.alpha = alpha
        self.beta = beta
    
    def calibrate(self, raw_confidence: float) -> float:
       
        logit = self.alpha * raw_confidence + self.beta
        return 1.0 / (1.0 + np.exp(-logit))
    
    def fit(self, raw_scores: np.ndarray, true_labels: np.ndarray):
       
        from scipy.optimize import minimize
        
        def neg_log_likelihood(params):
            alpha, beta = params
            calibrated = 1.0 / (1.0 + np.exp(-(alpha * raw_scores + beta)))
            epsilon = 1e-10
            calibrated = np.clip(calibrated, epsilon, 1 - epsilon)
            nll = -np.mean(
                true_labels * np.log(calibrated) + 
                (1 - true_labels) * np.log(1 - calibrated)
            )
            return nll
        
        result = minimize(neg_log_likelihood, [1.0, 0.0], method='BFGS')
        self.alpha, self.beta = result.x


class HybridFusionStrategy:
    
    def __init__(
        self,
        model_weights: Optional[Dict[str, float]] = None,
        use_confidence_weighting: bool = True
    ):
        self.model_weights = model_weights or {}
        self.use_confidence_weighting = use_confidence_weighting
        self.calibrators: Dict[str, ConfidenceCalibrator] = {}
        self.classes = [e.value for e in SentimentClass]
        
        # Default weights based on empirical performance
        self.default_weights = {
            "roberta": 0.45,      # Strong on context-dependent sentiment
            "vader": 0.25,        # Good for explicit sentiment words
            "bart_mnli": 0.30     # Robust for nuanced/implicit sentiment
        }
    
    def register_calibrator(self, model_name: str, calibrator: ConfidenceCalibrator):
        self.calibrators[model_name] = calibrator
    
    def get_weight(self, model_name: str) -> float:
        """Get weight for a model, using default if not specified."""
        if model_name in self.model_weights:
            return self.model_weights[model_name]
        return self.default_weights.get(model_name, 1.0 / 3.0)
    
    def fuse_predictions(
        self,
        predictions: List[ModelPrediction]
    ) -> FusedPrediction:
       
        if not predictions:
            raise ValueError("At least one prediction is required for fusion")
        
        fused_probs = {c: 0.0 for c in self.classes}
        normalization = 0.0
        model_contributions = {}
        
        for pred in predictions:
            weight = self.get_weight(pred.model_name)
            
            # Apply confidence calibration if available
            if pred.model_name in self.calibrators:
                confidence = self.calibrators[pred.model_name].calibrate(pred.confidence)
            else:
                confidence = pred.confidence
            
            # Compute effective weight
            if self.use_confidence_weighting:
                effective_weight = weight * confidence
            else:
                effective_weight = weight
            
            model_contributions[pred.model_name] = effective_weight
            normalization += effective_weight
            
            # Accumulate weighted probabilities
            for cls, prob in pred.class_probabilities.items():
                if cls in fused_probs:
                    fused_probs[cls] += effective_weight * prob
        
        # Normalize probabilities
        if normalization > 0:
            fused_probs = {c: p / normalization for c, p in fused_probs.items()}
            model_contributions = {m: w / normalization for m, w in model_contributions.items()}
        
        # Determine final prediction
        predicted_class = max(fused_probs, key=fused_probs.get)
        final_confidence = fused_probs[predicted_class]
        
        return FusedPrediction(
            predicted_class=predicted_class,
            fused_probabilities=fused_probs,
            confidence=final_confidence,
            model_contributions=model_contributions,
            individual_predictions=predictions
        )
    
    def ablation_study(
        self,
        predictions: List[ModelPrediction],
        true_label: str
    ) -> Dict[str, Dict]:
       
        from itertools import combinations
        
        results = {}
        model_names = [p.model_name for p in predictions]
        
        # Test each subset size
        for r in range(1, len(predictions) + 1):
            for combo in combinations(range(len(predictions)), r):
                subset = [predictions[i] for i in combo]
                subset_names = tuple(model_names[i] for i in combo)
                
                fused = self.fuse_predictions(subset)
                is_correct = fused.predicted_class == true_label
                
                results[subset_names] = {
                    "predicted": fused.predicted_class,
                    "confidence": fused.confidence,
                    "correct": is_correct,
                    "model_contributions": fused.model_contributions
                }
        
        return results


class MultiModelSentimentAnalyzer:
    
    
    def __init__(
        self,
        device: int = -1,
        fusion_strategy: Optional[HybridFusionStrategy] = None
    ):
        """
        Initialize multi-model analyzer.
        
        Args:
            device: GPU device ID (-1 for CPU)
            fusion_strategy: Custom fusion strategy (uses default if None)
        """
        self.device = device
        self.fusion_strategy = fusion_strategy or HybridFusionStrategy()
        
        # Initialize models lazily
        self._roberta_pipeline = None
        self._bart_pipeline = None
        self._vader_analyzer = SentimentIntensityAnalyzer()
        
        # Label mappings
        self.roberta_label_map = {
            "positive": "POSITIVE",
            "negative": "NEGATIVE", 
            "neutral": "NEUTRAL"
        }
    
    @property
    def roberta_pipeline(self):
        """Lazy initialization of RoBERTa pipeline."""
        if self._roberta_pipeline is None:
            from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
            
            model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            self._roberta_pipeline = pipeline(
                "sentiment-analysis",
                model=model,
                tokenizer=tokenizer,
                device=self.device,
                return_all_scores=True
            )
        return self._roberta_pipeline
    
    @property
    def bart_pipeline(self):
        """Lazy initialization of BART-MNLI pipeline."""
        if self._bart_pipeline is None:
            from transformers import pipeline
            self._bart_pipeline = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=self.device
            )
        return self._bart_pipeline
    
    def get_roberta_prediction(self, text: str) -> ModelPrediction:
        """Get sentiment prediction from RoBERTa model."""
        results = self.roberta_pipeline(text)[0]
        
        class_probs = {}
        max_score = 0.0
        
        for item in results:
            label = self.roberta_label_map.get(item["label"].lower(), item["label"].upper())
            class_probs[label] = item["score"]
            max_score = max(max_score, item["score"])
        
        return ModelPrediction(
            model_name="roberta",
            class_probabilities=class_probs,
            confidence=max_score,
            raw_output=results
        )
    
    def get_vader_prediction(self, text: str) -> ModelPrediction:
        """Get sentiment prediction from VADER."""
        scores = self._vader_analyzer.polarity_scores(text)
        compound = scores["compound"]
        
        # Convert VADER scores to class probabilities using softmax-like normalization
        raw_probs = {
            "POSITIVE": scores["pos"],
            "NEGATIVE": scores["neg"],
            "NEUTRAL": scores["neu"]
        }
        
        # Apply softmax for proper probability distribution
        prob_values = np.array(list(raw_probs.values()))
        prob_values = softmax(prob_values * 2)  # Scale factor for sharper distributions
        
        class_probs = dict(zip(raw_probs.keys(), prob_values))
        confidence = max(class_probs.values())
        
        return ModelPrediction(
            model_name="vader",
            class_probabilities=class_probs,
            confidence=confidence,
            raw_output=scores
        )
    
    def get_bart_mnli_prediction(self, text: str) -> ModelPrediction:
        """Get sentiment prediction from BART-MNLI zero-shot classification."""
        hypothesis_template = "This text expresses a {} sentiment"
        labels = ["positive", "negative", "neutral"]
        
        result = self.bart_pipeline(
            text,
            candidate_labels=labels,
            hypothesis_template=hypothesis_template
        )
        
        class_probs = {}
        for label, score in zip(result["labels"], result["scores"]):
            class_probs[label.upper()] = score
        
        confidence = max(result["scores"])
        
        return ModelPrediction(
            model_name="bart_mnli",
            class_probabilities=class_probs,
            confidence=confidence,
            raw_output=result
        )
    
    def analyze(
        self,
        text: str,
        models: Optional[List[str]] = None
    ) -> FusedPrediction:
        
        if models is None:
            models = ["roberta", "vader", "bart_mnli"]
        
        predictions = []
        
        if "roberta" in models:
            predictions.append(self.get_roberta_prediction(text))
        
        if "vader" in models:
            predictions.append(self.get_vader_prediction(text))
        
        if "bart_mnli" in models:
            predictions.append(self.get_bart_mnli_prediction(text))
        
        return self.fusion_strategy.fuse_predictions(predictions)


# Example usage and testing
if __name__ == "__main__":
    print("Testing Hybrid Fusion Strategy")
    print("=" * 60)
    
    # Create analyzer with default fusion strategy
    analyzer = MultiModelSentimentAnalyzer(device=-1)
    
    test_texts = [
        "I absolutely love this new feature! It's amazing!",
        "This is terrible, worst experience ever.",
        "The meeting is scheduled for tomorrow at 3 PM.",
        "Great job on the project, but there are some issues to fix."  # Mixed sentiment
    ]
    
    for text in test_texts:
        print(f"\nText: {text}")
        print("-" * 50)
        
        result = analyzer.analyze(text)
        
        print(f"Fused Prediction: {result.predicted_class}")
        print(f"Confidence: {result.confidence:.4f}")
        print(f"Class Probabilities:")
        for cls, prob in result.fused_probabilities.items():
            print(f"  {cls}: {prob:.4f}")
        print(f"Model Contributions:")
        for model, contrib in result.model_contributions.items():
            print(f"  {model}: {contrib:.4f}")
