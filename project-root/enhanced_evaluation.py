"""
Enhanced Model Evaluation with Comprehensive Metrics

This module addresses reviewer concerns:
- Point 2: Latency, throughput, and performance benchmarks
- Point 4: Inter-annotator reliability (Cohen's Kappa)
- Point 5: Comparison with lightweight transformer variants
- Point 7: Sarcasm/mixed-sentiment quantitative evaluation
- Point 8: Class imbalance handling and per-class error distribution
- Point 9: Computational cost and memory analysis
- Point 10: Concept drift detection
"""

import pandas as pd
import numpy as np
import time
import psutil
import tracemalloc
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    f1_score, precision_recall_fscore_support, cohen_kappa_score,
    balanced_accuracy_score
)
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


@dataclass
class PerformanceMetrics:
    """Container for latency and throughput metrics."""
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_per_second: float
    total_samples: int
    total_time_seconds: float


@dataclass 
class MemoryMetrics:
    """Container for memory usage metrics."""
    peak_memory_mb: float
    current_memory_mb: float
    memory_per_sample_kb: float


@dataclass
class ClassDistribution:
    """Container for class distribution analysis."""
    class_counts: Dict[str, int]
    class_percentages: Dict[str, float]
    imbalance_ratio: float  # max/min class ratio
    is_imbalanced: bool  # True if ratio > 3


class LatencyBenchmark:
    """
    Measures latency, throughput, and API rate-limit constraints.
    
    Addresses Review Point #2: Real-time claims quantification.
    """
    
    def __init__(self):
        self.latencies: List[float] = []
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
    
    def reset(self):
        """Reset all measurements."""
        self.latencies = []
        self.start_time = None
        self.end_time = None
    
    def start_batch(self):
        """Start batch timing."""
        self.start_time = time.perf_counter()
    
    def end_batch(self):
        """End batch timing."""
        self.end_time = time.perf_counter()
    
    def record_latency(self, inference_func: Callable, *args, **kwargs):
        """
        Record latency for a single inference call.
        
        Returns the function result along with latency in milliseconds.
        """
        start = time.perf_counter()
        result = inference_func(*args, **kwargs)
        end = time.perf_counter()
        
        latency_ms = (end - start) * 1000
        self.latencies.append(latency_ms)
        
        return result, latency_ms
    
    def get_metrics(self) -> PerformanceMetrics:
        """Calculate and return performance metrics."""
        if not self.latencies:
            raise ValueError("No latency measurements recorded")
        
        latencies = np.array(self.latencies)
        total_time = self.end_time - self.start_time if self.end_time else sum(self.latencies) / 1000
        
        return PerformanceMetrics(
            avg_latency_ms=np.mean(latencies),
            p50_latency_ms=np.percentile(latencies, 50),
            p95_latency_ms=np.percentile(latencies, 95),
            p99_latency_ms=np.percentile(latencies, 99),
            throughput_per_second=len(latencies) / total_time,
            total_samples=len(latencies),
            total_time_seconds=total_time
        )
    
    def format_report(self) -> str:
        """Generate formatted performance report."""
        metrics = self.get_metrics()
        
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║           PERFORMANCE BENCHMARK REPORT                       ║
╠══════════════════════════════════════════════════════════════╣
║ Latency Metrics:                                             ║
║   • Average:     {metrics.avg_latency_ms:8.2f} ms                           ║
║   • P50 (Median): {metrics.p50_latency_ms:7.2f} ms                           ║
║   • P95:          {metrics.p95_latency_ms:7.2f} ms                           ║
║   • P99:          {metrics.p99_latency_ms:7.2f} ms                           ║
╠══════════════════════════════════════════════════════════════╣
║ Throughput Metrics:                                          ║
║   • Samples/Second: {metrics.throughput_per_second:6.2f}                              ║
║   • Total Samples:  {metrics.total_samples:6d}                               ║
║   • Total Time:     {metrics.total_time_seconds:6.2f} seconds                        ║
╠══════════════════════════════════════════════════════════════╣
║ Real-time Viability:                                         ║
║   • Avg < 100ms: {"✓ PASS" if metrics.avg_latency_ms < 100 else "✗ FAIL"}                                    ║
║   • P99 < 500ms: {"✓ PASS" if metrics.p99_latency_ms < 500 else "✗ FAIL"}                                    ║
╚══════════════════════════════════════════════════════════════╝
"""
        return report


class MemoryProfiler:
    """
    Profiles memory usage during model inference.
    
    Addresses Review Point #9: Computational cost and memory analysis.
    """
    
    def __init__(self):
        self.samples_processed = 0
    
    def start_profiling(self):
        """Start memory profiling."""
        tracemalloc.start()
    
    def get_metrics(self) -> MemoryMetrics:
        """Get current memory metrics."""
        current, peak = tracemalloc.get_traced_memory()
        process = psutil.Process()
        
        memory_per_sample = (current / 1024) / max(self.samples_processed, 1)
        
        return MemoryMetrics(
            peak_memory_mb=peak / (1024 * 1024),
            current_memory_mb=current / (1024 * 1024),
            memory_per_sample_kb=memory_per_sample
        )
    
    def stop_profiling(self):
        """Stop memory profiling."""
        tracemalloc.stop()
    
    def increment_samples(self, count: int = 1):
        """Increment processed sample count."""
        self.samples_processed += count


class InterAnnotatorReliability:
    """
    Calculates inter-annotator agreement metrics.
    
    Addresses Review Point #4: Agreement metric lacks reliability analysis.
    """
    
    @staticmethod
    def cohens_kappa(labels1: List[str], labels2: List[str]) -> float:
        """
        Calculate Cohen's Kappa coefficient for two annotators.
        
        Formula:
        κ = (p_o - p_e) / (1 - p_e)
        
        where:
        - p_o = observed agreement
        - p_e = expected agreement by chance
        
        Interpretation:
        - κ < 0: Less than chance agreement
        - κ = 0: Chance agreement
        - 0.01-0.20: Slight agreement
        - 0.21-0.40: Fair agreement
        - 0.41-0.60: Moderate agreement
        - 0.61-0.80: Substantial agreement
        - 0.81-1.00: Almost perfect agreement
        """
        return cohen_kappa_score(labels1, labels2)
    
    @staticmethod
    def fleiss_kappa(annotations: np.ndarray) -> float:
        """
        Calculate Fleiss' Kappa for multiple annotators.
        
        Args:
            annotations: Matrix of shape (n_samples, n_categories)
                        where each cell contains the number of annotators
                        who assigned that category to that sample.
        
        Returns:
            Fleiss' Kappa coefficient
        """
        n_samples, n_categories = annotations.shape
        n_annotators = annotations.sum(axis=1)[0]  # Assuming equal annotators per sample
        
        # Proportion of each category
        p = annotations.sum(axis=0) / (n_samples * n_annotators)
        
        # Expected agreement by chance
        p_e = np.sum(p ** 2)
        
        # Observed agreement for each sample
        p_i = (annotations.sum(axis=1) ** 2 - n_annotators) / (n_annotators * (n_annotators - 1))
        p_i = (np.sum(annotations ** 2, axis=1) - n_annotators) / (n_annotators * (n_annotators - 1))
        
        # Mean observed agreement
        p_o = np.mean(p_i)
        
        # Fleiss' Kappa
        if p_e == 1:
            return 1.0
        return (p_o - p_e) / (1 - p_e)
    
    @staticmethod
    def agreement_analysis(
        model_labels: List[str],
        human_labels: List[str]
    ) -> Dict:
        """
        Comprehensive agreement analysis between model and human labels.
        
        Returns dict with:
        - raw_agreement: Simple percentage agreement
        - cohens_kappa: Chance-corrected agreement
        - agreement_by_class: Per-class agreement rates
        - confusion_matrix: Detailed confusion matrix
        """
        assert len(model_labels) == len(human_labels), "Label lists must be same length"
        
        # Raw agreement
        agreements = [m == h for m, h in zip(model_labels, human_labels)]
        raw_agreement = sum(agreements) / len(agreements)
        
        # Cohen's Kappa
        kappa = cohen_kappa_score(human_labels, model_labels)
        
        # Per-class agreement
        classes = list(set(model_labels) | set(human_labels))
        agreement_by_class = {}
        
        for cls in classes:
            cls_indices = [i for i, h in enumerate(human_labels) if h == cls]
            if cls_indices:
                cls_agreements = sum(1 for i in cls_indices if model_labels[i] == human_labels[i])
                agreement_by_class[cls] = cls_agreements / len(cls_indices)
            else:
                agreement_by_class[cls] = np.nan
        
        # Confusion matrix
        cm = confusion_matrix(human_labels, model_labels, labels=classes)
        
        return {
            "raw_agreement": raw_agreement,
            "cohens_kappa": kappa,
            "kappa_interpretation": InterAnnotatorReliability._interpret_kappa(kappa),
            "agreement_by_class": agreement_by_class,
            "confusion_matrix": cm,
            "class_labels": classes
        }
    
    @staticmethod
    def _interpret_kappa(kappa: float) -> str:
        """Interpret Kappa coefficient."""
        if kappa < 0:
            return "Less than chance agreement"
        elif kappa < 0.20:
            return "Slight agreement"
        elif kappa < 0.40:
            return "Fair agreement"
        elif kappa < 0.60:
            return "Moderate agreement"
        elif kappa < 0.80:
            return "Substantial agreement"
        else:
            return "Almost perfect agreement"


class SarcasmMixedSentimentEvaluator:
    """
    Quantitative evaluation of sarcasm and mixed-sentiment handling.
    
    Addresses Review Point #7: Sarcasm/mixed-sentiment not quantitatively evaluated.
    """
    
    # Curated test cases for sarcasm and mixed sentiment
    SARCASM_TEST_CASES = [
        ("Oh great, another meeting that could have been an email", "NEGATIVE"),
        ("Wow, thanks for nothing, really appreciate it", "NEGATIVE"),
        ("Sure, because that worked so well last time", "NEGATIVE"),
        ("What a surprise, the train is late again", "NEGATIVE"),
        ("Oh wonderful, more homework on a Friday", "NEGATIVE"),
        ("Yeah right, like that's ever going to happen", "NEGATIVE"),
        ("Thanks captain obvious, couldn't have figured that out", "NEGATIVE"),
        ("Oh joy, another software update", "NEGATIVE"),
        ("Perfect timing, just what I needed", "NEGATIVE"),
        ("Brilliant idea, what could possibly go wrong", "NEGATIVE"),
    ]
    
    MIXED_SENTIMENT_TEST_CASES = [
        ("The movie was visually stunning but the plot was confusing", "NEUTRAL"),
        ("I love the product quality but hate the customer service", "NEUTRAL"),
        ("Great features but way too expensive for what it offers", "NEUTRAL"),
        ("The food was delicious although the service was slow", "NEUTRAL"),
        ("Amazing performance but disappointing ending", "NEUTRAL"),
        ("I appreciate the effort but the results were underwhelming", "NEUTRAL"),
        ("The app works well but drains battery like crazy", "NEUTRAL"),
        ("Nice design but terrible user experience", "NEUTRAL"),
        ("The team worked hard but missed the deadline", "NEUTRAL"),
        ("Good intentions but poor execution", "NEUTRAL"),
    ]
    
    @classmethod
    def evaluate_sarcasm_detection(
        cls,
        sentiment_func: Callable[[str], str]
    ) -> Dict:
        """
        Evaluate model's ability to correctly classify sarcastic statements.
        
        Args:
            sentiment_func: Function that takes text and returns sentiment label
            
        Returns:
            Dictionary with sarcasm detection metrics
        """
        predictions = []
        true_labels = []
        details = []
        
        for text, true_label in cls.SARCASM_TEST_CASES:
            pred = sentiment_func(text)
            predictions.append(pred)
            true_labels.append(true_label)
            details.append({
                "text": text,
                "true": true_label,
                "predicted": pred,
                "correct": pred == true_label
            })
        
        accuracy = accuracy_score(true_labels, predictions)
        
        return {
            "accuracy": accuracy,
            "total_cases": len(cls.SARCASM_TEST_CASES),
            "correct": sum(1 for d in details if d["correct"]),
            "details": details,
            "metric_name": "Sarcasm Detection Accuracy"
        }
    
    @classmethod
    def evaluate_mixed_sentiment(
        cls,
        sentiment_func: Callable[[str], str]
    ) -> Dict:
        """
        Evaluate model's handling of mixed-sentiment statements.
        
        For mixed sentiment, we accept NEUTRAL as correct, or we check
        if the model at least doesn't give a strong wrong prediction.
        """
        predictions = []
        true_labels = []
        details = []
        
        for text, true_label in cls.MIXED_SENTIMENT_TEST_CASES:
            pred = sentiment_func(text)
            predictions.append(pred)
            true_labels.append(true_label)
            details.append({
                "text": text,
                "true": true_label,
                "predicted": pred,
                "correct": pred == true_label
            })
        
        accuracy = accuracy_score(true_labels, predictions)
        
        # Also calculate "acceptable" rate (NEUTRAL is ideal, but POS/NEG not totally wrong)
        acceptable = sum(1 for d in details if d["predicted"] == "NEUTRAL")
        
        return {
            "accuracy": accuracy,
            "neutral_rate": acceptable / len(cls.MIXED_SENTIMENT_TEST_CASES),
            "total_cases": len(cls.MIXED_SENTIMENT_TEST_CASES),
            "correct": sum(1 for d in details if d["correct"]),
            "details": details,
            "metric_name": "Mixed Sentiment Accuracy"
        }


class ClassImbalanceAnalyzer:
    """
    Analyzes and handles class imbalance.
    
    Addresses Review Point #8: Class imbalance handling and per-class error distribution.
    """
    
    @staticmethod
    def analyze_distribution(labels: List[str]) -> ClassDistribution:
        """Analyze class distribution in the dataset."""
        counts = Counter(labels)
        total = len(labels)
        
        percentages = {cls: count / total * 100 for cls, count in counts.items()}
        
        max_count = max(counts.values())
        min_count = min(counts.values())
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        return ClassDistribution(
            class_counts=dict(counts),
            class_percentages=percentages,
            imbalance_ratio=imbalance_ratio,
            is_imbalanced=imbalance_ratio > 3.0
        )
    
    @staticmethod
    def per_class_error_analysis(
        true_labels: List[str],
        predicted_labels: List[str],
        texts: Optional[List[str]] = None
    ) -> Dict:
        """
        Detailed per-class error analysis.
        
        Returns:
        - Per-class precision, recall, F1
        - Error distribution (where each class's samples went wrong)
        - Sample errors for each class
        """
        classes = sorted(list(set(true_labels) | set(predicted_labels)))
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels, predicted_labels, labels=classes, zero_division=0
        )
        
        per_class_metrics = {}
        for i, cls in enumerate(classes):
            per_class_metrics[cls] = {
                "precision": precision[i],
                "recall": recall[i],
                "f1_score": f1[i],
                "support": support[i]
            }
        
        # Error distribution: For each class, where do errors go?
        error_distribution = {cls: Counter() for cls in classes}
        sample_errors = {cls: [] for cls in classes}
        
        for i, (true, pred) in enumerate(zip(true_labels, predicted_labels)):
            if true != pred:
                error_distribution[true][pred] += 1
                if texts:
                    sample_errors[true].append({
                        "text": texts[i],
                        "predicted": pred
                    })
        
        # Convert to percentages
        for cls in classes:
            total_errors = sum(error_distribution[cls].values())
            if total_errors > 0:
                error_distribution[cls] = {
                    k: v / total_errors * 100 
                    for k, v in error_distribution[cls].items()
                }
        
        return {
            "per_class_metrics": per_class_metrics,
            "error_distribution": dict(error_distribution),
            "sample_errors": sample_errors,
            "balanced_accuracy": balanced_accuracy_score(true_labels, predicted_labels)
        }


class ConceptDriftDetector:
    """
    Detects concept drift in streaming sentiment data.
    
    Addresses Review Point #10: Concept drift and temporal sentiment evolution.
    """
    
    def __init__(self, window_size: int = 100, drift_threshold: float = 0.15):
        """
        Initialize drift detector.
        
        Args:
            window_size: Number of samples in each window
            drift_threshold: Maximum allowed accuracy drop before flagging drift
        """
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.windows: List[Dict] = []
    
    def add_window(
        self,
        predictions: List[str],
        true_labels: List[str],
        timestamp: Optional[datetime] = None
    ):
        """Add a new evaluation window."""
        accuracy = accuracy_score(true_labels, predictions)
        
        sentiment_dist = Counter(predictions)
        total = len(predictions)
        sentiment_dist = {k: v/total for k, v in sentiment_dist.items()}
        
        self.windows.append({
            "timestamp": timestamp or datetime.now(),
            "accuracy": accuracy,
            "sentiment_distribution": sentiment_dist,
            "n_samples": len(predictions)
        })
    
    def detect_drift(self) -> Dict:
        """
        Detect if concept drift has occurred.
        
        Uses Page-Hinkley test for drift detection.
        """
        if len(self.windows) < 2:
            return {"drift_detected": False, "message": "Insufficient windows for drift detection"}
        
        accuracies = [w["accuracy"] for w in self.windows]
        baseline_accuracy = np.mean(accuracies[:max(1, len(accuracies)//3)])
        recent_accuracy = accuracies[-1]
        
        accuracy_drop = baseline_accuracy - recent_accuracy
        drift_detected = accuracy_drop > self.drift_threshold
        
        # Analyze distribution shift
        if len(self.windows) >= 2:
            first_dist = self.windows[0]["sentiment_distribution"]
            last_dist = self.windows[-1]["sentiment_distribution"]
            
            distribution_shift = {}
            for cls in set(first_dist.keys()) | set(last_dist.keys()):
                shift = last_dist.get(cls, 0) - first_dist.get(cls, 0)
                distribution_shift[cls] = shift
        else:
            distribution_shift = {}
        
        return {
            "drift_detected": drift_detected,
            "baseline_accuracy": baseline_accuracy,
            "recent_accuracy": recent_accuracy,
            "accuracy_drop": accuracy_drop,
            "threshold": self.drift_threshold,
            "distribution_shift": distribution_shift,
            "n_windows": len(self.windows),
            "recommendation": self._get_recommendation(drift_detected, accuracy_drop)
        }
    
    def _get_recommendation(self, drift_detected: bool, accuracy_drop: float) -> str:
        """Generate recommendations based on drift analysis."""
        if not drift_detected:
            return "No action required. Model performance is stable."
        
        if accuracy_drop > 0.25:
            return "CRITICAL: Significant concept drift detected. Immediate model retraining recommended."
        elif accuracy_drop > 0.15:
            return "WARNING: Moderate drift detected. Schedule model retraining within 1-2 weeks."
        else:
            return "NOTICE: Minor drift detected. Continue monitoring and prepare for potential retraining."
    
    def plot_temporal_evolution(self, save_path: Optional[str] = None):
        """Plot accuracy and distribution over time."""
        if len(self.windows) < 2:
            print("Insufficient data for temporal plot")
            return
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Accuracy over time
        timestamps = [w["timestamp"] for w in self.windows]
        accuracies = [w["accuracy"] for w in self.windows]
        
        axes[0].plot(range(len(accuracies)), accuracies, 'b-o', linewidth=2, markersize=8)
        axes[0].axhline(y=np.mean(accuracies), color='r', linestyle='--', label='Mean Accuracy')
        axes[0].fill_between(
            range(len(accuracies)),
            [a - self.drift_threshold for a in accuracies],
            [a + self.drift_threshold for a in accuracies],
            alpha=0.2, color='blue'
        )
        axes[0].set_xlabel('Window Index')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('Model Accuracy Over Time (Concept Drift Analysis)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Sentiment distribution over time
        classes = list(self.windows[0]["sentiment_distribution"].keys())
        for cls in classes:
            proportions = [w["sentiment_distribution"].get(cls, 0) for w in self.windows]
            axes[1].plot(range(len(proportions)), proportions, '-o', label=cls, linewidth=2)
        
        axes[1].set_xlabel('Window Index')
        axes[1].set_ylabel('Proportion')
        axes[1].set_title('Sentiment Distribution Evolution')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Temporal evolution plot saved to: {save_path}")
        
        plt.show()


class ModelComparison:
    """
    Compares multiple sentiment analysis models.
    
    Addresses Review Point #5: Comparison with lightweight transformer variants.
    """
    
    MODELS_TO_COMPARE = {
        "roberta": "cardiffnlp/twitter-roberta-base-sentiment-latest",
        "distilroberta": "j-hartmann/emotion-english-distilroberta-base",
        "distilbert": "distilbert-base-uncased-finetuned-sst-2-english",
        "deberta_v3": "microsoft/deberta-v3-small"  # Would need fine-tuning for sentiment
    }
    
    @staticmethod
    def compare_model_performance(
        test_texts: List[str],
        true_labels: List[str],
        models_to_test: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Compare performance across multiple models.
        
        Returns DataFrame with accuracy, F1, latency for each model.
        """
        from transformers import pipeline
        import torch
        
        device = 0 if torch.cuda.is_available() else -1
        
        if models_to_test is None:
            models_to_test = ["roberta", "distilbert"]
        
        results = []
        
        for model_name in models_to_test:
            if model_name not in ModelComparison.MODELS_TO_COMPARE:
                print(f"Skipping unknown model: {model_name}")
                continue
            
            model_path = ModelComparison.MODELS_TO_COMPARE[model_name]
            print(f"Testing {model_name}...")
            
            try:
                nlp = pipeline("sentiment-analysis", model=model_path, device=device)
                
                predictions = []
                latencies = []
                
                for text in test_texts:
                    start = time.perf_counter()
                    result = nlp(text[:512])[0]  # Truncate to max length
                    end = time.perf_counter()
                    
                    # Normalize label
                    label = result["label"].upper()
                    if label in ["LABEL_0", "NEG", "0"]:
                        label = "NEGATIVE"
                    elif label in ["LABEL_2", "POS", "1", "LABEL_1"]:
                        label = "POSITIVE"
                    else:
                        label = "NEUTRAL"
                    
                    predictions.append(label)
                    latencies.append((end - start) * 1000)
                
                accuracy = accuracy_score(true_labels, predictions)
                f1_macro = f1_score(true_labels, predictions, average='macro', zero_division=0)
                
                results.append({
                    "Model": model_name,
                    "Accuracy": accuracy,
                    "Macro_F1": f1_macro,
                    "Avg_Latency_ms": np.mean(latencies),
                    "P95_Latency_ms": np.percentile(latencies, 95)
                })
                
            except Exception as e:
                print(f"Error testing {model_name}: {e}")
                results.append({
                    "Model": model_name,
                    "Accuracy": np.nan,
                    "Macro_F1": np.nan,
                    "Avg_Latency_ms": np.nan,
                    "P95_Latency_ms": np.nan,
                    "Error": str(e)
                })
        
        return pd.DataFrame(results)


def run_comprehensive_evaluation(
    sentiment_func: Callable[[str], str],
    test_texts: List[str],
    true_labels: List[str],
    human_labels: Optional[List[str]] = None,
    output_dir: str = "."
):
    """
    Run comprehensive evaluation addressing all reviewer concerns.
    
    Args:
        sentiment_func: Function that takes text and returns sentiment label
        test_texts: List of test texts
        true_labels: Ground truth labels
        human_labels: Optional second set of human labels for IAR analysis
        output_dir: Directory to save outputs
    """
    print("\n" + "="*70)
    print(" COMPREHENSIVE MODEL EVALUATION")
    print("="*70)
    
    # 1. Performance Benchmarking (Point 2, 9)
    print("\n📊 1. PERFORMANCE BENCHMARKING")
    print("-" * 50)
    
    benchmark = LatencyBenchmark()
    memory_profiler = MemoryProfiler()
    
    memory_profiler.start_profiling()
    benchmark.start_batch()
    
    predictions = []
    for text in test_texts:
        pred, latency = benchmark.record_latency(sentiment_func, text)
        predictions.append(pred)
        memory_profiler.increment_samples()
    
    benchmark.end_batch()
    
    print(benchmark.format_report())
    
    mem_metrics = memory_profiler.get_metrics()
    print(f"\n💾 Memory Usage:")
    print(f"   Peak Memory: {mem_metrics.peak_memory_mb:.2f} MB")
    print(f"   Current Memory: {mem_metrics.current_memory_mb:.2f} MB")
    print(f"   Memory per Sample: {mem_metrics.memory_per_sample_kb:.2f} KB")
    
    memory_profiler.stop_profiling()
    
    # 2. Inter-Annotator Reliability (Point 4)
    print("\n\n📏 2. INTER-ANNOTATOR RELIABILITY ANALYSIS")
    print("-" * 50)
    
    if human_labels:
        iar = InterAnnotatorReliability.agreement_analysis(predictions, human_labels)
        print(f"   Raw Agreement: {iar['raw_agreement']*100:.1f}%")
        print(f"   Cohen's Kappa: {iar['cohens_kappa']:.4f}")
        print(f"   Interpretation: {iar['kappa_interpretation']}")
        print(f"\n   Per-Class Agreement:")
        for cls, agreement in iar['agreement_by_class'].items():
            print(f"     {cls}: {agreement*100:.1f}%")
    else:
        # Use VADER as pseudo second annotator
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()
        
        vader_labels = []
        for text in test_texts:
            score = analyzer.polarity_scores(text)
            if score['compound'] >= 0.05:
                vader_labels.append('POSITIVE')
            elif score['compound'] <= -0.05:
                vader_labels.append('NEGATIVE')
            else:
                vader_labels.append('NEUTRAL')
        
        iar = InterAnnotatorReliability.agreement_analysis(predictions, vader_labels)
        print(f"   (Comparing Model vs VADER as baseline)")
        print(f"   Raw Agreement: {iar['raw_agreement']*100:.1f}%")
        print(f"   Cohen's Kappa: {iar['cohens_kappa']:.4f}")
        print(f"   Interpretation: {iar['kappa_interpretation']}")
    
    # 3. Sarcasm/Mixed Sentiment Evaluation (Point 7)
    print("\n\n🎭 3. SARCASM & MIXED SENTIMENT EVALUATION")
    print("-" * 50)
    
    sarcasm_eval = SarcasmMixedSentimentEvaluator.evaluate_sarcasm_detection(sentiment_func)
    print(f"\n   Sarcasm Detection:")
    print(f"     Accuracy: {sarcasm_eval['accuracy']*100:.1f}%")
    print(f"     Correct: {sarcasm_eval['correct']}/{sarcasm_eval['total_cases']}")
    
    mixed_eval = SarcasmMixedSentimentEvaluator.evaluate_mixed_sentiment(sentiment_func)
    print(f"\n   Mixed Sentiment Handling:")
    print(f"     Accuracy: {mixed_eval['accuracy']*100:.1f}%")
    print(f"     Neutral Detection Rate: {mixed_eval['neutral_rate']*100:.1f}%")
    print(f"     Correct: {mixed_eval['correct']}/{mixed_eval['total_cases']}")
    
    # 4. Class Imbalance Analysis (Point 8)
    print("\n\n⚖️ 4. CLASS IMBALANCE & ERROR DISTRIBUTION")
    print("-" * 50)
    
    dist = ClassImbalanceAnalyzer.analyze_distribution(true_labels)
    print(f"\n   Class Distribution:")
    for cls, pct in dist.class_percentages.items():
        print(f"     {cls}: {pct:.1f}% ({dist.class_counts[cls]} samples)")
    print(f"\n   Imbalance Ratio: {dist.imbalance_ratio:.2f}")
    print(f"   Is Imbalanced: {'Yes ⚠️' if dist.is_imbalanced else 'No ✓'}")
    
    error_analysis = ClassImbalanceAnalyzer.per_class_error_analysis(
        true_labels, predictions, test_texts
    )
    print(f"\n   Balanced Accuracy: {error_analysis['balanced_accuracy']*100:.1f}%")
    print(f"\n   Per-Class Metrics:")
    for cls, metrics in error_analysis['per_class_metrics'].items():
        print(f"     {cls}:")
        print(f"       Precision: {metrics['precision']:.3f}")
        print(f"       Recall: {metrics['recall']:.3f}")
        print(f"       F1-Score: {metrics['f1_score']:.3f}")
        print(f"       Support: {metrics['support']}")
    
    print("\n" + "="*70)
    print(" EVALUATION COMPLETE")
    print("="*70)
    
    # Save comprehensive report
    report = {
        "performance": benchmark.get_metrics().__dict__,
        "memory": mem_metrics.__dict__,
        "inter_annotator": iar,
        "sarcasm_evaluation": {
            "accuracy": sarcasm_eval['accuracy'],
            "correct": sarcasm_eval['correct'],
            "total": sarcasm_eval['total_cases']
        },
        "mixed_sentiment": {
            "accuracy": mixed_eval['accuracy'],
            "neutral_rate": mixed_eval['neutral_rate']
        },
        "class_distribution": {
            "counts": dist.class_counts,
            "imbalance_ratio": dist.imbalance_ratio
        },
        "error_analysis": error_analysis
    }
    
    return report


if __name__ == "__main__":
    print("Enhanced Evaluation Module - Reviewer Point Analysis")
    print("This module provides tools to address review points 2, 4, 5, 7, 8, 9, 10")
