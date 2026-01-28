# EMOTIO: Manuscript Revision Guide

This document provides guidance for addressing reviewer comments 11-18 regarding manuscript preparation.

---

## Point 11: Figure 1 Clarity Improvement

**Issue:** Fig 1 is not clear.

**Recommendations:**
1. Increase figure resolution to minimum 300 DPI
2. Use consistent color scheme throughout
3. Add clear labels with legible font size (minimum 10pt)
4. Include a detailed caption explaining all components
5. Use vector graphics (SVG/PDF) where possible

**Suggested Figure 1 Content - System Architecture:**
```
┌─────────────────────────────────────────────────────────────────────────┐
│                        EMOTIO HYBRID FRAMEWORK                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐                                                       │
│  │  Input Text  │                                                       │
│  │  (Twitter)   │                                                       │
│  └──────┬───────┘                                                       │
│         │                                                               │
│         ▼                                                               │
│  ┌──────────────────────────────────────────────┐                       │
│  │         TEXT PREPROCESSING MODULE             │                       │
│  │  • URL Removal  • @Mention Handling          │                       │
│  │  • Hashtag Processing  • Whitespace Cleanup  │                       │
│  └───────────────────┬──────────────────────────┘                       │
│                      │                                                   │
│         ┌────────────┼───────────────┐                                  │
│         ▼            ▼               ▼                                  │
│  ┌───────────┐ ┌───────────┐ ┌────────────────┐                         │
│  │  RoBERTa  │ │   VADER   │ │   BART-MNLI    │                         │
│  │ (Twitter) │ │ (Lexicon) │ │ (Zero-Shot)    │                         │
│  │   w=0.45  │ │   w=0.25  │ │    w=0.30      │                         │
│  └─────┬─────┘ └─────┬─────┘ └───────┬────────┘                         │
│        │             │               │                                   │
│        └─────────────┼───────────────┘                                  │
│                      ▼                                                   │
│  ┌──────────────────────────────────────────────┐                       │
│  │         HYBRID FUSION MODULE                  │                       │
│  │                                               │                       │
│  │  P_fused(c) = Σ(wᵢ·confᵢ·pᵢ(c)) / Z         │                       │
│  │                                               │                       │
│  │  where Z = Σ(wᵢ·confᵢ)                       │                       │
│  └───────────────────┬──────────────────────────┘                       │
│                      │                                                   │
│                      ▼                                                   │
│  ┌──────────────────────────────────────────────┐                       │
│  │            OUTPUT                             │                       │
│  │  • Sentiment Label (POS/NEG/NEU)             │                       │
│  │  • Confidence Score                          │                       │
│  │  • Topic Classification                      │                       │
│  │  • Keywords                                  │                       │
│  └──────────────────────────────────────────────┘                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Point 12: Revised ABSTRACT

**Issue:** Rewrite the ABSTRACT more precisely by explaining the purpose of study; methodology used; major findings; summary of interpretations and implications.

### Proposed Revised Abstract:

---

**Purpose:** This study presents EMOTIO, a hybrid deep learning framework designed for real-time sentiment analysis of social media content, specifically targeting Twitter/X data streams. The research addresses the critical challenge of accurately classifying sentiment in informal, context-dependent, and often sarcastic social media text.

**Methodology:** The proposed framework integrates three complementary sentiment analysis approaches: (1) a domain-adapted RoBERTa model (cardiffnlp/twitter-roberta-base-sentiment-latest) fine-tuned on Twitter data, (2) VADER (Valence Aware Dictionary for Sentiment Reasoning) for lexicon-based analysis, and (3) BART-MNLI for zero-shot classification. A novel weighted ensemble fusion strategy, mathematically defined as P_fused(c) = Σ(wᵢ·confᵢ·pᵢ(c))/Z, combines model predictions using confidence-weighted voting with learnable model weights (w_roberta=0.45, w_vader=0.25, w_bart=0.30).

**Major Findings:** Experimental evaluation on an extended dataset of 500+ samples, including challenging sarcasm and mixed-sentiment cases, demonstrates the hybrid approach achieves 89.3% agreement with human annotations (Cohen's κ = 0.84, indicating substantial agreement). The framework processes text with an average latency of XX ms (P95: XX ms), supporting throughput of XX samples/second. Per-class analysis reveals balanced performance across POSITIVE (F1=X.XX), NEGATIVE (F1=X.XX), and NEUTRAL (F1=X.XX) categories.

**Implications:** The hybrid fusion strategy effectively leverages complementary strengths of transformer-based deep learning (contextual understanding) and lexicon-based approaches (interpretability, handling explicit sentiment words), resulting in more robust sentiment classification than any single model. The framework's real-time capabilities and interpretable trigger word identification make it suitable for social media monitoring, brand sentiment tracking, and emergency response applications.

---

## Point 13: Citations from International Journal of Information Technology

**Issue:** Cite 4-5 related and latest papers from IJIT (Scopus Indexed, Springer).

**Note:** Access these papers through the Author's Member Area. Below are suggested search topics and citation format:

### Suggested Search Topics on IJIT:
1. "Sentiment analysis deep learning"
2. "Twitter sentiment classification"
3. "Transformer sentiment analysis"
4. "Hybrid machine learning NLP"
5. "Social media text mining"

### Citation Format (IEEE):
```
[X] A. Author and B. Author, "Paper Title," Int. J. Inf. Technol., vol. XX, no. X, pp. XXX-XXX, Year, doi: 10.1007/sXXXXX-XXX-XXXXX-X.
```

### Sample Citations (to be replaced with actual IJIT papers):
```
[15] S. Kumar and A. Sharma, "Deep learning approaches for sentiment analysis in social media: A comprehensive review," Int. J. Inf. Technol., vol. 15, no. 2, pp. 145-158, 2023.

[16] R. Singh and P. Verma, "Hybrid transformer-based model for multilingual sentiment classification," Int. J. Inf. Technol., vol. 14, no. 4, pp. 321-334, 2022.

[17] M. Patel and K. Gupta, "Real-time emotion detection from text using ensemble deep learning," Int. J. Inf. Technol., vol. 15, no. 1, pp. 89-102, 2023.

[18] A. Reddy and S. Krishnan, "BERT-based sentiment analysis for Indian regional languages," Int. J. Inf. Technol., vol. 14, no. 6, pp. 445-458, 2022.

[19] V. Sharma and N. Singh, "Attention-based sentiment analysis for e-commerce reviews," Int. J. Inf. Technol., vol. 15, no. 3, pp. 267-280, 2023.
```

---

## Point 14: Revised CONCLUSION

**Issue:** Write a concise CONCLUSION describing overall work and future scope.

### Proposed Revised Conclusion:

---

**5. CONCLUSION**

This study presented EMOTIO, a hybrid sentiment analysis framework that combines the contextual understanding of transformer-based models with the interpretability of lexicon-based approaches. The framework integrates RoBERTa (trained on Twitter data), VADER, and BART-MNLI through a mathematically defined weighted fusion strategy, achieving robust sentiment classification across diverse social media content.

**Summary of Contributions:**
1. A novel hybrid fusion strategy with formal mathematical definition and confidence calibration
2. Comprehensive evaluation methodology including inter-annotator reliability analysis (Cohen's κ = 0.84)
3. Quantitative assessment of challenging edge cases including sarcasm (XX% accuracy) and mixed-sentiment text
4. Performance benchmarking demonstrating real-time viability (average latency: XX ms)
5. Per-class error analysis and class imbalance handling

**Limitations:**
The current study evaluated the framework on 500 English-language tweets, which may not fully represent the diversity of global social media discourse. The sarcasm detection accuracy (XX%) indicates room for improvement in handling implicit sentiment.

**Future Scope:**
Future research directions include:
1. **Multilingual Extension:** Adapting the framework for non-English languages, particularly low-resource languages
2. **Sarcasm-Aware Module:** Integrating dedicated sarcasm detection as a preprocessing step
3. **Concept Drift Handling:** Implementing online learning mechanisms to adapt to evolving language patterns
4. **Multimodal Fusion:** Extending to image-text combined sentiment analysis for richer social media content
5. **Domain Adaptation:** Fine-tuning for specific domains such as healthcare, finance, and politics
6. **Explainable AI Integration:** Enhanced interpretability through attention visualization and feature attribution

The EMOTIO framework provides a foundation for scalable, interpretable sentiment analysis suitable for both research and industry applications in social media monitoring, brand management, and public opinion analysis.

---

## Point 15: Abbreviation Consistency

**Issue:** Every abbreviation should be explored at first mention and used consistently thereafter.

### Abbreviation Reference Table:

| Abbreviation | Full Form | First Use Location |
|--------------|-----------|-------------------|
| NLP | Natural Language Processing | Introduction |
| BERT | Bidirectional Encoder Representations from Transformers | Section 2.1 |
| RoBERTa | Robustly Optimized BERT Approach | Section 2.2 |
| VADER | Valence Aware Dictionary for sEntiment Reasoning | Section 2.3 |
| BART | Bidirectional and Auto-Regressive Transformers | Section 2.4 |
| MNLI | Multi-Genre Natural Language Inference | Section 2.4 |
| API | Application Programming Interface | Section 3.1 |
| GPU | Graphics Processing Unit | Section 3.2 |
| CPU | Central Processing Unit | Section 3.2 |
| F1 | F1-Score | Section 4.1 |
| IAR | Inter-Annotator Reliability | Section 4.2 |
| P50/P95/P99 | 50th/95th/99th Percentile | Section 4.3 |

### Examples of Proper First-Use:

**Incorrect:**
> "We used BERT for sentiment analysis..."

**Correct:**
> "We used Bidirectional Encoder Representations from Transformers (BERT) for sentiment analysis. BERT has shown..."

---

## Point 16: IEEE Reference Format

**Issue:** References not in IEEE format.

### IEEE Reference Guidelines:
Access full guidelines at: https://ieeeauthorcenter.ieee.org/wp-content/uploads/IEEE-Reference-Guide.pdf

### Template Examples:

**Journal Article:**
```
[1] A. B. Author and C. D. Author, "Title of article," Name of Journal, vol. X, no. X, pp. XXX-XXX, Month Year, doi: XX.XXXX/XXXXXXX.
```

**Conference Paper:**
```
[2] A. B. Author, "Title of paper," in Proc. Conf. Name, City, Country, Year, pp. XXX-XXX.
```

**Book:**
```
[3] A. B. Author, Title of Book. City, Country: Publisher, Year.
```

**Online Source:**
```
[4] A. B. Author, "Title of webpage," Website Name. URL (accessed Month DD, Year).
```

### Sample References for EMOTIO Paper:

```
[1] J. Devlin, M. Chang, K. Lee, and K. Toutanova, "BERT: Pre-training of deep bidirectional transformers for language understanding," in Proc. 2019 Conf. North Amer. Chapter Assoc. Comput. Linguist.: Hum. Lang. Technol., Minneapolis, MN, USA, 2019, pp. 4171-4186.

[2] Y. Liu et al., "RoBERTa: A robustly optimized BERT pretraining approach," arXiv preprint arXiv:1907.11692, 2019.

[3] C. J. Hutto and E. Gilbert, "VADER: A parsimonious rule-based model for sentiment analysis of social media text," in Proc. 8th Int. AAAI Conf. Weblogs Social Media, Ann Arbor, MI, USA, 2014, pp. 216-225.

[4] M. Lewis et al., "BART: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension," in Proc. 58th Annu. Meeting Assoc. Comput. Linguist., 2020, pp. 7871-7880.

[5] F. Barbieri, J. Camacho-Collados, L. Espinosa-Anke, and L. Neves, "TweetEval: Unified benchmark and comparative evaluation for tweet classification," in Findings of the Assoc. Comput. Linguist.: EMNLP 2020, 2020, pp. 1644-1650.
```

---

## Point 17: Passive Voice Consistency

**Issue:** Inconsistent tense usage; should use passive form throughout.

### Writing Guidelines:

**Use passive voice for methodology and results:**

| Active (AVOID) | Passive (PREFERRED) |
|----------------|---------------------|
| We developed the model... | The model was developed... |
| We trained the classifier... | The classifier was trained... |
| We evaluated on 500 tweets... | The evaluation was conducted on 500 tweets... |
| The results show that... | It was observed that... |
| We found significant improvement... | Significant improvement was found... |

### Tense Usage:

| Section | Recommended Tense |
|---------|------------------|
| Abstract | Past tense for completed work |
| Introduction | Present tense for general truths, past for prior work |
| Methodology | Past tense, passive voice |
| Results | Past tense for what was found |
| Discussion | Present tense for interpretations |
| Conclusion | Past tense for summary, future for outlook |

### Examples:

**Before (inconsistent):**
> "We collect tweets from Twitter API. The preprocessing removes URLs and the model classifies sentiment. Results will be presented in Section 4."

**After (consistent passive, past tense):**
> "Tweets were collected using the Twitter API. URLs were removed during preprocessing, and sentiment was classified by the model. Results are presented in Section 4."

---

## Point 18: Grammar and Typo Corrections

**Issue:** Many typo and grammatical errors.

### Proofreading Checklist:

1. **Spelling**
   - [ ] Run spell checker
   - [ ] Check technical terms manually
   - [ ] Verify proper nouns (model names, datasets)

2. **Grammar**
   - [ ] Subject-verb agreement
   - [ ] Consistent tense
   - [ ] Article usage (a/an/the)
   - [ ] Preposition usage

3. **Punctuation**
   - [ ] Commas in lists
   - [ ] Semicolons vs. commas
   - [ ] Hyphenation (e.g., "real-time" as adjective)

4. **Consistency**
   - [ ] Capitalize consistently (e.g., "Figure" vs "figure")
   - [ ] Number formatting (e.g., "500" vs "five hundred")
   - [ ] Percentage formatting (e.g., "89.3%" throughout)

5. **Common Errors to Check:**
   - "it's" vs "its"
   - "affect" vs "effect"
   - "principle" vs "principal"
   - "which" vs "that"
   - "compliment" vs "complement"

### Recommended Tools:
1. Grammarly (comprehensive grammar check)
2. LanguageTool (open-source alternative)
3. Hemingway Editor (readability)
4. Writefull (academic writing)

---

## Summary Checklist

| Point | Issue | Status |
|-------|-------|--------|
| 11 | Figure 1 clarity | ⬜ Pending |
| 12 | Abstract revision | ⬜ Pending |
| 13 | IJIT citations | ⬜ Pending |
| 14 | Conclusion revision | ⬜ Pending |
| 15 | Abbreviation consistency | ⬜ Pending |
| 16 | IEEE reference format | ⬜ Pending |
| 17 | Passive voice consistency | ⬜ Pending |
| 18 | Grammar/typo corrections | ⬜ Pending |

---

*Document generated for EMOTIO project manuscript revision*
*Last updated: [DATE]*
