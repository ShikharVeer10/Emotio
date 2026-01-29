#!/bin/bash
#
# Script to Create and Push SentiFlow Repository to GitHub
# 
# This script recreates the SentiFlow repository from Emotio's codebase
# and pushes it to a new GitHub repository.
#
# Usage:
#   1. Create a new repository on GitHub named "SentiFlow"
#   2. Run this script: bash CREATE_AND_PUSH_SENTIFLOW.sh
#   3. Enter your GitHub username when prompted
#

set -e  # Exit on error

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║     SentiFlow Repository Creation and Deployment Script       ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Check if we're in the Emotio repository
if [ ! -d "$SCRIPT_DIR/project-root" ]; then
    echo "❌ Error: This script must be run from the Emotio repository root"
    echo "   Expected to find 'project-root' directory"
    exit 1
fi

# Get GitHub username
echo "Please enter your GitHub username:"
read -r GITHUB_USERNAME

if [ -z "$GITHUB_USERNAME" ]; then
    echo "❌ Error: GitHub username cannot be empty"
    exit 1
fi

echo ""
echo "Creating SentiFlow repository for: $GITHUB_USERNAME"
echo ""

# Create temporary directory for the new repository
SENTIFLOW_DIR="$HOME/SentiFlow"

if [ -d "$SENTIFLOW_DIR" ]; then
    echo "⚠️  Warning: $SENTIFLOW_DIR already exists"
    echo "   Do you want to remove it and recreate? (y/n)"
    read -r CONFIRM
    if [ "$CONFIRM" = "y" ] || [ "$CONFIRM" = "Y" ]; then
        rm -rf "$SENTIFLOW_DIR"
        echo "✓ Removed existing directory"
    else
        echo "❌ Aborted"
        exit 1
    fi
fi

echo "📁 Creating new repository at: $SENTIFLOW_DIR"
mkdir -p "$SENTIFLOW_DIR"
cd "$SENTIFLOW_DIR"

# Initialize git repository
echo "🔧 Initializing git repository..."
git init
git config user.name "$GITHUB_USERNAME"
git config user.email "${GITHUB_USERNAME}@users.noreply.github.com"

# Create directory structure
echo "📂 Creating directory structure..."
mkdir -p src

# Copy Python modules
echo "🐍 Copying Python modules..."
cp "$SCRIPT_DIR/project-root"/*.py src/ 2>/dev/null || true

# Create README.md
echo "📝 Creating README.md..."
cat > README.md << 'EOF'
# SentiFlow — Hybrid Sentiment Analysis Framework

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Transformers](https://img.shields.io/badge/Transformers-HuggingFace-orange)
![License](https://img.shields.io/badge/License-MIT-green)

SentiFlow is a **hybrid deep learning framework** for real-time sentiment analysis of social media content, specifically designed for Twitter/X data streams. The framework integrates multiple complementary models through a mathematically defined fusion strategy.

---

## 🎯 Key Features

- **Hybrid Multi-Model Fusion:** Combines RoBERTa, VADER, and BART-MNLI with weighted ensemble
- **Mathematical Fusion Strategy:** Formally defined as `P_fused(c) = Σ(wᵢ·confᵢ·pᵢ(c))/Z`
- **Real-Time Analysis:** Optimized for low-latency inference with performance benchmarking
- **Comprehensive Evaluation:** Inter-annotator reliability, sarcasm detection, concept drift analysis
- **Topic Classification:** Zero-shot classification across 30+ topic categories
- **Keyword Extraction:** BERT-based keyword extraction with KeyBERT
- **GPU Acceleration:** Automatic GPU detection and utilization

---

## 📁 Project Structure

```
SentiFlow/
├── src/                          # Source code modules
│   ├── model_training.py
│   ├── hybrid_fusion.py
│   ├── advanced_sentiment_model.py
│   └── ...
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/SentiFlow.git
cd SentiFlow

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 💻 Usage

### Basic Sentiment Analysis

```python
from src.model_training import get_sentiment, get_topic
from src.keyword_extractor import extract_keywords

text = "I absolutely love this new AI feature! It's amazing!"

# Get sentiment
sentiment = get_sentiment(text)
print(f"Sentiment: {sentiment['label']} (confidence: {sentiment['score']:.3f})")

# Get topic
topic, score = get_topic(text)
print(f"Topic: {topic} (score: {score:.3f})")

# Extract keywords
keywords = extract_keywords(text, sentiment)
print(f"Keywords: {keywords['keywords']}")
```

---

## 📊 Models

- **RoBERTa** (Twitter-RoBERTa-base-sentiment) - 45% weight
- **VADER** Sentiment Analyzer - 25% weight
- **BART-MNLI** Zero-shot Classification - 30% weight
- **KeyBERT** for keyword extraction

---

## 📄 License

MIT License - see LICENSE file for details

---

*SentiFlow - Flowing with Sentiments* 🌊
EOF

# Create requirements.txt
echo "📦 Creating requirements.txt..."
cat > requirements.txt << 'EOF'
# Core ML Libraries
transformers>=4.30.0
torch>=2.0.0
vaderSentiment>=3.3.2
keybert>=0.7.0

# Scientific Computing
numpy>=1.24.0
scipy>=1.10.0
pandas>=2.0.0

# Machine Learning
scikit-learn>=1.3.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0

# Utilities
tqdm>=4.65.0
psutil>=5.9.0

# Twitter API (optional)
tweepy>=4.14.0

# Additional NLP tools
nltk>=3.8.0
sentence-transformers>=2.2.0
EOF

# Create .gitignore
echo "🚫 Creating .gitignore..."
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so

# Virtual environments
venv/
env/
ENV/

# Data files
*.csv
*.json
*.txt
*.log

# Model files
*.pkl
*.model
*.h5
*.pt
*.bin
*.onnx

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db
EOF

# Create LICENSE
echo "📜 Creating LICENSE..."
cat > LICENSE << 'EOF'
MIT License

Copyright (c) 2026 SentiFlow Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
EOF

# Add all files to git
echo "➕ Adding files to git..."
git add -A

# Create initial commit
echo "💾 Creating initial commit..."
git commit -m "Initial commit: SentiFlow - Hybrid Sentiment Analysis Framework

- Core sentiment analysis modules (RoBERTa, VADER, BART-MNLI)
- Hybrid fusion strategy implementation
- Advanced sentiment model with sarcasm detection
- Comprehensive evaluation framework
- Topic classification and keyword extraction
- Complete documentation"

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                   Repository Created! ✅                       ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "📁 Repository location: $SENTIFLOW_DIR"
echo ""
echo "📊 Repository statistics:"
git log --oneline | head -1
echo "   Files: $(find . -type f ! -path './.git/*' | wc -l)"
echo "   Size: $(du -sh . | cut -f1)"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 NEXT STEPS TO PUSH TO GITHUB:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "1️⃣  Create a new repository on GitHub:"
echo "    → Go to: https://github.com/new"
echo "    → Name: SentiFlow"
echo "    → Description: Hybrid Sentiment Analysis Framework"
echo "    → Visibility: Public or Private"
echo "    → ⚠️  DO NOT initialize with README, .gitignore, or license"
echo ""
echo "2️⃣  Push to GitHub (run these commands):"
echo ""
echo "    cd $SENTIFLOW_DIR"
echo "    git remote add origin https://github.com/$GITHUB_USERNAME/SentiFlow.git"
echo "    git branch -M main"
echo "    git push -u origin main"
echo ""
echo "3️⃣  Verify deployment:"
echo "    → Visit: https://github.com/$GITHUB_USERNAME/SentiFlow"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "✨ Your SentiFlow repository is ready to be pushed! ✨"
echo ""
