# E-Commerce Price Comparison Tool with RAG & Neural Network Sentiment Analysis

## Project Documentation

**Author:** Anurag U  
**Repository:** AnuragU03/Major  
**Date:** December 2025

---

## Table of Contents

1. [Scope / Objectives of the Project](#1-scope--objectives-of-the-project)
2. [Methodology](#2-methodology)
3. [Technical Details](#3-technical-details)

---

## 1. Scope / Objectives of the Project

### 1.1 Primary Objective

To develop an **intelligent e-commerce price comparison system** that aggregates product information from multiple Indian e-commerce platforms (**Amazon.in, Flipkart, Croma, and Reliance Digital**) and provides users with comprehensive product analysis including pricing, specifications, reviews, and **Neural Network-powered sentiment analysis using DistilBERT**.

### 1.2 Specific Objectives

| # | Objective | Description |
|---|-----------|-------------|
| 1 | **Multi-Platform Product Aggregation** | Scrape and collect product data from **Amazon.in, Flipkart, Croma, and Reliance Digital** simultaneously for unified comparison |
| 2 | **RAG-Based Smart Caching** | Implement Retrieval-Augmented Generation (RAG) pipeline with semantic search to cache products locally and reduce redundant scraping |
| 3 | **Deep Product Information Extraction** | Extract comprehensive product details including technical specifications, features, customer reviews (with pagination up to 50 reviews), rating breakdowns, and descriptions |
| 4 | **Neural Network Sentiment Analysis** | Deploy DistilBERT transformer model fine-tuned on SST-2 with aspect-based sentiment analysis (quality, performance, battery, camera, display, value) |
| 5 | **Interactive Visualization** | Provide a rich GUI interface displaying products with images, prices, specs, sentiment scores, and direct purchase links |
| 6 | **Intelligent Product Filtering** | Automatically filter out accessories, irrelevant products, and validate product relevance to search queries |
| 7 | **Anti-Detection Measures** | Implement StealthBrowser with randomized user agents, human-like scrolling, and rate limiting to avoid bot detection |

### 1.3 Scope Boundaries

**In Scope:**
- Amazon.in, Flipkart, Croma, and Reliance Digital price comparison
- Product specifications, reviews, and ratings extraction
- Advanced review scraping with pagination (up to 50 reviews per product)
- Neural Network sentiment analysis using DistilBERT (Transformer-based)
- Aspect-based sentiment analysis (quality, performance, battery, camera, display, value)
- Support for multiple training datasets (Amazon Polarity, Amazon Reviews 2023, Yelp Reviews)
- Local RAG storage with semantic search
- Anti-detection measures (StealthBrowser, RateLimiter, human-like scrolling)
- Desktop GUI application (Tkinter)

**Out of Scope:**
- Mobile application
- Real-time price alerts/notifications
- User authentication or personalized pricing
- Other e-commerce platforms (Myntra, Snapdeal, etc.)

---

## 2. Methodology

### 2.1 System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                               │
│                    (Tkinter GUI Application)                         │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        RAG PIPELINE                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                  │
│  │ Local Exact │──▶│ Fuzzy Match │──▶│ Web Scrape  │                  │
│  │   Search    │  │   (60%)     │  │  (Selenium) │                  │
│  └─────────────┘  └─────────────┘  └─────────────┘                  │
└─────────────────────────────────────────────────────────────────────┘
                                   │
     ┌─────────────────────────────────────────────────────────┐
     │                  MULTI-AGENT SCRAPERS                   │
     │  ┌───────────┐ ┌───────────┐ ┌───────┐ ┌───────────┐  │
     │  │ Amazon.in │ │ Flipkart  │ │ Croma │ │ Reliance  │  │
     │  │   Agent   │ │   Agent   │ │ Agent │ │   Agent   │  │
     │  └───────────┘ └───────────┘ └───────┘ └───────────┘  │
     └─────────────────────────────────────────────────────────┘
                                   │
              ┌────────────────────┴────────────────────┐
              ▼                                         ▼
┌─────────────────────────┐              ┌─────────────────────────────┐
│   ProductRAGStorage     │              │   Enhanced Sentiment Analyzer   │
│   - TF-IDF Vectorizer   │              │   - DistilBERT Transformer      │
│   - Cosine Similarity   │              │   - Aspect-Based Analysis       │
│   - Pickle Persistence  │              │   - GPU/CPU Auto-detection      │
└─────────────────────────┘              └─────────────────────────────┘
              │                                         │
              ▼                                         ▼
┌─────────────────────────┐              ┌─────────────────────────────┐
│  product_rag_db.pkl     │              │  Pre-trained Model (cached)     │
└─────────────────────────┘              └─────────────────────────────┘
```

### 2.2 Search Workflow (RAG Pipeline)

**Step-by-Step Process:**

```
User Query: "Samsung Galaxy Watch"
        │
        ▼
┌───────────────────────────────────┐
│ STEP 1: Local Exact Match         │  ◄── Fast (< 1 second)
│ Check cached products database    │
└───────────────────────────────────┘
        │ Not Found
        ▼
┌───────────────────────────────────┐
│ STEP 2: Fuzzy Matching (60%)      │  ◄── Semantic Search
│ TF-IDF + Cosine Similarity        │
│ Token overlap threshold: 60%      │
└───────────────────────────────────┘
        │ Not Found
        ▼
┌───────────────────────────────────┐
│ STEP 3: External Web Scraping     │  ◄── Parallel (4 threads)
│ Amazon.in + Flipkart + Croma +    │
│ Reliance Digital (StealthBrowser) │
└───────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────┐
│ STEP 4: Validation & Filtering    │
│ - Remove accessories              │
│ - Validate brand matching         │
│ - Check price validity            │
└───────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────┐
│ STEP 5: Sentiment Analysis        │
│ Enhanced multi-aspect analysis    │
│ Analyze reviews → Score products  │
└───────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────┐
│ STEP 6: Auto-Storage in RAG DB    │
│ Cache for future searches         │
└───────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────┐
│ STEP 7: Display in GUI            │
│ With images, specs, sentiment     │
└───────────────────────────────────┘
```

### 2.3 Sentiment Analysis Methodology

The system uses a **Neural Network-based sentiment analyzer** powered by DistilBERT, a transformer model pre-trained on SST-2 (Stanford Sentiment Treebank).

| Step | Process | Technical Details |
|------|---------|-------------------|
| **1. Model Loading** | Load pre-trained DistilBERT | `distilbert-base-uncased-finetuned-sst-2-english` from HuggingFace |
| **2. Text Collection** | Gather product text | Priority: Customer Reviews > Review Summary > Description |
| **3. Preprocessing** | Clean input text | Remove URLs, special chars, truncate to 512 tokens |
| **4. Inference** | Run through transformer | HuggingFace sentiment-analysis pipeline |
| **5. Score Mapping** | Convert to 3-class | POSITIVE/NEGATIVE with confidence threshold (0.6) |
| **6. Aggregation** | Combine multiple reviews | Average scores across all analyzed texts |
| **7. Explanation** | Generate summary | "Based on X reviews: Y positive, Z negative" |

#### Supported Training Datasets (for fine-tuning)

| Dataset | Source | Size | Labels |
|---------|--------|------|--------|
| **Amazon Polarity** | `mteb/amazon_polarity` | 400K+ | Binary (Positive/Negative) |
| **Amazon Reviews 2023** | `McAuley-Lab/Amazon-Reviews-2023` | Millions | 1-5 Stars by Category |
| **Yelp Reviews** | `Yelp/yelp_review_full` | 650K+ | 1-5 Stars |

#### Model Architecture

```
DistilBERT (66M parameters)
├── 6 Transformer Layers
├── 768 Hidden Dimensions
├── 12 Attention Heads
├── Max Sequence Length: 512 tokens
└── Fine-tuned on SST-2 for Binary Sentiment
```

### 2.4 Web Scraping Methodology

**Anti-Bot Measures Implemented:**
- Random User-Agent rotation (multiple Chrome versions)
- WebDriver flags disabled (`navigator.webdriver = undefined`)
- Chrome automation extension disabled (`excludeSwitches: enable-automation`)
- Blink automation features disabled (`disable-blink-features=AutomationControlled`)
- Random delays between requests (3-10 seconds)
- Human-like scrolling with randomized distances (200-800 pixels)
- Window size randomization (1920x1080, 1366x768, 1440x900)
- RateLimiter class to prevent IP blocks (10 requests/minute)
- Session persistence for cookie management

**Data Extraction Strategy:**
- Multiple CSS selector fallbacks for each field
- Tab management for product detail pages
- Retry logic (2 attempts with 30-second timeouts)
- Pagination support for deep review extraction (up to 50 reviews)

**Supported E-Commerce Platforms:**

| Platform | URL | Features |
|----------|-----|----------|
| Amazon.in | amazon.in | Full specs, reviews, rating breakdown |
| Flipkart | flipkart.com | Specs, reviews, highlights |
| Croma | croma.com | Product search, pricing |
| Reliance Digital | reliancedigital.in | Product search, pricing |

---

## 3. Technical Details

### 3.1 Technology Stack

| Component | Technology | Version |
|-----------|------------|---------|
| **Programming Language** | Python | 3.8+ |
| **Web Scraping** | Selenium WebDriver | 4.38.0 |
| **Browser Automation** | Chrome + ChromeDriver | Auto-managed |
| **Neural Networks** | PyTorch + Transformers | 2.0+, 4.35+ |
| **Pre-trained Model** | DistilBERT (HuggingFace) | SST-2 fine-tuned |
| **Dataset Loading** | HuggingFace Datasets | 2.14+ |
| **Data Processing** | Pandas, NumPy | 2.3.3, 2.3.4 |
| **Machine Learning** | scikit-learn | 1.7.2 |
| **GUI Framework** | Tkinter | Built-in |
| **Image Processing** | Pillow (PIL) | 12.0.0 |
| **HTTP Requests** | Requests | 2.32.5 |

### 3.2 Key Classes & Modules

| Class/Module | File | Purpose |
|--------------|------|---------|
| `NeuralSentimentAnalyzer` | `neural_sentiment_analyzer.py` | DistilBERT-based sentiment analysis |
| `EnhancedSentimentAnalyzer` | `multi_agent_scraper.py` | Multi-model analyzer with aspect-based analysis |
| `DatasetLoader` | `neural_sentiment_analyzer.py` | Load Amazon/Yelp datasets from HuggingFace |
| `ProductRAGStorage` | `Try.py` | RAG-based product caching with TF-IDF semantic search |
| `AdvancedReviewScraper` | `multi_agent_scraper.py` | Deep review extraction with pagination (50+ reviews) |
| `CromaScraper` | `multi_agent_scraper.py` | Croma.com product scraper |
| `RelianceDigitalScraper` | `multi_agent_scraper.py` | RelianceDigital.in product scraper |
| `StealthBrowser` | `multi_agent_scraper.py` | Anti-detection Chrome automation |
| `RateLimiter` | `multi_agent_scraper.py` | Request rate limiting to avoid IP blocks |
| `AmazonAgent` | `multi_agent_scraper.py` | Dedicated agent for Amazon.in scraping |
| `FlipkartAgent` | `multi_agent_scraper.py` | Dedicated agent for Flipkart scraping |
| `CromaAgent` | `multi_agent_scraper.py` | Dedicated agent for Croma.com scraping |
| `RelianceAgent` | `multi_agent_scraper.py` | Dedicated agent for RelianceDigital.in scraping |
| `SentimentAgent` | `multi_agent_scraper.py` | Agent wrapper for neural sentiment analysis |
| `FilterAgent` | `multi_agent_scraper.py` | Product filtering and validation |
| `GUIAgent` | `multi_agent_scraper.py` | Results display agent |

### 3.3 Neural Network Model Specifications

**DistilBERT Sentiment Classifier:**

```python
# Model from HuggingFace
model_name = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

# Pipeline Configuration
pipeline(
    "sentiment-analysis",
    model=model_name,
    device=0 if cuda_available else -1,  # GPU/CPU auto-detection
    truncation=True,
    max_length=512
)

# Output Format
{
    'label': 'POSITIVE' or 'NEGATIVE',
    'score': 0.0 to 1.0  # Confidence
}
```

**Sentiment Score Mapping:**

```python
# Convert binary to 3-class sentiment
if label == 'POSITIVE' and score > 0.6:
    sentiment = 'positive'
elif label == 'NEGATIVE' and score > 0.6:
    sentiment = 'negative'
else:
    sentiment = 'neutral'

# Score normalization (0 = negative, 0.5 = neutral, 1 = positive)
final_score = 0.5 + (score * 0.5) if positive else 0.5 - (score * 0.5)
```

**Semantic Search (RAG) Configuration:**

```python
# Vectorization
TfidfVectorizer(max_features=500)

# Similarity Computation
cosine_similarity(query_vector, document_vectors)
```

### 3.4 Data Extracted Per Product

| Field | Source | Description |
|-------|--------|-------------|
| `name` | Search page | Product title |
| `price` | Search/Product page | Price in INR |
| `rating` | Both pages | Star rating (e.g., "4.2 out of 5") |
| `reviews` | Both pages | Review count |
| `image_url` | Search page | Product thumbnail |
| `product_link` | Search page | Direct URL to product |
| `technical_details` | Product page | Dict of specifications |
| `features` | Product page | List of feature bullets |
| `description` | Product page | Full product description |
| `customer_reviews` | Product page | List of top 5 reviews |
| `rating_breakdown` | Product page | Percentage per star level |
| `review_summary` | Product page | "Customers say" summary |
| `sentiment` | ML Model | positive/neutral/negative |
| `sentiment_score` | ML Model | 0.0 to 1.0 |
| `sentiment_explanation` | ML Model | Key words driving sentiment |

### 3.5 Project File Structure

```
Major Project/
├── Try.py                        # Main application entry point (RAG + GUI)
├── multi_agent_scraper.py        # Multi-agent web scraping system
├── neural_sentiment_analyzer.py  # Neural network sentiment analysis (DistilBERT)
├── requirements.txt              # Python dependencies
├── README.md                     # Quick start guide
├── PROJECT_DOCUMENTATION.md      # This documentation
├── product_rag_db.pkl            # RAG storage (auto-generated)
└── __pycache__/                  # Python cache
```

### 3.6 Dependencies (requirements.txt)

```
# Core Dependencies
selenium==4.38.0
pandas==2.3.3
numpy==2.3.4
scikit-learn==1.7.2
Pillow==12.0.0
requests==2.32.5
webdriver-manager==4.0.2

# Neural Network Sentiment Analysis (DistilBERT)
transformers>=4.35.0
torch>=2.0.0
datasets>=2.14.0
```

### 3.6 Dependencies (requirements.txt)

```
# Core Dependencies
selenium==4.38.0
pandas==2.3.3
numpy==2.3.4
scikit-learn==1.7.2
Pillow==12.0.0
requests==2.32.5
webdriver-manager==4.0.2

# Neural Network Sentiment Analysis (DistilBERT)
transformers>=4.35.0
torch>=2.0.0
datasets>=2.14.0
```

### 3.7 Performance Characteristics

| Metric | Value |
|--------|-------|
| First search (cold cache) | 30-60 seconds |
| Cached search (warm cache) | < 1 second |
| Products per search | 2-20 (configurable) |
| Storage per 100 products | ~1 MB |
| Neural model first load | 5-15 seconds (downloads ~260MB model) |
| Sentiment analysis per product | < 1 second |
| Model accuracy (SST-2) | ~91% |
| GPU acceleration | Automatic if CUDA available |

### 3.8 Algorithms Used

1. **Transformer Architecture (DistilBERT)**
   - 6-layer distilled version of BERT
   - Self-attention mechanism for context understanding
   - Pre-trained on large text corpus, fine-tuned on SST-2

2. **TF-IDF (Term Frequency-Inverse Document Frequency)**
   - Used for text vectorization in RAG storage
   - Converts text to numerical feature vectors for similarity search

3. **Cosine Similarity**
   - Measures similarity between product descriptions
   - Used for semantic product matching in RAG pipeline

4. **Sentiment Aggregation**
   - Analyzes multiple reviews per product
   - Averages confidence scores for final sentiment

### 3.9 HuggingFace Datasets Integration

The system supports loading datasets for potential fine-tuning:

```python
from neural_sentiment_analyzer import DatasetLoader

# Load Amazon Polarity (binary sentiment)
amazon_data = DatasetLoader.load_amazon_polarity(sample_size=5000)

# Load Amazon Reviews 2023 (by category)
electronics = DatasetLoader.load_amazon_reviews_2023(category="Electronics")

# Load Yelp Reviews (5-star ratings)
yelp_data = DatasetLoader.load_yelp_reviews(sample_size=5000)

# Prepare combined dataset for fine-tuning
combined = DatasetLoader.prepare_combined_dataset(
    amazon_samples=5000,
    yelp_samples=5000
)
```

### 3.10 Error Handling Mechanisms

| Mechanism | Description |
|-----------|-------------|
| **Timeout Protection** | 30-second page load limits |
| **Element Not Found** | Multiple CSS selector fallbacks |
| **Tab Management** | Auto-cleanup of browser tabs on errors |
| **Data Validation** | Skips invalid products without crashing |
| **Model Fallback** | Returns 'unknown' sentiment if model not loaded |
| **GPU Fallback** | Automatic CPU fallback if CUDA unavailable |
| **Import Protection** | Graceful degradation if transformers not installed |

### 3.11 System Requirements

- **Operating System:** Windows/Linux/macOS
- **Python:** 3.8 or higher
- **Browser:** Google Chrome (latest version)
- **RAM:** 8GB minimum (16GB recommended for GPU)
- **Storage:** 1GB for application + model cache
- **GPU:** Optional (CUDA-compatible for faster inference)
- **Internet:** Required for web scraping and first model download

### 3.12 Installation & Usage

**Installation:**
```bash
# Clone repository
git clone https://github.com/AnuragU03/Major.git
cd "Major Project"

# Install dependencies
pip install -r requirements.txt

# Test neural sentiment analyzer (optional)
python neural_sentiment_analyzer.py

# Run the application
python Try.py
```

**Alternative: Multi-Agent Scraper**
```bash
# Run the multi-agent version
python multi_agent_scraper.py
```

**Usage:**
1. Launch the application with `python Try.py`
2. Enter product name (e.g., "Samsung Galaxy Watch")
3. Specify number of products per source
4. View results in interactive GUI with neural sentiment analysis

---

## Summary

This project implements a comprehensive e-commerce price comparison system that combines:

- **Web Scraping** (Selenium) for multi-platform data extraction (Amazon, Flipkart, Croma, Reliance Digital)
- **RAG Pipeline** (TF-IDF + Cosine Similarity) for smart caching
- **Neural Network Sentiment Analysis** (DistilBERT Transformer) for accurate review analysis
- **Aspect-Based Sentiment** for analyzing quality, performance, battery, camera, display, and value
- **Advanced Anti-Detection** (StealthBrowser, RateLimiter, human-like scrolling)
- **Multiple Dataset Support** (Amazon Polarity, Amazon Reviews 2023, Yelp Reviews)
- **GUI** (Tkinter) for interactive user experience

The system provides users with an intelligent way to compare products across **4 major Indian e-commerce platforms** while gaining insights into customer sentiment through state-of-the-art transformer-based NLP models.

### Key Features

| Feature | Technology | Benefit |
|---------|------------|---------|
| **Multi-Platform Scraping** | Selenium + 4 Parallel Threads | Amazon + Flipkart + Croma + Reliance Digital |
| **Smart Caching** | TF-IDF RAG | Instant results for repeated searches |
| **Neural Sentiment** | DistilBERT (91% accuracy) | Accurate sentiment from reviews |
| **Aspect-Based Analysis** | EnhancedSentimentAnalyzer | Quality, performance, battery insights |
| **Anti-Detection** | StealthBrowser + RateLimiter | Avoid bot detection and IP blocks |
| **Deep Review Scraping** | AdvancedReviewScraper | Up to 50 reviews with pagination |
| **Dataset Flexibility** | HuggingFace Integration | Fine-tune on custom datasets |
| **GPU Acceleration** | PyTorch CUDA | Fast inference on GPU |

### New Classes Added (December 2025)

| Class | Purpose |
|-------|---------|
| `AdvancedReviewScraper` | Deep review extraction with pagination support |
| `CromaScraper` | Scraper for Croma.com products |
| `RelianceDigitalScraper` | Scraper for RelianceDigital.in products |
| `StealthBrowser` | Anti-detection browser with stealth configuration |
| `RateLimiter` | Request throttling to avoid IP blocks |
| `EnhancedSentimentAnalyzer` | Multi-model sentiment with aspect-based analysis |
| `CromaAgent` | Multi-agent wrapper for Croma scraping |
| `RelianceAgent` | Multi-agent wrapper for Reliance Digital scraping |
| `UnifiedProductScraper` | Master coordinator for all platforms |

---

*Document updated for Major Project - December 2025*
*Neural Network Sentiment Analysis using DistilBERT*
*4-Platform Support: Amazon.in, Flipkart, Croma, Reliance Digital*
