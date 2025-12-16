# E-Commerce Price Comparison Tool with RAG & Sentiment Analysis

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

To develop an **intelligent e-commerce price comparison system** that aggregates product information from multiple Indian e-commerce platforms (Amazon.in and Flipkart) and provides users with comprehensive product analysis including pricing, specifications, reviews, and **AI-powered sentiment analysis**.

### 1.2 Specific Objectives

| # | Objective | Description |
|---|-----------|-------------|
| 1 | **Multi-Platform Product Aggregation** | Scrape and collect product data from Amazon.in and Flipkart simultaneously for unified comparison |
| 2 | **RAG-Based Smart Caching** | Implement Retrieval-Augmented Generation (RAG) pipeline with semantic search to cache products locally and reduce redundant scraping |
| 3 | **Deep Product Information Extraction** | Extract comprehensive product details including technical specifications, features, customer reviews, rating breakdowns, and descriptions |
| 4 | **Machine Learning Sentiment Analysis** | Train and deploy a sentiment classifier on Amazon reviews dataset to analyze customer sentiment for products |
| 5 | **Interactive Visualization** | Provide a rich GUI interface displaying products with images, prices, specs, sentiment scores, and direct purchase links |
| 6 | **Intelligent Product Filtering** | Automatically filter out accessories, irrelevant products, and validate product relevance to search queries |

### 1.3 Scope Boundaries

**In Scope:**
- Amazon.in and Flipkart price comparison
- Product specifications, reviews, and ratings extraction
- Sentiment analysis using ML (Logistic Regression + TF-IDF)
- Local RAG storage with semantic search
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
              ┌────────────────────┴────────────────────┐
              ▼                                         ▼
┌─────────────────────────┐              ┌─────────────────────────────┐
│   ProductRAGStorage     │              │   Sentiment Analyzer        │
│   - TF-IDF Vectorizer   │              │   - Logistic Regression     │
│   - Cosine Similarity   │              │   - TF-IDF Features         │
│   - Pickle Persistence  │              │   - Negation Handling       │
└─────────────────────────┘              └─────────────────────────────┘
              │                                         │
              ▼                                         ▼
┌─────────────────────────┐              ┌─────────────────────────────┐
│  product_rag_db.pkl     │              │  sentiment_model.pkl        │
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
│ STEP 3: External Web Scraping     │  ◄── Slow (30-60 seconds)
│ Parallel: Amazon.in + Flipkart    │
│ Using Selenium WebDriver          │
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

| Step | Process | Technical Details |
|------|---------|-------------------|
| **1. Data Collection** | Use Amazon Reviews dataset | `amazon_review.csv` with ratings (1-5 stars) |
| **2. Label Mapping** | Convert ratings to sentiment | 1-2★ → Negative, 3★ → Neutral, 4-5★ → Positive |
| **3. Class Balancing** | Undersample majority class | Equal samples per sentiment class |
| **4. Text Preprocessing** | Clean and prepare text | Lowercase, expand contractions, negation handling |
| **5. Negation Scope Marking** | Handle negated words | "not good" → "not good_NEG" |
| **6. Feature Extraction** | TF-IDF Vectorization | max_features=5000, ngram_range=(1,2) |
| **7. Model Training** | Logistic Regression | Multinomial, balanced class weights |
| **8. Evaluation** | Classification metrics | Accuracy, Precision, Recall, F1-Score |

### 2.4 Web Scraping Methodology

**Anti-Bot Measures Implemented:**
- Random User-Agent rotation
- WebDriver flags disabled (`navigator.webdriver = undefined`)
- Random delays between requests (3-10 seconds)
- Window maximization to appear human-like

**Data Extraction Strategy:**
- Multiple CSS selector fallbacks for each field
- Tab management for product detail pages
- Retry logic (2 attempts with 30-second timeouts)

---

## 3. Technical Details

### 3.1 Technology Stack

| Component | Technology | Version |
|-----------|------------|---------|
| **Programming Language** | Python | 3.8+ |
| **Web Scraping** | Selenium WebDriver | 4.38.0 |
| **Browser Automation** | Chrome + ChromeDriver | Auto-managed |
| **Machine Learning** | scikit-learn | 1.7.2 |
| **Data Processing** | Pandas, NumPy | 2.3.3, 2.3.4 |
| **GUI Framework** | Tkinter | Built-in |
| **Image Processing** | Pillow (PIL) | 12.0.0 |
| **HTTP Requests** | Requests | 2.32.5 |

### 3.2 Key Classes & Modules

| Class/Module | File | Purpose |
|--------------|------|---------|
| `ProductRAGStorage` | `Try.py` | RAG-based product caching with TF-IDF semantic search |
| `SentimentAnalyzer` | `sentiment_analyzer.py` | ML-based sentiment classification (Logistic Regression) |
| `scrape_amazon_in()` | `multi_agent_scraper.py` | Amazon.in product scraper |
| `scrape_flipkart()` | `multi_agent_scraper.py` | Flipkart product scraper |
| `scrape_amazon_product_details()` | `multi_agent_scraper.py` | Deep extraction of Amazon product specs |
| `scrape_flipkart_product_details()` | `multi_agent_scraper.py` | Deep extraction of Flipkart product specs |
| `display_results_gui_with_details()` | `Try.py` | Interactive product comparison GUI |

### 3.3 Machine Learning Model Specifications

**Sentiment Classifier Configuration:**

```python
# Vectorization
TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),   # Unigrams and bigrams
    min_df=2,
    max_df=0.95
)

# Classifier
LogisticRegression(
    max_iter=1000,
    multi_class='multinomial',
    solver='lbfgs',
    class_weight='balanced'
)
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
├── Try.py                    # Main application entry point
├── multi_agent_scraper.py    # Web scraping modules (1900 lines)
├── sentiment_analyzer.py     # ML sentiment analysis (640 lines)
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
├── amazon_review.csv         # Training data (Amazon reviews)
├── training.1600000...csv    # Alternative training data (Sentiment140)
├── product_rag_db.pkl        # RAG storage (auto-generated)
├── sentiment_model.pkl       # Trained model (auto-generated)
└── __pycache__/              # Python cache
```

### 3.6 Performance Characteristics

| Metric | Value |
|--------|-------|
| First search (cold cache) | 30-60 seconds |
| Cached search (warm cache) | < 1 second |
| Products per search | 2-20 (configurable) |
| Storage per 100 products | ~1 MB |
| Sentiment model accuracy | ~75-85% (depends on dataset) |

### 3.7 Algorithms Used

1. **TF-IDF (Term Frequency-Inverse Document Frequency)**
   - Used for text vectorization in both RAG and sentiment analysis
   - Converts text to numerical feature vectors

2. **Cosine Similarity**
   - Measures similarity between product descriptions
   - Used for semantic product matching in RAG pipeline

3. **Logistic Regression (Multinomial)**
   - 3-class sentiment classification (Positive, Neutral, Negative)
   - Trained on Amazon product reviews

4. **Negation Scope Detection**
   - Custom NLP preprocessing
   - Handles "not good" → negative sentiment correctly

### 3.8 Error Handling Mechanisms

| Mechanism | Description |
|-----------|-------------|
| **Timeout Protection** | 30-second page load limits |
| **Element Not Found** | Multiple CSS selector fallbacks |
| **Tab Management** | Auto-cleanup of browser tabs on errors |
| **Data Validation** | Skips invalid products without crashing |
| **Model Fallback** | Returns neutral sentiment if model not trained |

### 3.9 System Requirements

- **Operating System:** Windows/Linux/macOS
- **Python:** 3.8 or higher
- **Browser:** Google Chrome (latest version)
- **RAM:** 4GB minimum (8GB recommended)
- **Storage:** 500MB for application + data
- **Internet:** Required for web scraping

### 3.10 Installation & Usage

**Installation:**
```bash
# Clone repository
git clone https://github.com/AnuragU03/Major.git
cd "Major Project"

# Install dependencies
pip install -r requirements.txt

# Run the application
python Try.py
```

**Usage:**
1. Launch the application with `python Try.py`
2. Enter product name (e.g., "Samsung Galaxy Watch")
3. Specify number of products per source
4. View results in interactive GUI with sentiment analysis

---

## Summary

This project implements a comprehensive e-commerce price comparison system that combines:

- **Web Scraping** (Selenium) for data extraction
- **RAG Pipeline** (TF-IDF + Cosine Similarity) for smart caching
- **Machine Learning** (Logistic Regression) for sentiment analysis
- **GUI** (Tkinter) for user interaction

The system provides users with an intelligent way to compare products across platforms while gaining insights into customer sentiment through ML-powered review analysis.

---

*Document generated for Major Project - December 2025*
