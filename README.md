# 🛍️ E-Commerce Price Comparison Tool with RAG & Neural Sentiment Analysis

An intelligent web scraper that compares product prices across Amazon and Flipkart using Retrieval-Augmented Generation (RAG) for smart caching, semantic search, and **Neural Network-based Sentiment Analysis** using DistilBERT transformers.

## 🌟 Features

### Core Functionality
- **Multi-Platform Scraping**: Extracts product data from Amazon.in and Flipkart
- **Deep Product Details**: Scrapes technical specifications, ratings, reviews, and descriptions
- **RAG-Based Caching**: Smart local database with semantic search capabilities
- **Intelligent Filtering**: Automatically filters accessories and validates product relevance
- **Interactive GUI**: Rich interface displaying products with images, prices, sentiment indicators, and detailed specs

### 🧠 Neural Sentiment Analysis
- **Transformer-Based Model**: Uses DistilBERT fine-tuned on SST-2 from HuggingFace
- **High Accuracy**: ~91% accuracy with 66M parameter transformer model
- **Real-Time Analysis**: Analyzes product names and descriptions for sentiment
- **Visual Indicators**: Emoji-based sentiment display (😊 Positive, 😐 Neutral, 😞 Negative)
- **Confidence Scores**: Provides sentiment confidence percentages
- **Batch Processing**: Efficient analysis of multiple products simultaneously

### RAG Pipeline Strategy
1. **Local Exact Search** - Fast retrieval from cached products
2. **Fuzzy Matching** - Flexible search with 60% token matching
3. **External Scraping** - Fetches fresh data when cache misses
4. **Auto-Storage** - Grows knowledge base with each search

## 📋 Requirements

### Python Dependencies
```
selenium>=4.0.0
pandas>=1.3.0
scikit-learn>=1.0.0
numpy>=1.21.0
Pillow>=9.0.0
requests>=2.26.0
webdriver-manager>=3.8.0
transformers>=4.35.0
torch>=2.0.0
datasets>=2.14.0
```

### System Requirements
- Python 3.8+
- Chrome Browser
- ChromeDriver (auto-installed via webdriver-manager)
- 8GB RAM minimum (recommended for neural models)
- Internet connection
- GPU optional (CUDA-enabled for faster inference)

## 🚀 Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd "Major Project"
```

2. **Install dependencies**
```bash
pip install selenium pandas scikit-learn numpy Pillow requests webdriver-manager transformers torch datasets
```

Or use requirements.txt:
```bash
pip install -r requirements.txt
```
```

3. **Run the application**
```bash
# Single Agent Version
python Try.py

# Multi-Agent Version (with dedicated sentiment analysis agent)
python multi_agent_scraper.py
```

## 💻 Usage

### Main Menu Options

**1. Search Products**
- Enter product name (e.g., "samsung watch", "iphone 15")
- Specify number of products per source (default: 5)
- View results in interactive GUI with sentiment analysis

**2. View Database Statistics**
- Total products stored
- Price statistics (min, max, average, median)
- Rating statistics
- Source and category breakdown

**3. Clear Database**
- Remove all cached products
- Start fresh with new searches

**4. Exit**
- Close the application

### Search Examples
```
🔍 Enter product name: samsung galaxy watch
📦 Products per source: 5

🔍 Enter product name: iphone 15 pro
📦 Products per source: 3

🔍 Enter product name: oneplus nord
📦 Products per source: 10
```

## 🏗️ Architecture

### Project Structure
```
Major Project/
├── Try.py                          # Single-agent application
├── multi_agent_scraper.py          # Multi-agent application with dedicated agents
├── neural_sentiment_analyzer.py    # DistilBERT-based sentiment analysis
├── README.md                       # Documentation
├── PROJECT_DOCUMENTATION.md        # Detailed technical documentation
├── requirements.txt                # Python dependencies
├── product_rag_database.pkl        # RAG storage (auto-generated)
└── knowledge_base.json             # Optional LLM enrichment data
```

### Key Components

**1. RAG Pipeline**
- `RAGPipeline`: Generic retrieval-augmented generation system
- `ProductRAGStorage`: Product-specific storage with TF-IDF vectorization
- Semantic search using cosine similarity

**2. Neural Sentiment Analyzer**
- `NeuralSentimentAnalyzer`: DistilBERT transformer model for sentiment analysis
- `DatasetLoader`: HuggingFace dataset loader for Amazon/Yelp reviews
- Pre-trained on SST-2 with ~91% accuracy

**3. Web Scrapers**
- `scrape_detailed_amazon()`: Amazon.in scraper with deep product details
- `scrape_detailed_flipkart()`: Flipkart scraper with retry logic
- `scrape_amazon_product_details()`: Extracts technical specifications
- `scrape_flipkart_product_details()`: Extracts product features

**4. Data Processing**
- `unified_rag_search()`: Orchestrates search workflow
- `filter_only_phones()`: Removes accessories for phone searches
- `categorize_product()`: Auto-categorizes products
- `clean_price()`: Normalizes price formats

**5. GUI**
- `display_results_gui_with_details()`: Interactive product comparison
- Async image loading
- Sentiment indicators with emoji and confidence scores
- Detailed product view with specifications and sentiment analysis
- Direct links to product pages

## 🎯 How It Works

### Search Workflow
```
User Query
    ↓
Local Exact Match (cached)
    ↓ (if not found)
Fuzzy Search (60% match)
    ↓ (if not found)
Web Scraping (Amazon + Flipkart)
    ↓
Validation & Filtering
    ↓
Neural Sentiment Analysis (DistilBERT)
    ↓
Store in RAG Database
    ↓
Display Results in GUI with Sentiment
```

### Sentiment Analysis Pipeline
```
Product Data
    ↓
Text Extraction (name + description)
    ↓
DistilBERT Tokenization (512 max tokens)
    ↓
Transformer Inference
    ↓
Softmax Classification
    ↓
Sentiment Label + Confidence Score
```

### Validation Logic
- **Accessory Detection**: Filters "back cover", "phone case", "screen protector", etc.
- **Brand Matching**: For brand-specific searches, validates brand presence
- **Generic Text Filtering**: Removes "Bestseller", "Coming Soon" placeholders
- **Price Validation**: Excludes products with invalid prices

### Scraping Strategy
- **Retry Logic**: 2 attempts with 30-second timeouts
- **Multiple Selectors**: Tries various CSS selectors for robustness
- **Tab Management**: Opens products in new tabs to preserve search results
- **Wait Times**: Random delays (3-10s) to avoid detection

## 📊 Data Extracted

### Basic Information
- Product name
- Price (numeric and formatted)
- Rating and review count
- Product image URL
- Source (Amazon/Flipkart)
- Product link
- Availability status

### Sentiment Information
- Sentiment label (Positive/Negative/Neutral)
- Confidence score (0-100%)
- Sentiment emoji (😊/😐/😞)
- Detailed explanation

### Detailed Information
- Technical specifications (dict)
- Additional product info (dict)
- Features and highlights (list)
- Full product description
- Category classification

## 🔧 Configuration

### Adjustable Parameters

**Search Settings**
```python
max_products = 5  # Products per source
target_count = max_products * 2  # Total target (both sources)
```

**Sentiment Analysis Settings**
```python
# Model configuration
model_name = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
max_length = 512  # Maximum tokens for analysis
device = "cuda" if torch.cuda.is_available() else "cpu"
```

**Validation Thresholds**
```python
fuzzy_match_threshold = 0.6  # 60% token match required
min_name_length = 10  # Minimum product name length
```

**Wait Times**
```python
page_load_timeout = 30  # Seconds
retry_attempts = 2  # Number of retries
```

## 🛡️ Error Handling

- **Timeout Protection**: Graceful handling of slow-loading pages
- **Element Not Found**: Multiple selector fallbacks
- **Tab Management**: Auto-cleanup of browser tabs on errors
- **Data Validation**: Skips invalid products without crashing
- **Model Loading**: Graceful degradation if transformers unavailable

## 📈 Performance

### Optimization Features
- **Caching**: Reduces redundant scraping via RAG storage
- **Batch Processing**: Stores multiple products at once
- **Async Image Loading**: Non-blocking GUI image display
- **Vectorized Search**: Fast semantic similarity using TF-IDF
- **GPU Acceleration**: Optional CUDA support for neural models

### Neural Model Specifications
- **Model**: DistilBERT (66M parameters)
- **Layers**: 6 transformer layers
- **Max Tokens**: 512
- **Accuracy**: ~91% on SST-2 benchmark
- **Inference**: ~50ms per product (CPU), ~5ms (GPU)

### Typical Performance
- First search: 30-60 seconds (scraping + sentiment analysis)
- Cached search: <1 second (local retrieval)
- Sentiment analysis: ~50ms per product
- Products per search: 2-20 (configurable)
- Storage size: ~1MB per 100 products
- Model size: ~268MB (downloaded once)

## ⚠️ Limitations

- **Website Changes**: Scrapers may break if Amazon/Flipkart update HTML structure
- **Rate Limiting**: Excessive requests may trigger anti-bot measures
- **Regional Availability**: Designed for Amazon.in and Flipkart India
- **Dynamic Content**: Some products may not load properly
- **No Authentication**: Cannot access user-specific prices or deals
- **Model Download**: First run requires ~268MB model download from HuggingFace

## 🔮 Future Enhancements

- [ ] Add more e-commerce platforms (Myntra, Snapdeal)
- [ ] Price history tracking and alerts
- [ ] Email notifications for price drops
- [ ] Export to Excel/CSV with charts
- [ ] Mobile app version
- [ ] API for third-party integration
- [ ] Fine-tune sentiment model on product-specific data
- [ ] Multi-language sentiment analysis support

## 📝 License

This project is for educational purposes. Respect the terms of service of Amazon and Flipkart when using this tool.

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📧 Contact

For questions or support, please open an issue in the repository.

## 🙏 Acknowledgments

- Selenium WebDriver for browser automation
- ChromeDriver for Chrome integration
- scikit-learn for semantic search capabilities
- HuggingFace Transformers for neural sentiment analysis
- DistilBERT model from HuggingFace Hub
- Flipkart scraping reference from [StackOverflow](https://stackoverflow.com/questions/28122882/) (CC BY-SA 3.0)

## 🧠 Neural Model Information

### Supported Datasets (for fine-tuning)
- **Amazon Polarity**: `mteb/amazon_polarity` - Binary sentiment classification
- **Amazon Reviews 2023**: `McAuley-Lab/Amazon-Reviews-2023` - Multi-class reviews
- **Yelp Reviews**: `Yelp/yelp_review_full` - 5-star rating prediction

### Pre-trained Model
- **DistilBERT SST-2**: `distilbert/distilbert-base-uncased-finetuned-sst-2-english`
- Fine-tuned on Stanford Sentiment Treebank
- 66 million parameters
- 6 transformer layers

---

**⚡ Built with Python | Powered by RAG & Neural Networks | Made for Smart Shopping**
