"""
Fix PKL Data: Add missing fields to scraped pkl files

This script adds:
- price_numeric (parsed float prices)
- sentiment_score, sentiment_label, sentiment (from neural sentiment analyzer)

Run this instead of re-running the full comparison test.
"""

import pickle
import re
from datetime import datetime
from typing import Dict, List

print("\n" + "=" * 60)
print("   FIX PKL DATA - Add Missing Fields")
print("=" * 60)

# Try to import sentiment analyzer
try:
    from neural_sentiment_analyzer import NeuralSentimentAnalyzer, TRANSFORMERS_AVAILABLE
    SENTIMENT_AVAILABLE = TRANSFORMERS_AVAILABLE
    print("✓ Neural sentiment analyzer available")
except ImportError:
    SENTIMENT_AVAILABLE = False
    print("⚠️ Neural sentiment not available - using keyword-based fallback")


def extract_numeric_price(price_str) -> float:
    """Extract numeric price from various formats"""
    if price_str is None:
        return None
    price_str = str(price_str)
    cleaned = re.sub(r'[₹$,\s]', '', price_str)
    match = re.search(r'[\d.]+', cleaned)
    if match:
        try:
            return float(match.group())
        except ValueError:
            return None
    return None


def keyword_sentiment(text: str) -> Dict:
    """Simple keyword-based sentiment analysis"""
    positive_words = ['excellent', 'amazing', 'great', 'good', 'best', 'love', 
                      'perfect', 'awesome', 'fantastic', 'wonderful', 'quality',
                      'recommended', 'satisfied', 'happy', 'fast', 'value']
    negative_words = ['bad', 'poor', 'terrible', 'worst', 'hate', 'awful',
                      'disappointing', 'broken', 'defective', 'slow', 'waste',
                      'cheap', 'useless', 'horrible', 'problem', 'issue']
    
    text_lower = text.lower()
    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)
    
    if pos_count > neg_count:
        sentiment = 'positive'
        score = min(0.6 + pos_count * 0.05, 0.95)
    elif neg_count > pos_count:
        sentiment = 'negative'
        score = max(0.4 - neg_count * 0.05, 0.1)
    else:
        sentiment = 'neutral'
        score = 0.5
    
    confidence = min(60 + abs(pos_count - neg_count) * 5, 95)
    
    return {
        'sentiment': sentiment,
        'score': score,
        'confidence': confidence
    }


def add_missing_fields(products: List[Dict], use_neural: bool = True) -> List[Dict]:
    """Add price_numeric and sentiment fields to products"""
    
    neural_analyzer = None
    if use_neural and SENTIMENT_AVAILABLE:
        try:
            neural_analyzer = NeuralSentimentAnalyzer()
            print("✓ Using neural sentiment analyzer")
        except Exception as e:
            print(f"⚠️ Neural analyzer failed: {e}")
            neural_analyzer = None
    
    updated = 0
    for i, product in enumerate(products):
        # Add price_numeric
        if not product.get('price_numeric'):
            product['price_numeric'] = extract_numeric_price(product.get('price'))
        
        # Add sentiment fields
        if not product.get('sentiment_score'):
            # Combine text for analysis
            text_parts = []
            if product.get('name'):
                text_parts.append(str(product['name']))
            if product.get('description'):
                text_parts.append(str(product['description'])[:500])
            if product.get('review_summary'):
                text_parts.append(str(product['review_summary'])[:500])
            if product.get('customer_reviews'):
                reviews = product['customer_reviews']
                if isinstance(reviews, list):
                    for r in reviews[:3]:
                        if isinstance(r, dict):
                            text_parts.append(str(r.get('text', ''))[:200])
                        else:
                            text_parts.append(str(r)[:200])
                else:
                    text_parts.append(str(reviews)[:500])
            
            text = ' '.join(text_parts)[:1500]
            
            if text.strip():
                if neural_analyzer:
                    try:
                        result = neural_analyzer.analyze(text)
                        product['sentiment'] = result.get('sentiment', 'neutral')
                        product['sentiment_score'] = result.get('score', 0.5)
                        product['sentiment_label'] = result.get('sentiment', 'neutral').capitalize()
                        product['sentiment_confidence'] = result.get('confidence', 0.5) * 100
                    except:
                        result = keyword_sentiment(text)
                        product['sentiment'] = result['sentiment']
                        product['sentiment_score'] = result['score']
                        product['sentiment_label'] = result['sentiment'].capitalize()
                        product['sentiment_confidence'] = result['confidence']
                else:
                    result = keyword_sentiment(text)
                    product['sentiment'] = result['sentiment']
                    product['sentiment_score'] = result['score']
                    product['sentiment_label'] = result['sentiment'].capitalize()
                    product['sentiment_confidence'] = result['confidence']
            else:
                product['sentiment'] = 'neutral'
                product['sentiment_score'] = 0.5
                product['sentiment_label'] = 'Neutral'
                product['sentiment_confidence'] = 50.0
            
            updated += 1
        
        if (i + 1) % 25 == 0:
            print(f"   Processed {i+1}/{len(products)} products...")
    
    return products, updated


def fix_pkl_file(filepath: str) -> bool:
    """Fix a single pkl file"""
    print(f"\n📂 Processing: {filepath}")
    
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        products = data.get('products', [])
        print(f"   Found {len(products)} products")
        
        # Check current state
        has_sentiment = sum(1 for p in products if p.get('sentiment_score'))
        has_price_numeric = sum(1 for p in products if p.get('price_numeric'))
        
        print(f"   Current state:")
        print(f"     - sentiment_score: {has_sentiment}/{len(products)}")
        print(f"     - price_numeric: {has_price_numeric}/{len(products)}")
        
        if has_sentiment == len(products) and has_price_numeric == len(products):
            print("   ✓ All fields already present!")
            return True
        
        # Add missing fields
        products, updated = add_missing_fields(products)
        print(f"   Updated {updated} products")
        
        # Save updated file
        data['products'] = products
        data['fixed_at'] = datetime.now().isoformat()
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"   ✅ Saved: {filepath}")
        
        # Verify
        has_sentiment = sum(1 for p in products if p.get('sentiment_score'))
        has_price_numeric = sum(1 for p in products if p.get('price_numeric'))
        print(f"   Final state:")
        print(f"     - sentiment_score: {has_sentiment}/{len(products)}")
        print(f"     - price_numeric: {has_price_numeric}/{len(products)}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


if __name__ == "__main__":
    import os
    
    files_to_fix = ['scraped_multi_agent.pkl', 'scraped_single_agent.pkl']
    
    for filepath in files_to_fix:
        if os.path.exists(filepath):
            fix_pkl_file(filepath)
        else:
            print(f"\n⚠️ File not found: {filepath}")
    
    print("\n" + "=" * 60)
    print("   FIX COMPLETE!")
    print("=" * 60)
    print("\n📊 Now run:")
    print("   python analyze_pkl_results.py")
    print("   python umap_rag_analyzer.py")
