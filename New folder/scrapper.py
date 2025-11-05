from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import pandas as pd
import time
import random
import re
import json
import pickle
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import tkinter as tk
from tkinter import ttk, scrolledtext
from PIL import Image, ImageTk
import requests
from io import BytesIO
from threading import Thread
from webdriver_manager.chrome import ChromeDriverManager

# ===========================
# RAG Storage System
# ===========================

class ProductRAGStorage:
    """RAG-based storage system for product data with semantic search"""
    
    def __init__(self, storage_file='product_rag_db.pkl'):
        self.storage_file = storage_file
        self.products = []
        self.vectorizer = TfidfVectorizer(max_features=500)
        self.vectors = None
        self.load_storage()
    
    def add_product(self, product_data):
        """Add product to RAG storage"""
        product_data['timestamp'] = datetime.now().isoformat()
        product_data['id'] = f"{product_data['source']}_{len(self.products)}"
        self.products.append(product_data)
        self._update_vectors()
        self.save_storage()
    
    def add_products_batch(self, products_list):
        """Add multiple products at once"""
        for product in products_list:
            product['timestamp'] = datetime.now().isoformat()
            product['id'] = f"{product.get('source', 'unknown')}_{len(self.products)}"
            self.products.append(product)
        self._update_vectors()
        self.save_storage()
    
    def _update_vectors(self):
        """Update TF-IDF vectors for semantic search"""
        if not self.products:
            return
        
        texts = []
        for p in self.products:
            text = f"{p.get('name', '')} {p.get('subcategory', '')} {p.get('category', '')} {p.get('specifications', {})}"
            texts.append(text)
        
        self.vectors = self.vectorizer.fit_transform(texts)
    
    def semantic_search(self, query, top_k=10):
        """Search products using semantic similarity"""
        if not self.products or self.vectors is None:
            return []
        
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.vectors)[0]
        
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        results = []
        for idx in top_indices:
            product = self.products[idx].copy()
            product['similarity_score'] = float(similarities[idx])
            results.append(product)
        
        return results
    
    def filter_products(self, **filters):
        """Filter products by attributes"""
        filtered = self.products
        
        for key, value in filters.items():
            if key == 'price_range':
                min_price, max_price = value
                filtered = [p for p in filtered if min_price <= p.get('price_numeric', 0) <= max_price]
            elif key == 'min_rating':
                filtered = [p for p in filtered if self._extract_rating(p.get('rating', '')) >= value]
            elif key == 'source':
                filtered = [p for p in filtered if p.get('source') == value]
            elif key == 'category':
                filtered = [p for p in filtered if p.get('category') == value]
            else:
                filtered = [p for p in filtered if p.get(key) == value]
        
        return filtered
    
    def _extract_rating(self, rating_str):
        """Extract numeric rating from string"""
        if not rating_str:
            return 0.0
        match = re.search(r'(\d+\.?\d*)', str(rating_str))
        return float(match.group(1)) if match else 0.0
    
    def get_statistics(self):
        """Get statistics about stored products"""
        if not self.products:
            return {}
        
        stats = {
            'total_products': len(self.products),
            'by_source': defaultdict(int),
            'by_category': defaultdict(int),
            'price_stats': {},
            'rating_stats': {}
        }
        
        prices = []
        ratings = []
        
        for p in self.products:
            stats['by_source'][p.get('source', 'unknown')] += 1
            stats['by_category'][p.get('category', 'unknown')] += 1
            
            price = p.get('price_numeric', 0)
            if price > 0:
                prices.append(price)
            
            rating = self._extract_rating(p.get('rating', ''))
            if rating > 0:
                ratings.append(rating)
        
        if prices:
            stats['price_stats'] = {
                'min': min(prices),
                'max': max(prices),
                'avg': np.mean(prices),
                'median': np.median(prices)
            }
        
        if ratings:
            stats['rating_stats'] = {
                'min': min(ratings),
                'max': max(ratings),
                'avg': np.mean(ratings),
                'median': np.median(ratings)
            }
        
        return stats
    
    def save_storage(self):
        """Save storage to disk"""
        data = {
            'products': self.products,
            'vectorizer': self.vectorizer,
            'vectors': self.vectors
        }
        with open(self.storage_file, 'wb') as f:
            pickle.dump(data, f)
    
    def load_storage(self):
        """Load storage from disk"""
        try:
            with open(self.storage_file, 'rb') as f:
                data = pickle.load(f)
                self.products = data.get('products', [])
                self.vectorizer = data.get('vectorizer', TfidfVectorizer(max_features=500))
                self.vectors = data.get('vectors')
            print(f"Loaded {len(self.products)} products from storage")
        except FileNotFoundError:
            print("No existing storage found, starting fresh")
        except Exception as e:
            print(f"Error loading storage: {e}")
    
    def export_to_csv(self, filename='products_export.csv'):
        """Export products to CSV"""
        if not self.products:
            print("No products to export")
            return
        
        df = pd.DataFrame(self.products)
        df.to_csv(filename, index=False)
        print(f"Exported {len(self.products)} products to {filename}")
    
    def clear_storage(self):
        """Clear all stored products"""
        self.products = []
        self.vectors = None
        self.save_storage()

# ===========================
# Enhanced Web Scraping
# ===========================

user_agents = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'
]

def setup_driver():
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument(f"user-agent={random.choice(user_agents)}")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation", "enable-logging"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    chrome_options.add_experimental_option("prefs", {"profile.default_content_setting_values.notifications": 2})

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=chrome_options
    )

    try:
        driver.execute_cdp_cmd('Network.setUserAgentOverride', {"userAgent": random.choice(user_agents)})
    except Exception:
        pass
    try:
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    except Exception:
        pass
    driver.set_page_load_timeout(30)

    return driver

def scrape_detailed_amazon(driver, product_name):
    """Enhanced Amazon scraper with detailed specifications"""
    try:
        driver.get("https://www.amazon.in")
        time.sleep(random.uniform(3, 5))
        
        url = f"https://www.amazon.in/s?k={product_name.replace(' ', '+')}"
        driver.get(url)
        time.sleep(random.uniform(4, 6))
        
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "[data-component-type='s-search-result']"))
        )
        time.sleep(2)
    except TimeoutException:
        print("Amazon: Could not load results")
        return []
    except Exception as e:
        print(f"Amazon error: {str(e)}")
        return []
    
    products = []
    items = driver.find_elements(By.CSS_SELECTOR, "[data-component-type='s-search-result']")
    
    for item in items[:10]:
        try:
            # Basic info
            name_element = None
            name_selectors = ["h2 a span", "h2 span"]
            
            for selector in name_selectors:
                try:
                    name_element = item.find_element(By.CSS_SELECTOR, selector)
                    break
                except NoSuchElementException:
                    continue
            
            if not name_element:
                continue
                
            name = name_element.text.strip()
            if not name:
                continue
            
            # Price
            price_element = None
            price_selectors = [".a-price-whole", ".a-price .a-offscreen"]
            
            for selector in price_selectors:
                try:
                    price_element = item.find_element(By.CSS_SELECTOR, selector)
                    break
                except NoSuchElementException:
                    continue
            
            if not price_element:
                continue
                
            price_text = price_element.text.strip()
            if not price_text:
                continue
            
            # Image
            image_url = ""
            try:
                img_element = item.find_element(By.CSS_SELECTOR, "img.s-image")
                image_url = img_element.get_attribute("src")
            except:
                pass
            
            # Rating
            rating = ""
            try:
                rating_element = item.find_element(By.CSS_SELECTOR, ".a-icon-alt")
                rating = rating_element.get_attribute("textContent").strip()
            except:
                pass
            
            # Reviews count
            reviews = ""
            try:
                reviews_element = item.find_element(By.CSS_SELECTOR, ".a-size-base.s-underline-text")
                reviews = reviews_element.text.strip()
            except:
                pass
            
            # Subcategory
            subcategory = ""
            try:
                subcat_element = item.find_element(By.CSS_SELECTOR, ".a-size-base-plus.a-color-base")
                subcategory = subcat_element.text.strip()
            except:
                pass
            
            # Extract specifications from name (for phones)
            specifications = extract_specifications(name, subcategory)
            
            # Determine category
            category = categorize_product(name, subcategory)
            
            # Product link
            product_link = ""
            try:
                link_element = item.find_element(By.CSS_SELECTOR, "h2 a")
                product_link = link_element.get_attribute("href")
            except:
                pass
            
            # Availability
            availability = "In Stock"
            try:
                avail_element = item.find_element(By.CSS_SELECTOR, ".a-color-price, .a-color-secondary")
                avail_text = avail_element.text.strip().lower()
                if "unavailable" in avail_text or "out of stock" in avail_text:
                    availability = "Out of Stock"
            except:
                pass
                
            products.append({
                "name": name,
                "subcategory": subcategory,
                "price": price_text,
                "price_numeric": clean_price(price_text),
                "rating": rating,
                "reviews": reviews,
                "image_url": image_url,
                "category": category,
                "source": "Amazon.in",
                "product_link": product_link,
                "availability": availability,
                "specifications": specifications
            })
            
        except Exception:
            continue
    
    print(f"Found {len(products)} products on Amazon")
    return products

def scrape_detailed_flipkart(driver, product_name):
    """Enhanced Flipkart scraper with detailed specifications"""
    try:
        url = f"https://www.flipkart.com/search?q={product_name.replace(' ', '+')}"
        driver.get(url)
        time.sleep(random.uniform(5, 7))
        
        try:
            close_btn = WebDriverWait(driver, 3).until(
                EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), '✕')]"))
            )
            close_btn.click()
            time.sleep(1)
        except:
            pass
        
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "._1AtVbE, ._13oc-S, .tUxRFH, .CGtC98"))
        )
        time.sleep(3)
    except TimeoutException:
        print("Flipkart: Results took too long to load")
        return []
    except Exception as e:
        print(f"Flipkart error: {str(e)}")
        return []
    
    products = []
    item_selectors = ["._1AtVbE", "._13oc-S", ".tUxRFH", ".CGtC98"]
    items = []
    
    for selector in item_selectors:
        items = driver.find_elements(By.CSS_SELECTOR, selector)
        if items:
            print(f"Flipkart: Found {len(items)} items")
            break
    
    for item in items[:10]:
        try:
            # Basic info
            name_element = None
            name_selectors = ["._4rR01T", ".IRpwTa", "._2WkVRV", ".KzDlHZ", ".s1Q9rs"]
            
            for selector in name_selectors:
                try:
                    name_element = item.find_element(By.CSS_SELECTOR, selector)
                    break
                except NoSuchElementException:
                    continue
            
            if not name_element:
                continue
                
            name = name_element.text.strip()
            if not name:
                continue
            
            # Price
            price_element = None
            price_selectors = ["._30jeq3", "._1_WHN1", ".Nx9bqj", "._4b5DiR"]
            
            for selector in price_selectors:
                try:
                    price_element = item.find_element(By.CSS_SELECTOR, selector)
                    break
                except NoSuchElementException:
                    continue
            
            if not price_element:
                continue
                
            price_text = price_element.text.strip()
            if not price_text:
                continue
            
            # Image
            image_url = ""
            try:
                img_element = item.find_element(By.CSS_SELECTOR, "img")
                image_url = img_element.get_attribute("src")
            except:
                pass
            
            # Rating
            rating = ""
            try:
                rating_element = item.find_element(By.CSS_SELECTOR, "div[class*='_3LWZlK']")
                rating = rating_element.text.strip()
            except:
                pass
            
            # Reviews
            reviews = ""
            try:
                reviews_element = item.find_element(By.CSS_SELECTOR, "span[class*='_2_R_DZ']")
                reviews = reviews_element.text.strip()
            except:
                pass
            
            # Subcategory
            subcategory = ""
            try:
                subcat_element = item.find_element(By.CSS_SELECTOR, "._2WkVRV, .NqpwHC")
                subcategory = subcat_element.text.strip()
            except:
                pass
            
            # Extract specifications
            specifications = extract_specifications(name, subcategory)
            
            # Category
            category = categorize_product(name, subcategory)
            
            # Product link
            product_link = ""
            try:
                link_element = item.find_element(By.CSS_SELECTOR, "a")
                product_link = "https://www.flipkart.com" + link_element.get_attribute("href")
            except:
                pass
            
            # Availability
            availability = "In Stock"
            try:
                avail_elements = item.find_elements(By.CSS_SELECTOR, "div")
                for elem in avail_elements:
                    text = elem.text.strip().lower()
                    if "sold out" in text or "out of stock" in text:
                        availability = "Out of Stock"
                        break
            except:
                pass
                
            products.append({
                "name": name,
                "subcategory": subcategory,
                "price": price_text,
                "price_numeric": clean_price(price_text),
                "rating": rating,
                "reviews": reviews,
                "image_url": image_url,
                "category": category,
                "source": "Flipkart",
                "product_link": product_link,
                "availability": availability,
                "specifications": specifications
            })
            
        except Exception:
            continue
    
    print(f"Found {len(products)} products on Flipkart")
    return products

def extract_specifications(name, subcategory):
    """Extract technical specifications from product name and description"""
    text = f"{name} {subcategory}".lower()
    specs = {}
    
    # Storage
    storage_match = re.search(r'(\d+)\s*(gb|tb)', text)
    if storage_match:
        specs['storage'] = f"{storage_match.group(1)} {storage_match.group(2).upper()}"
    
    # RAM
    ram_match = re.search(r'(\d+)\s*gb\s*ram', text)
    if ram_match:
        specs['ram'] = f"{ram_match.group(1)} GB"
    
    # Display size
    display_match = re.search(r'(\d+\.?\d*)\s*(inch|cm|")', text)
    if display_match:
        specs['display'] = f"{display_match.group(1)} {display_match.group(2)}"
    
    # Camera
    camera_match = re.search(r'(\d+)\s*mp', text)
    if camera_match:
        specs['camera'] = f"{camera_match.group(1)} MP"
    
    # Battery
    battery_match = re.search(r'(\d+)\s*mah', text)
    if battery_match:
        specs['battery'] = f"{battery_match.group(1)} mAh"
    
    # Processor
    processors = ['snapdragon', 'mediatek', 'helio', 'exynos', 'bionic', 'tensor', 'dimensity']
    for proc in processors:
        if proc in text:
            proc_match = re.search(rf'{proc}\s*(\d+)', text)
            if proc_match:
                specs['processor'] = f"{proc.capitalize()} {proc_match.group(1)}"
            else:
                specs['processor'] = proc.capitalize()
            break
    
    # OS
    if 'ios' in text or 'iphone' in text:
        specs['os'] = 'iOS'
    elif 'android' in text:
        specs['os'] = 'Android'
    
    # Color
    colors = ['black', 'white', 'blue', 'red', 'green', 'silver', 'gold', 'pink', 'purple', 'gray', 'grey']
    for color in colors:
        if color in text:
            specs['color'] = color.capitalize()
            break
    
    return specs

def categorize_product(name, subcategory=""):
    name_lower = (name + " " + subcategory).lower()
    
    shoe_keywords = ['shoe', 'sneaker', 'boot', 'sandal', 'slipper', 'footwear', 'nike air', 'jordan', 'running']
    clothing_keywords = ['shirt', 't-shirt', 'pant', 'jean', 'jacket', 'hoodie', 'dress', 'skirt', 'shorts', 'clothing', 'apparel']
    electronics_keywords = ['phone', 'mobile', 'smartphone', 'laptop', 'tablet', 'headphone', 'earphone', 'watch', 'camera']
    
    for keyword in electronics_keywords:
        if keyword in name_lower:
            return "Electronics"
    
    for keyword in shoe_keywords:
        if keyword in name_lower:
            return "Shoes"
    
    for keyword in clothing_keywords:
        if keyword in name_lower:
            return "Clothing"
    
    return "Other"

def clean_price(price_text):
    if not price_text:
        return 0.0
    
    cleaned = re.sub(r'[₹,\s]', '', price_text)
    numbers = re.findall(r'\d+\.?\d*', cleaned)
    
    if numbers:
        try:
            return float(numbers[0])
        except ValueError:
            return 0.0
    return 0.0

def filter_only_phones(products, search_term):
    """Keep only phone-related products"""
    if not products:
        return products

    generic_terms = {"mobile", "phone", "phones", "smartphone", "smartphones", "cell"}
    tokens = [t for t in re.split(r'\W+', search_term.lower()) if t and t not in generic_terms]

    fallback_include = {"phone", "mobile", "smartphone", "iphone"}

    exclude_keywords = [
        "case", "cover", "bumper", "back cover", "protective",
        "charger", "cable", "adapter", "tempered", "screen guard",
        "screen protector", "glass", "protector", "pouch", "strap",
        "stand", "skin", "holder", "battery", "lens", "camera protector",
        "headphone", "earphone", "powerbank", "power bank", "wire", "ring", "popsocket"
    ]

    filtered = []
    for p in products:
        title = p.get("name", "").lower()
        if not title:
            continue

        if any(ex in title for ex in exclude_keywords):
            continue

        if tokens:
            if any(tok in title for tok in tokens):
                filtered.append(p)
            else:
                if any(f in title for f in fallback_include):
                    filtered.append(p)
        else:
            if any(f in title for f in fallback_include):
                filtered.append(p)

    return filtered

# ===========================
# Visualization Functions
# ===========================

def create_price_comparison_graph(products):
    """Create price comparison visualization"""
    if not products:
        return
    
    df = pd.DataFrame(products)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Product Analysis Dashboard', fontsize=16, fontweight='bold')
    
    # Price distribution by source
    if 'source' in df.columns and 'price_numeric' in df.columns:
        df_price = df[df['price_numeric'] > 0]
        sources = df_price['source'].unique()
        
        ax = axes[0, 0]
        for source in sources:
            source_data = df_price[df_price['source'] == source]['price_numeric']
            ax.hist(source_data, alpha=0.6, label=source, bins=15)
        ax.set_xlabel('Price (₹)')
        ax.set_ylabel('Frequency')
        ax.set_title('Price Distribution by Source')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Price comparison boxplot
    if 'source' in df.columns and 'price_numeric' in df.columns:
        ax = axes[0, 1]
        df_price = df[df['price_numeric'] > 0]
        sources = df_price['source'].unique()
        data_to_plot = [df_price[df_price['source'] == s]['price_numeric'].values for s in sources]
        ax.boxplot(data_to_plot, labels=sources)
        ax.set_ylabel('Price (₹)')
        ax.set_title('Price Range by Source')
        ax.grid(True, alpha=0.3)
    
    # Category distribution
    if 'category' in df.columns:
        ax = axes[1, 0]
        category_counts = df['category'].value_counts()
        colors = plt.cm.Set3(range(len(category_counts)))
        ax.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', colors=colors)
        ax.set_title('Products by Category')
    
    # Rating distribution
    if 'rating' in df.columns:
        ax = axes[1, 1]
        ratings = []
        for rating_str in df['rating']:
            match = re.search(r'(\d+\.?\d*)', str(rating_str))
            if match:
                ratings.append(float(match.group(1)))
        
        if ratings:
            ax.hist(ratings, bins=10, color='skyblue', edgecolor='black')
            ax.set_xlabel('Rating')
            ax.set_ylabel('Frequency')
            ax.set_title('Rating Distribution')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('product_analysis.png', dpi=300, bbox_inches='tight')
    print("Graph saved as 'product_analysis.png'")
    plt.show()

def create_detailed_report(rag_storage):
    """Create detailed statistics report"""
    stats = rag_storage.get_statistics()
    
    if not stats:
        print("No statistics available")
        return
    
    print("\n" + "="*60)
    print("PRODUCT DATABASE STATISTICS")
    print("="*60)
    print(f"Total Products: {stats['total_products']}")
    
    print("\nProducts by Source:")
    for source, count in stats['by_source'].items():
        print(f"  {source}: {count}")
    
    print("\nProducts by Category:")
    for category, count in stats['by_category'].items():
        print(f"  {category}: {count}")
    
    if stats.get('price_stats'):
        print("\nPrice Statistics:")
        ps = stats['price_stats']
        print(f"  Min: ₹{ps['min']:.2f}")
        print(f"  Max: ₹{ps['max']:.2f}")
        print(f"  Average: ₹{ps['avg']:.2f}")
        print(f"  Median: ₹{ps['median']:.2f}")
    
    if stats.get('rating_stats'):
        print("\nRating Statistics:")
        rs = stats['rating_stats']
        print(f"  Min: {rs['min']:.1f}")
        print(f"  Max: {rs['max']:.1f}")
        print(f"  Average: {rs['avg']:.2f}")
        print(f"  Median: {rs['median']:.2f}")
    
    print("="*60 + "\n")

# ===========================
# Main Agent Function
# ===========================

def smart_product_agent(product_name, rag_storage):
    """Main agent that scrapes, stores, and analyzes products"""
    print(f"\n{'='*60}")
    print(f"Smart Product Agent Starting...")
    print(f"Search Query: {product_name}")
    print(f"{'='*60}\n")
    
    driver = setup_driver()
    
    try:
        print("Step 1: Scraping Amazon...")
        amazon_products = scrape_detailed_amazon(driver, product_name)
        
        print("Step 2: Scraping Flipkart...")
        flipkart_products = scrape_detailed_flipkart(driver, product_name)
        
    finally:
        driver.quit()

    all_products = amazon_products + flipkart_products
    
    # Filter for phone-specific searches
    if any(term in product_name.lower() for term in ['phone', 'mobile', 'smartphone', 'iphone']):
        all_products = filter_only_phones(all_products, product_name)

    if not all_products:
        print("No products found after filtering.")
        return None

    print(f"\nStep 3: Storing {len(all_products)} products in RAG database...")
    rag_storage.add_products_batch(all_products)
    
    print("Step 4: Generating analysis and visualizations...")
    create_price_comparison_graph(all_products)
    
    print("Step 5: Creating detailed report...")
    create_detailed_report(rag_storage)
    
    df = pd.DataFrame(all_products)
    df = df.sort_values(by="price_numeric")
    
    print(f"\nAgent completed successfully!")
    print(f"Found {len(df)} products")
    if not df.empty and 'price_numeric' in df.columns:
        print(f"Price range: ₹{df['price_numeric'].min():.0f} - ₹{df['price_numeric'].max():.0f}")
    
    return df

# ===========================
# Enhanced GUI with RAG Search
# ===========================

def load_image_from_url(url, size=(130, 130)):
    try:
        response = requests.get(url, timeout=4)
        img = Image.open(BytesIO(response.content))
        img = img.resize(size, Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(img)
    except Exception:
        return None

def load_image_async(img_label, url):
    """Load image in background thread"""
    try:
        img = load_image_from_url(url)
        if img and img_label.winfo_exists():
            img_label.config(image=img, text="", bg="white")
            img_label.image = img
    except:
        pass

def display_results_gui_with_rag(df, rag_storage):
    """Enhanced GUI with RAG search capabilities"""
    root = tk.Tk()
    root.title("Smart Product Comparison with RAG")
    root.geometry("1400x800")
    
    # Top control panel
    control_frame = ttk.Frame(root)
    control_frame.pack(fill=tk.X, padx=10, pady=10)
    
    # Search box
    search_label = ttk.Label(control_frame, text="Semantic Search:")
    search_label.pack(side=tk.LEFT, padx=5)
    
    search_var = tk.StringVar()
    search_entry = ttk.Entry(control_frame, textvariable=search_var, width=40)
    search_entry.pack(side=tk.LEFT, padx=5)
    
    # Filter options
    ttk.Label(control_frame, text="Source:").pack(side=tk.LEFT, padx=5)
    source_var = tk.StringVar(value="All")
    source_combo = ttk.Combobox(control_frame, textvariable=source_var, 
                                values=["All", "Amazon.in", "Flipkart"], width=15)
    source_combo.pack(side=tk.LEFT, padx=5)
    
    ttk.Label(control_frame, text="Category:").pack(side=tk.LEFT, padx=5)
    category_var = tk.StringVar(value="All")
    category_combo = ttk.Combobox(control_frame, textvariable=category_var,
                                  values=["All", "Electronics", "Shoes", "Clothing", "Other"], width=15)
    category_combo.pack(side=tk.LEFT, padx=5)
    
    # Statistics button
    def show_statistics():
        stats_window = tk.Toplevel(root)
        stats_window.title("Database Statistics")
        stats_window.geometry("600x400")
        
        text_widget = scrolledtext.ScrolledText(stats_window, wrap=tk.WORD, font=("Courier", 10))
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        stats = rag_storage.get_statistics()
        stats_text = f"""
DATABASE STATISTICS
{'='*60}
Total Products: {stats.get('total_products', 0)}

Products by Source:
"""
        for source, count in stats.get('by_source', {}).items():
            stats_text += f"  {source}: {count}\n"
        
        stats_text += "\nProducts by Category:\n"
        for category, count in stats.get('by_category', {}).items():
            stats_text += f"  {category}: {count}\n"
        
        if stats.get('price_stats'):
            ps = stats['price_stats']
            stats_text += f"""
Price Statistics:
  Min: ₹{ps['min']:.2f}
  Max: ₹{ps['max']:.2f}
  Average: ₹{ps['avg']:.2f}
  Median: ₹{ps['median']:.2f}
"""
        
        if stats.get('rating_stats'):
            rs = stats['rating_stats']
            stats_text += f"""
Rating Statistics:
  Min: {rs['min']:.1f}
  Max: {rs['max']:.1f}
  Average: {rs['avg']:.2f}
  Median: {rs['median']:.2f}
"""
        
        text_widget.insert(tk.END, stats_text)
        text_widget.config(state=tk.DISABLED)
    
    stats_btn = ttk.Button(control_frame, text="View Statistics", command=show_statistics)
    stats_btn.pack(side=tk.LEFT, padx=10)
    
    # Export button
    def export_data():
        rag_storage.export_to_csv('products_export.csv')
        tk.messagebox.showinfo("Export", "Data exported to products_export.csv")
    
    export_btn = ttk.Button(control_frame, text="Export to CSV", command=export_data)
    export_btn.pack(side=tk.LEFT, padx=5)
    
    # Main frame with scrollbar
    main_frame = ttk.Frame(root)
    main_frame.pack(fill=tk.BOTH, expand=1, padx=10, pady=5)
    
    canvas = tk.Canvas(main_frame)
    scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=canvas.yview)
    scrollable_frame = ttk.Frame(canvas)
    
    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )
    
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    
    def display_products(products_df):
        # Clear existing widgets
        for widget in scrollable_frame.winfo_children():
            widget.destroy()
        
        if products_df.empty:
            no_results = tk.Label(scrollable_frame, text="No products found", 
                                 font=("Arial", 14), pady=20)
            no_results.pack()
            return
        
        # Header
        header = tk.Label(scrollable_frame, 
                         text=f"Product Comparison Results ({len(products_df)} products)", 
                         font=("Arial", 16, "bold"), pady=10)
        header.pack()
        
        # Display products
        for idx, row in products_df.iterrows():
            product_frame = tk.Frame(scrollable_frame, relief=tk.RAISED, 
                                   borderwidth=2, padx=10, pady=10, bg="white")
            product_frame.pack(fill=tk.X, padx=5, pady=5)
            
            # Image
            img_label = tk.Label(product_frame, text="Loading...", 
                               width=130, height=130, bg="lightgray")
            img_label.grid(row=0, column=0, rowspan=5, padx=10, sticky=tk.N)
            
            if row.get('image_url'):
                try:
                    img = load_image_from_url(row['image_url'])
                    if img:
                        img_label.config(image=img, text="", bg="white")
                        img_label.image = img
                except:
                    pass
            
            # Product details
            name_label = tk.Label(product_frame, text=row['name'], 
                                font=("Arial", 11, "bold"), wraplength=700, 
                                justify=tk.LEFT, bg="white")
            name_label.grid(row=0, column=1, sticky=tk.W, padx=10, pady=2)
            
            # Subcategory
            if row.get('subcategory'):
                subcat_label = tk.Label(product_frame, text=row['subcategory'], 
                                      font=("Arial", 9), fg="gray", bg="white")
                subcat_label.grid(row=1, column=1, sticky=tk.W, padx=10, pady=2)
            
            # Price
            price_text = f"Price: ₹{row['price_numeric']:.0f}"
            if row.get('availability') == "Out of Stock":
                price_text += " (Out of Stock)"
                price_color = "red"
            else:
                price_color = "green"
            
            price_label = tk.Label(product_frame, text=price_text, 
                                 font=("Arial", 13, "bold"), fg=price_color, bg="white")
            price_label.grid(row=2, column=1, sticky=tk.W, padx=10, pady=2)
            
            # Info line
            info_parts = [f"Source: {row['source']}", f"Category: {row['category']}"]
            if row.get('rating'):
                info_parts.append(f"Rating: {row['rating']}")
            if row.get('reviews'):
                info_parts.append(f"Reviews: {row['reviews']}")
            
            info_text = " | ".join(info_parts)
            info_label = tk.Label(product_frame, text=info_text, 
                                font=("Arial", 9), bg="white")
            info_label.grid(row=3, column=1, sticky=tk.W, padx=10, pady=2)
            
            # Specifications
            specs = row.get('specifications', {})
            if specs and isinstance(specs, dict):
                specs_text = "Specs: " + " | ".join([f"{k}: {v}" for k, v in specs.items()])
                specs_label = tk.Label(product_frame, text=specs_text, 
                                     font=("Arial", 9), fg="navy", bg="white")
                specs_label.grid(row=4, column=1, sticky=tk.W, padx=10, pady=2)
            
            # Similarity score if available
            if 'similarity_score' in row and row['similarity_score'] > 0:
                similarity_label = tk.Label(product_frame, 
                                          text=f"Relevance: {row['similarity_score']:.2%}", 
                                          font=("Arial", 9), fg="purple", bg="white")
                similarity_label.grid(row=5, column=1, sticky=tk.W, padx=10, pady=2)
    
    def apply_filters():
        query = search_var.get().strip()
        source = source_var.get()
        category = category_var.get()
        
        if query:
            # Semantic search
            results = rag_storage.semantic_search(query, top_k=50)
            filtered_df = pd.DataFrame(results)
        else:
            # Use current dataframe
            filtered_df = df.copy()
        
        # Apply filters
        if not filtered_df.empty:
            if source != "All":
                filtered_df = filtered_df[filtered_df['source'] == source]
            if category != "All":
                filtered_df = filtered_df[filtered_df['category'] == category]
        
        display_products(filtered_df)
    
    search_btn = ttk.Button(control_frame, text="Search/Filter", command=apply_filters)
    search_btn.pack(side=tk.LEFT, padx=5)
    
    # Initial display
    display_products(df)
    
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    root.mainloop()

# ===========================
# Main Execution
# ===========================

if __name__ == "__main__":
    # Initialize RAG storage
    rag_storage = ProductRAGStorage('product_rag_database.pkl')
    
    print("\n" + "="*60)
    print("SMART PRODUCT COMPARISON AGENT WITH RAG")
    print("="*60)
    print("\nOptions:")
    print("1. Search new products")
    print("2. Search in existing database")
    print("3. View database statistics")
    print("4. Export database to CSV")
    print("5. Clear database")
    print("6. Exit")
    
    while True:
        choice = input("\nEnter choice (1-6): ").strip()
        
        if choice == "1":
            product_name = input("\nEnter product name to search: ").strip()
            if product_name:
                result_df = smart_product_agent(product_name, rag_storage)
                if result_df is not None and not result_df.empty:
                    print("\nOpening GUI...")
                    display_results_gui_with_rag(result_df, rag_storage)
                else:
                    print("\nNo products found.")
        
        elif choice == "2":
            query = input("\nEnter search query: ").strip()
            if query:
                results = rag_storage.semantic_search(query, top_k=20)
                if results:
                    result_df = pd.DataFrame(results)
                    print(f"\nFound {len(result_df)} matching products")
                    display_results_gui_with_rag(result_df, rag_storage)
                else:
                    print("\nNo matching products found in database.")
        
        elif choice == "3":
            create_detailed_report(rag_storage)
        
        elif choice == "4":
            filename = input("Enter filename (default: products_export.csv): ").strip()
            if not filename:
                filename = "products_export.csv"
            rag_storage.export_to_csv(filename)
        
        elif choice == "5":
            confirm = input("Are you sure you want to clear all data? (yes/no): ").strip().lower()
            if confirm == "yes":
                rag_storage.clear_storage()
                print("Database cleared successfully.")
        
        elif choice == "6":
            print("\nExiting... Goodbye!")
            break
        
        else:
            print("Invalid choice. Please enter 1-6.")