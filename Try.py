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
            # Include all text fields for better semantic search
            text_parts = [
                p.get('name', ''),
                p.get('subcategory', ''),
                p.get('category', ''),
                str(p.get('technical_details', {})),
                str(p.get('additional_info', {})),
                p.get('description', '')
            ]
            texts.append(' '.join(text_parts))
        
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
    
    def get_statistics(self):
        """Get statistics about stored products"""
        if not self.products:
            return {}
        
        stats = {
            'total_products': len(self.products),
            'by_source': defaultdict(int),
            'by_category': defaultdict(int),
            'price_stats': {},
            'rating_stats': {},
            'detailed_products': 0
        }
        
        prices = []
        ratings = []
        
        for p in self.products:
            stats['by_source'][p.get('source', 'unknown')] += 1
            stats['by_category'][p.get('category', 'unknown')] += 1
            
            if p.get('technical_details'):
                stats['detailed_products'] += 1
            
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
    
    def _extract_rating(self, rating_str):
        """Extract numeric rating from string"""
        if not rating_str:
            return 0.0
        match = re.search(r'(\d+\.?\d*)', str(rating_str))
        return float(match.group(1)) if match else 0.0
    
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
        
        # Flatten nested dictionaries for CSV
        flattened = []
        for p in self.products:
            flat = p.copy()
            if 'technical_details' in flat and isinstance(flat['technical_details'], dict):
                for k, v in flat['technical_details'].items():
                    flat[f'tech_{k}'] = v
                del flat['technical_details']
            if 'additional_info' in flat and isinstance(flat['additional_info'], dict):
                for k, v in flat['additional_info'].items():
                    flat[f'info_{k}'] = v
                del flat['additional_info']
            flattened.append(flat)
        
        df = pd.DataFrame(flattened)
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


def scrape_amazon_product_details(driver, product_url):
    """Scrape detailed information from Amazon product page"""
    try:
        if product_url:
            driver.get(product_url)
            time.sleep(random.uniform(3, 5))
        
        details = {
            'technical_details': {},
            'additional_info': {},
            'description': '',
            'features': []
        }
        
        # Technical Details
        try:
            tech_table = driver.find_element(By.ID, "productDetails_techSpec_section_1")
            rows = tech_table.find_elements(By.TAG_NAME, "tr")
            for row in rows:
                try:
                    th = row.find_element(By.TAG_NAME, "th").text.strip()
                    td = row.find_element(By.TAG_NAME, "td").text.strip()
                    if th and td:
                        details['technical_details'][th] = td
                except:
                    continue
        except:
            pass
        
        # Additional Information
        try:
            additional_table = driver.find_element(By.ID, "productDetails_detailBullets_sections1")
            rows = additional_table.find_elements(By.TAG_NAME, "tr")
            for row in rows:
                try:
                    th = row.find_element(By.TAG_NAME, "th").text.strip()
                    td = row.find_element(By.TAG_NAME, "td").text.strip()
                    if th and td:
                        details['additional_info'][th] = td
                except:
                    continue
        except:
            pass
        
        # Product Description
        try:
            desc_element = driver.find_element(By.ID, "feature-bullets")
            features = desc_element.find_elements(By.TAG_NAME, "li")
            for feat in features:
                text = feat.text.strip()
                if text:
                    details['features'].append(text)
            details['description'] = ' | '.join(details['features'])
        except:
            pass
        
        # Alternative description location
        if not details['description']:
            try:
                desc = driver.find_element(By.ID, "productDescription")
                details['description'] = desc.text.strip()
            except:
                pass
        
        return details
        
    except Exception as e:
        print(f"Error scraping Amazon details: {e}")
        return None


def scrape_flipkart_product_details(driver, product_url):
    """Scrape detailed information from Flipkart product page"""
    try:
        # Only navigate if URL is provided (for backward compatibility)
        if product_url:
            driver.get(product_url)
            time.sleep(random.uniform(3, 5))
        
        details = {
            'technical_details': {},
            'additional_info': {},
            'description': '',
            'features': []
        }
        
        # Specifications
        try:
            spec_sections = driver.find_elements(By.CSS_SELECTOR, "._1s76Cw, .GNDEQ-")
            for section in spec_sections:
                try:
                    section_title = section.find_element(By.CSS_SELECTOR, "._1AtVbE, ._3dtsli").text.strip()
                    rows = section.find_elements(By.CSS_SELECTOR, "._2-N8s, .WJdYP6")
                    
                    for row in rows:
                        try:
                            cols = row.find_elements(By.CSS_SELECTOR, "._2H-kL, ._2vIOIi, .URwL2w, ._21Ahn-")
                            if len(cols) >= 2:
                                key = cols[0].text.strip()
                                value = cols[1].text.strip()
                                if key and value:
                                    details['technical_details'][f"{section_title} - {key}"] = value
                        except:
                            continue
                except:
                    continue
        except:
            pass
        
        # Description
        try:
            desc_element = driver.find_element(By.CSS_SELECTOR, "._1mXcCf, ._2418kt")
            details['description'] = desc_element.text.strip()
        except:
            pass
        
        # Highlights
        try:
            highlights = driver.find_elements(By.CSS_SELECTOR, "._21Ahn-")
            for hl in highlights:
                text = hl.text.strip()
                if text:
                    details['features'].append(text)
        except:
            pass
        
        return details
        
    except Exception as e:
        print(f"Error scraping Flipkart details: {e}")
        return None


def scrape_detailed_amazon(driver, product_name, max_products=5):
    """Enhanced Amazon scraper with deep product details"""
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
    
    print(f"Amazon: Found {len(items)} items, scraping details for top {max_products}...")
    
    for idx, item in enumerate(items[:max_products]):
        try:
            print(f"  Scraping Amazon product {idx+1}/{max_products}...")
            
            # Get product link with multiple selector attempts
            product_link = ""
            link_selectors = [
                "h2 a",
                "h2.a-size-mini a",
                ".a-link-normal.s-no-outline",
                ".a-link-normal.a-text-normal",
                "a.a-link-normal[href*='/dp/']",
                "a[href*='/dp/']"
            ]
            
            for selector in link_selectors:
                try:
                    link_element = item.find_element(By.CSS_SELECTOR, selector)
                    product_link = link_element.get_attribute("href")
                    if product_link and '/dp/' in product_link:
                        print(f"    Found link with selector '{selector}': {product_link[:60]}...")
                        break
                except NoSuchElementException:
                    continue
            
            if not product_link:
                print(f"    Failed to extract link: No valid link found with any selector")
                continue
            
            # Get product name with multiple selector attempts
            name = ""
            name_selectors = [
                "h2 span",
                "h2 a span",
                "h2.a-size-mini span.a-size-medium",
                "h2.a-size-mini span.a-size-base-plus",
                "span.a-size-medium.a-color-base.a-text-normal",
                "span.a-size-base-plus"
            ]
            
            for selector in name_selectors:
                try:
                    name_element = item.find_element(By.CSS_SELECTOR, selector)
                    name = name_element.text.strip()
                    if name and len(name) > 3:
                        print(f"    Found name with selector '{selector}': {name[:60]}")
                        break
                except NoSuchElementException:
                    continue
            
            if not name:
                # Final fallback - try to get name from the link element
                try:
                    link_element = item.find_element(By.CSS_SELECTOR, f"a[href='{product_link}']")
                    name = link_element.get_attribute("aria-label") or link_element.text.strip()
                    print(f"    Found name from link aria-label/text: {name[:60]}")
                except:
                    print(f"    Failed to extract name: No valid name found")
                    continue
            
            if not product_link or not name or len(name) < 3:
                print(f"    Skipping - invalid link or name")
                continue
            
            # Get basic info from search results
            price_text = "0"
            try:
                price_element = item.find_element(By.CSS_SELECTOR, ".a-price-whole")
                price_text = price_element.text.strip()
            except:
                pass
            
            image_url = ""
            try:
                img_element = item.find_element(By.CSS_SELECTOR, "img.s-image")
                image_url = img_element.get_attribute("src")
            except:
                pass
            
            rating = ""
            try:
                rating_element = item.find_element(By.CSS_SELECTOR, "span.a-icon-alt")
                rating = rating_element.text.strip()
            except:
                pass
            
            reviews = ""
            try:
                reviews_element = item.find_element(By.CSS_SELECTOR, "span.a-size-base.s-underline-text")
                reviews = reviews_element.text.strip()
            except:
                pass
            
            # Open product in new tab
            print(f"    Opening product page in new tab...")
            driver.execute_script("window.open(arguments[0]);", product_link)
            time.sleep(3)
            driver.switch_to.window(driver.window_handles[1])
            print(f"    Switched to new tab, extracting details...")
            
            # Scrape detailed info
            detailed_info = scrape_amazon_product_details(driver, None)
            
            # Close tab and switch back
            driver.close()
            driver.switch_to.window(driver.window_handles[0])
            print(f"    Closed tab, back to search results")
            time.sleep(1)
            
            product_data = {
                "name": name,
                "price": price_text,
                "price_numeric": clean_price(price_text),
                "rating": rating,
                "reviews": reviews,
                "image_url": image_url,
                "category": categorize_product(name),
                "source": "Amazon.in",
                "product_link": product_link,
                "availability": "In Stock"
            }
            
            # Add detailed info
            if detailed_info:
                product_data.update(detailed_info)
            
            products.append(product_data)
            print(f"    ✓ Successfully scraped product {idx+1}")
            
        except Exception as e:
            print(f"  ✗ Error on Amazon product {idx+1}: {e}")
            import traceback
            traceback.print_exc()
            # Close any extra tabs and return to main window
            try:
                while len(driver.window_handles) > 1:
                    driver.switch_to.window(driver.window_handles[-1])
                    driver.close()
                driver.switch_to.window(driver.window_handles[0])
            except:
                pass
            continue
    
    print(f"Amazon: Successfully scraped {len(products)} products with details")
    return products
def scrape_detailed_flipkart(driver, product_name, max_products=5):
    """Enhanced Flipkart scraper with deep product details"""
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
            print(f"Flipkart: Found {len(items)} items, scraping details for top {max_products}...")
            break
    
    for idx, item in enumerate(items[:max_products]):
        try:
            print(f"  Scraping Flipkart product {idx+1}/{max_products}...")
            
            # Get product link with multiple selector attempts
            product_link = ""
            link_selectors = [
                "a[href*='/p/']",
                "a[href*='/product/']",
                "a.s1Q9rs",
                "a._1fQZEK",
                "a._2rpwqI"
            ]
            
            for selector in link_selectors:
                try:
                    link_element = item.find_element(By.CSS_SELECTOR, selector)
                    href = link_element.get_attribute("href")
                    if href and ('/p/' in href or '/product/' in href) and 'search' not in href:
                        product_link = href if href.startswith('http') else f"https://www.flipkart.com{href}"
                        print(f"    Found link with selector '{selector}': {product_link[:60]}...")
                        break
                except NoSuchElementException:
                    continue
            
            if not product_link:
                print(f"    Failed to extract link: No valid product link found")
                continue
            
            # Get product name with multiple selector attempts
            name = ""
            name_selectors = [
                "._4rR01T",
                ".IRpwTa",
                "._2WkVRV",
                ".KzDlHZ",
                ".s1Q9rs",
                ".wjcEIp",
                "a.s1Q9rs",
                "div._4rR01T"
            ]
            
            for selector in name_selectors:
                try:
                    name_element = item.find_element(By.CSS_SELECTOR, selector)
                    name = name_element.text.strip()
                    if name and len(name) > 3:
                        print(f"    Found name with selector '{selector}': {name[:60]}")
                        break
                except NoSuchElementException:
                    continue
            
            if not name:
                print(f"    Failed to extract name: No valid name found")
                continue

            # --- NON-ESSENTIAL INFO ---
            # Price
            price_text = "0"
            try:
                price_element = item.find_element(By.CSS_SELECTOR, "._30jeq3, ._1_WHN1, .Nx9bqj")
                price_text = price_element.text.strip()
            except NoSuchElementException:
                pass # Optional
            
            # Image
            image_url = ""
            try:
                img_element = item.find_element(By.CSS_SELECTOR, "img")
                image_url = img_element.get_attribute("src")
            except NoSuchElementException:
                pass # Optional
            
            # Rating
            rating = ""
            try:
                rating_element = item.find_element(By.CSS_SELECTOR, "div[class*='_3LWZlK']")
                rating = rating_element.text.strip()
            except NoSuchElementException:
                pass # Optional
            
            # Reviews
            reviews = ""
            try:
                reviews_element = item.find_element(By.CSS_SELECTOR, "span[class*='_2_R_DZ']")
                reviews = reviews_element.text.strip()
            except NoSuchElementException:
                pass # Optional
            
            # Open product page in new tab (similar to Amazon approach)
            print(f"    Opening product page in new tab...")
            driver.execute_script("window.open(arguments[0]);", product_link)
            time.sleep(3)
            
            # Switch to new tab
            driver.switch_to.window(driver.window_handles[1])
            print(f"    Switched to new tab, extracting details...")
            
            # Scrape detailed info from product page
            detailed_info = scrape_flipkart_product_details(driver, None)
            
            # Close tab and switch back
            driver.close()
            driver.switch_to.window(driver.window_handles[0])
            print(f"    Closed tab, back to search results")
            time.sleep(1)
            
            product_data = {
                "name": name,
                "price": price_text,
                "price_numeric": clean_price(price_text),
                "rating": rating,
                "reviews": reviews,
                "image_url": image_url,
                "category": categorize_product(name),
                "source": "Flipkart",
                "product_link": product_link,
                "availability": "In Stock"
            }
            
            # Add detailed info
            if detailed_info:
                product_data.update(detailed_info)
            
            products.append(product_data)
            print(f"    ✓ Successfully scraped product {idx+1}")
            
        except Exception as e:
            print(f"  ✗ Error on Flipkart product {idx+1}: {e}")
            import traceback
            traceback.print_exc()
            # Close any extra tabs and return to main window
            try:
                while len(driver.window_handles) > 1:
                    driver.switch_to.window(driver.window_handles[-1])
                    driver.close()
                driver.switch_to.window(driver.window_handles[0])
            except:
                pass
            continue
    
    print(f"Flipkart: Successfully scraped {len(products)} products with details")
    return products

def categorize_product(name, subcategory=""):
    name_lower = (name + " " + subcategory).lower()
    
    shoe_keywords = ['shoe', 'sneaker', 'boot', 'sandal', 'slipper', 'footwear']
    clothing_keywords = ['shirt', 't-shirt', 'pant', 'jean', 'jacket', 'hoodie', 'dress', 'skirt', 'shorts']
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

def create_comprehensive_graphs(rag_storage):
    """Create comprehensive analysis graphs from RAG database"""
    if not rag_storage.products:
        print("No products in database to visualize")
        return
    
    df = pd.DataFrame(rag_storage.products)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    fig.suptitle('Comprehensive Product Analysis Dashboard', fontsize=20, fontweight='bold')
    
    # 1. Price Distribution by Source
    ax1 = fig.add_subplot(gs[0, 0])
    if 'source' in df.columns and 'price_numeric' in df.columns:
        df_price = df[df['price_numeric'] > 0]
        sources = df_price['source'].unique()
        for source in sources:
            source_data = df_price[df_price['source'] == source]['price_numeric']
            ax1.hist(source_data, alpha=0.6, label=source, bins=15)
        ax1.set_xlabel('Price (₹)', fontsize=10)
        ax1.set_ylabel('Frequency', fontsize=10)
        ax1.set_title('Price Distribution by Source', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # 2. Price Comparison Boxplot
    ax2 = fig.add_subplot(gs[0, 1])
    if 'source' in df.columns and 'price_numeric' in df.columns:
        df_price = df[df['price_numeric'] > 0]
        sources = df_price['source'].unique()
        data_to_plot = [df_price[df_price['source'] == s]['price_numeric'].values for s in sources]
        bp = ax2.boxplot(data_to_plot, labels=sources, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        ax2.set_ylabel('Price (₹)', fontsize=10)
        ax2.set_title('Price Range Comparison', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
    
    # 3. Category Distribution
    ax3 = fig.add_subplot(gs[0, 2])
    if 'category' in df.columns:
        category_counts = df['category'].value_counts()
        colors = plt.cm.Set3(range(len(category_counts)))
        wedges, texts, autotexts = ax3.pie(category_counts, labels=category_counts.index, 
                                            autopct='%1.1f%%', colors=colors, startangle=90)
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        ax3.set_title('Products by Category', fontsize=12, fontweight='bold')
    
    # 4. Rating Distribution
    ax4 = fig.add_subplot(gs[1, 0])
    if 'rating' in df.columns:
        ratings = []
        for rating_str in df['rating']:
            match = re.search(r'(\d+\.?\d*)', str(rating_str))
            if match:
                ratings.append(float(match.group(1)))
        
        if ratings:
            ax4.hist(ratings, bins=10, color='gold', edgecolor='black', alpha=0.7)
            ax4.set_xlabel('Rating', fontsize=10)
            ax4.set_ylabel('Frequency', fontsize=10)
            ax4.set_title('Rating Distribution', fontsize=12, fontweight='bold')
            ax4.grid(True, alpha=0.3)
            ax4.axvline(np.mean(ratings), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(ratings):.2f}')
            ax4.legend()
    
    # 5. Top 10 Products by Price
    ax5 = fig.add_subplot(gs[1, 1])
    if 'price_numeric' in df.columns and 'name' in df.columns:
        df_sorted = df[df['price_numeric'] > 0].nsmallest(10, 'price_numeric')
        y_pos = range(len(df_sorted))
        # Truncate long names
        names = [name[:30] + '...' if len(name) > 30 else name for name in df_sorted['name']]
        ax5.barh(y_pos, df_sorted['price_numeric'], color='green', alpha=0.7)
        ax5.set_yticks(y_pos)
        ax5.set_yticklabels(names, fontsize=8)
        ax5.set_xlabel('Price (₹)', fontsize=10)
        ax5.set_title('Top 10 Lowest Priced Products', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3)
    
    # 6. Price vs Rating Scatter
    ax6 = fig.add_subplot(gs[1, 2])
    if 'price_numeric' in df.columns and 'rating' in df.columns:
        prices = []
        ratings = []
        for _, row in df.iterrows():
            price = row.get('price_numeric', 0)
            rating_str = row.get('rating', '')
            match = re.search(r'(\d+\.?\d*)', str(rating_str))
            if price > 0 and match:
                prices.append(price)
                ratings.append(float(match.group(1)))
        
        if prices and ratings:
            ax6.scatter(prices, ratings, alpha=0.6, c='purple', s=50)
            ax6.set_xlabel('Price (₹)', fontsize=10)
            ax6.set_ylabel('Rating', fontsize=10)
            ax6.set_title('Price vs Rating', fontsize=12, fontweight='bold')
            ax6.grid(True, alpha=0.3)
    
    # 7. Products by Source
    ax7 = fig.add_subplot(gs[2, 0])
    if 'source' in df.columns:
        source_counts = df['source'].value_counts()
        ax7.bar(source_counts.index, source_counts.values, color=['#FF6B6B', '#4ECDC4'], alpha=0.8)
        ax7.set_xlabel('Source', fontsize=10)
        ax7.set_ylabel('Number of Products', fontsize=10)
        ax7.set_title('Products by Source', fontsize=12, fontweight='bold')
        ax7.grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(source_counts.values):
            ax7.text(i, v + 0.5, str(v), ha='center', va='bottom', fontweight='bold')
    
    # 8. Average Price by Source
    ax8 = fig.add_subplot(gs[2, 1])
    if 'source' in df.columns and 'price_numeric' in df.columns:
        df_price = df[df['price_numeric'] > 0]
        avg_prices = df_price.groupby('source')['price_numeric'].mean()
        ax8.bar(avg_prices.index, avg_prices.values, color=['#FFD93D', '#6BCB77'], alpha=0.8)
        ax8.set_xlabel('Source', fontsize=10)
        ax8.set_ylabel('Average Price (₹)', fontsize=10)
        ax8.set_title('Average Price by Source', fontsize=12, fontweight='bold')
        ax8.grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(avg_prices.values):
            ax8.text(i, v + 500, f'₹{v:.0f}', ha='center', va='bottom', fontweight='bold')
    
    # 9. Statistics Summary
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    stats = rag_storage.get_statistics()
    
    stats_text = "DATABASE STATISTICS\n\n"
    stats_text += f"Total Products: {stats.get('total_products', 0)}\n"
    stats_text += f"Detailed Products: {stats.get('detailed_products', 0)}\n\n"
    
    if stats.get('price_stats'):
        ps = stats['price_stats']
        stats_text += "PRICE STATISTICS\n"
        stats_text += f"Min: ₹{ps['min']:.2f}\n"
        stats_text += f"Max: ₹{ps['max']:.2f}\n"
        stats_text += f"Avg: ₹{ps['avg']:.2f}\n"
        stats_text += f"Median: ₹{ps['median']:.2f}\n\n"
    
    if stats.get('rating_stats'):
        rs = stats['rating_stats']
        stats_text += "RATING STATISTICS\n"
        stats_text += f"Min: {rs['min']:.1f}⭐\n"
        stats_text += f"Max: {rs['max']:.1f}⭐\n"
        stats_text += f"Avg: {rs['avg']:.2f}⭐\n"
    
    ax9.text(0.1, 0.9, stats_text, transform=ax9.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             family='monospace')
    
    plt.savefig('product_analysis_comprehensive.png', dpi=300, bbox_inches='tight')
    print("\n✓ Comprehensive graph saved as 'product_analysis_comprehensive.png'")
    plt.show()


def create_detailed_report(rag_storage):
    """Create detailed statistics report"""
    stats = rag_storage.get_statistics()
    
    if not stats:
        print("No statistics available")
        return
    
    print("\n" + "="*70)
    print("COMPREHENSIVE PRODUCT DATABASE REPORT".center(70))
    print("="*70)
    print(f"\nTotal Products: {stats['total_products']}")
    print(f"Products with Full Details: {stats.get('detailed_products', 0)}")
    
    print("\n" + "-"*70)
    print("PRODUCTS BY SOURCE")
    print("-"*70)
    for source, count in stats['by_source'].items():
        percentage = (count / stats['total_products']) * 100
        print(f"  {source:20} : {count:4} products ({percentage:.1f}%)")
    
    print("\n" + "-"*70)
    print("PRODUCTS BY CATEGORY")
    print("-"*70)
    for category, count in stats['by_category'].items():
        percentage = (count / stats['total_products']) * 100
        print(f"  {category:20} : {count:4} products ({percentage:.1f}%)")
    
    if stats.get('price_stats'):
        print("\n" + "-"*70)
        print("PRICE STATISTICS")
        print("-"*70)
        ps = stats['price_stats']
        print(f"  Minimum Price      : ₹{ps['min']:,.2f}")
        print(f"  Maximum Price      : ₹{ps['max']:,.2f}")
        print(f"  Average Price      : ₹{ps['avg']:,.2f}")
        print(f"  Median Price       : ₹{ps['median']:,.2f}")
        print(f"  Price Range        : ₹{ps['max'] - ps['min']:,.2f}")
    
    if stats.get('rating_stats'):
        print("\n" + "-"*70)
        print("RATING STATISTICS")
        print("-"*70)
        rs = stats['rating_stats']
        print(f"  Minimum Rating     : {rs['min']:.1f}⭐")
        print(f"  Maximum Rating     : {rs['max']:.1f}⭐")
        print(f"  Average Rating     : {rs['avg']:.2f}⭐")
        print(f"  Median Rating      : {rs['median']:.2f}⭐")
    
    print("\n" + "="*70 + "\n")


# ===========================
# Enhanced GUI
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
        if img:
            img_label.config(image=img, text="", bg="white")
            img_label.image = img
    except:
        pass


def display_results_gui_with_details(df, rag_storage):
    """Enhanced GUI showing detailed product information"""
    root = tk.Tk()
    root.title("Product Price Comparison - Detailed View")
    root.geometry("1400x800")
    
    # Main frame with scrollbar
    main_frame = ttk.Frame(root)
    main_frame.pack(fill=tk.BOTH, expand=1)
    
    canvas = tk.Canvas(main_frame)
    scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=canvas.yview)
    scrollable_frame = ttk.Frame(canvas)
    
    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )
    
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    
    # Header with price range
    if not df.empty and 'price_numeric' in df.columns:
        min_price = df['price_numeric'].min()
        max_price = df['price_numeric'].max()
        header_text = f"Product Comparison Results - Price Range: ₹{min_price:.0f} - ₹{max_price:.0f}"
    else:
        header_text = "Product Comparison Results"
    
    header = tk.Label(scrollable_frame, text=header_text, font=("Arial", 18, "bold"), pady=10, bg="#2c3e50", fg="white")
    header.pack(fill=tk.X)
    
    # Highlight lowest price
    if not df.empty and 'price_numeric' in df.columns:
        lowest_price = df['price_numeric'].min()
        lowest_info = tk.Label(scrollable_frame, 
                              text=f"✓ LOWEST PRICE: ₹{lowest_price:.0f}", 
                              font=("Arial", 16, "bold"), fg="white", bg="green", pady=8)
        lowest_info.pack(fill=tk.X)
    
    # Display products sorted by price
    for idx, row in df.iterrows():
        # Check if this is the lowest price
        is_lowest = False
        if 'price_numeric' in df.columns:
            is_lowest = (row['price_numeric'] == df['price_numeric'].min())
        
        # Frame color - highlight lowest price
        if is_lowest:
            product_frame = tk.Frame(scrollable_frame, relief=tk.RAISED, 
                                   borderwidth=4, padx=15, pady=15, bg="#e8f5e9")
        else:
            product_frame = tk.Frame(scrollable_frame, relief=tk.RAISED, 
                                   borderwidth=2, padx=15, pady=15, bg="white")
        
        product_frame.pack(fill=tk.X, padx=10, pady=8)
        
        # Image placeholder
        img_label = tk.Label(product_frame, text="Loading...", width=150, height=150, bg="lightgray")
        img_label.grid(row=0, column=0, rowspan=6, padx=10, sticky=tk.N)
        
        # Load image asynchronously
        if row.get('image_url'):
            Thread(target=load_image_async, args=(img_label, row['image_url']), daemon=True).start()
        
        # Product name
        name_label = tk.Label(product_frame, text=row['name'], 
                            font=("Arial", 13, "bold"), wraplength=800, justify=tk.LEFT,
                            bg=product_frame['bg'])
        name_label.grid(row=0, column=1, sticky=tk.W, padx=10, pady=3)
        
        # Price with lowest indicator
        if is_lowest:
            price_text = f"Price: ₹{row['price_numeric']:.0f} ⭐ LOWEST PRICE ⭐"
            price_color = "darkgreen"
            price_font = ("Arial", 16, "bold")
        else:
            price_text = f"Price: ₹{row['price_numeric']:.0f}"
            price_color = "green"
            price_font = ("Arial", 15, "bold")
        
        price_label = tk.Label(product_frame, text=price_text, 
                             font=price_font, fg=price_color, bg=product_frame['bg'])
        price_label.grid(row=1, column=1, sticky=tk.W, padx=10, pady=3)
        
        # Basic info line
        info_parts = [f"Source: {row['source']}", f"Category: {row['category']}"]
        if row.get('rating'):
            info_parts.append(f"Rating: {row['rating']}")
        if row.get('reviews'):
            info_parts.append(f"Reviews: {row['reviews']}")
        
        info_text = " | ".join(info_parts)
        info_label = tk.Label(product_frame, text=info_text, font=("Arial", 10),
                            bg=product_frame['bg'], fg="#555")
        info_label.grid(row=2, column=1, sticky=tk.W, padx=10, pady=3)
        
        # Description/Features
        if row.get('description'):
            desc_text = row['description'][:200] + "..." if len(row.get('description', '')) > 200 else row.get('description', '')
            desc_label = tk.Label(product_frame, text=f"Description: {desc_text}", 
                                font=("Arial", 9), wraplength=800, justify=tk.LEFT,
                                bg=product_frame['bg'], fg="#333")
            desc_label.grid(row=3, column=1, sticky=tk.W, padx=10, pady=3)
        
        # Technical Details Button
        if row.get('technical_details') or row.get('additional_info'):
            def show_details(product_row=row):
                details_window = tk.Toplevel(root)
                details_window.title(f"Details - {product_row['name'][:50]}")
                details_window.geometry("800x600")
                
                text_widget = scrolledtext.ScrolledText(details_window, wrap=tk.WORD, font=("Courier", 9))
                text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
                
                details_text = f"PRODUCT: {product_row['name']}\n"
                details_text += f"PRICE: ₹{product_row['price_numeric']:.0f}\n"
                details_text += f"SOURCE: {product_row['source']}\n"
                details_text += "="*80 + "\n\n"
                
                if product_row.get('technical_details'):
                    details_text += "TECHNICAL DETAILS:\n" + "-"*80 + "\n"
                    for key, value in product_row['technical_details'].items():
                        details_text += f"{key:40} : {value}\n"
                    details_text += "\n"
                
                if product_row.get('additional_info'):
                    details_text += "ADDITIONAL INFORMATION:\n" + "-"*80 + "\n"
                    for key, value in product_row['additional_info'].items():
                        details_text += f"{key:40} : {value}\n"
                    details_text += "\n"
                
                if product_row.get('features'):
                    details_text += "FEATURES:\n" + "-"*80 + "\n"
                    for feat in product_row['features']:
                        details_text += f"• {feat}\n"
                
                text_widget.insert(tk.END, details_text)
                text_widget.config(state=tk.DISABLED)
            
            details_btn = tk.Button(product_frame, text="View Full Details", 
                                   command=show_details, bg="#3498db", fg="white",
                                   font=("Arial", 9, "bold"), cursor="hand2")
            details_btn.grid(row=4, column=1, sticky=tk.W, padx=10, pady=5)
        
        # Product link
        if row.get('product_link'):
            link_label = tk.Label(product_frame, text="🔗 View on Website", 
                                font=("Arial", 9, "underline"), fg="blue",
                                bg=product_frame['bg'], cursor="hand2")
            link_label.grid(row=5, column=1, sticky=tk.W, padx=10, pady=3)
            # Note: In real implementation, you'd use webbrowser.open()
    
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    # Bottom buttons
    button_frame = ttk.Frame(root)
    button_frame.pack(fill=tk.X, padx=10, pady=10)
    
    def show_graphs():
        create_comprehensive_graphs(rag_storage)
    
    graph_btn = tk.Button(button_frame, text="📊 Show Analysis Graphs", 
                         command=show_graphs, bg="#2ecc71", fg="white",
                         font=("Arial", 11, "bold"), padx=20, pady=8)
    graph_btn.pack(side=tk.LEFT, padx=5)
    
    def export_data():
        rag_storage.export_to_csv('products_detailed_export.csv')
        tk.messagebox.showinfo("Export", "Data exported to products_detailed_export.csv")
    
    export_btn = tk.Button(button_frame, text="💾 Export to CSV", 
                          command=export_data, bg="#9b59b6", fg="white",
                          font=("Arial", 11, "bold"), padx=20, pady=8)
    export_btn.pack(side=tk.LEFT, padx=5)
    
    root.mainloop()


# ===========================
# Main Agent Function
# ===========================

def smart_product_agent(product_name, rag_storage, max_products=5):
    """Main agent that scrapes detailed info, stores in RAG, and analyzes"""
    print(f"\n{'='*70}")
    print(f"SMART PRODUCT AGENT - DEEP SCRAPER".center(70))
    print(f"{'='*70}")
    print(f"Search Query: {product_name}")
    print(f"Max Products per Source: {max_products}")
    print(f"{'='*70}\n")
    
    driver = setup_driver()
    
    try:
        print("STEP 1: Deep Scraping Amazon with Full Product Details...")
        print("-"*70)
        amazon_products = scrape_detailed_amazon(driver, product_name, max_products)
        
        print("\nSTEP 2: Deep Scraping Flipkart with Full Product Details...")
        print("-"*70)
        flipkart_products = scrape_detailed_flipkart(driver, product_name, max_products)
        
    finally:
        driver.quit()

    all_products = amazon_products + flipkart_products
    
    # Filter for phone-specific searches
    if any(term in product_name.lower() for term in ['phone', 'mobile', 'smartphone', 'iphone']):
        before_count = len(all_products)
        all_products = filter_only_phones(all_products, product_name)
        after_count = len(all_products)
        print(f"\nFiltered out {before_count - after_count} non-phone accessories")

    if not all_products:
        print("\n❌ No products found after filtering.")
        return None

    print(f"\n{'='*70}")
    print("STEP 3: Storing Products in RAG Database...")
    print("-"*70)
    rag_storage.add_products_batch(all_products)
    print(f"✓ Successfully stored {len(all_products)} products with full details")
    
    print(f"\n{'='*70}")
    print("STEP 4: Generating Comprehensive Analysis Report...")
    print("-"*70)
    create_detailed_report(rag_storage)
    
    df = pd.DataFrame(all_products)
    df = df.sort_values(by="price_numeric")
    
    print(f"{'='*70}")
    print("AGENT EXECUTION COMPLETED SUCCESSFULLY!".center(70))
    print(f"{'='*70}")
    print(f"\n📊 Total Products Found: {len(df)}")
    if not df.empty and 'price_numeric' in df.columns:
        print(f"💰 Lowest Price: ₹{df['price_numeric'].min():.0f}")
        print(f"💰 Highest Price: ₹{df['price_numeric'].max():.0f}")
    detailed_count = sum(1 for p in all_products if p.get('technical_details') or p.get('additional_info'))
    print(f"📝 Products with Full Details: {detailed_count}/{len(df)}")
    print(f"\n{'='*70}\n")
    
    return df


# ===========================
# Main Execution
# ===========================

if __name__ == "__main__":
    # Initialize RAG storage
    rag_storage = ProductRAGStorage('product_rag_database.pkl')
    
    print("\n" + "="*70)
    print("SMART PRODUCT COMPARISON AGENT WITH DEEP SCRAPING & RAG".center(70))
    print("="*70)
    print("\nOptions:")
    print("1. Search new products (Deep Scrape with Full Details)")
    print("2. Search in existing database (Semantic Search)")
    print("3. View database statistics")
    print("4. Show analysis graphs")
    print("5. Export database to CSV")
    print("6. Clear database")
    print("7. Exit")
    
    while True:
        choice = input("\nEnter choice (1-7): ").strip()
        
        if choice == "1":
            product_name = input("\nEnter product name to search: ").strip()
            if product_name:
                max_products = input("How many products per source? (default: 5): ").strip()
                max_products = int(max_products) if max_products.isdigit() else 5
                
                result_df = smart_product_agent(product_name, rag_storage, max_products)
                if result_df is not None and not result_df.empty:
                    print("\n🖥️  Opening GUI with detailed product information...")
                    display_results_gui_with_details(result_df, rag_storage)
                else:
                    print("\n❌ No products found.")
        
        elif choice == "2":
            query = input("\nEnter search query: ").strip()
            if query:
                results = rag_storage.semantic_search(query, top_k=20)
                if results:
                    result_df = pd.DataFrame(results)
                    result_df = result_df.sort_values(by="price_numeric")
                    print(f"\n✓ Found {len(result_df)} matching products")
                    display_results_gui_with_details(result_df, rag_storage)
                else:
                    print("\n❌ No matching products found in database.")
        
        elif choice == "3":
            create_detailed_report(rag_storage)
        
        elif choice == "4":
            create_comprehensive_graphs(rag_storage)
        
        elif choice == "5":
            filename = input("Enter filename (default: products_detailed_export.csv): ").strip()
            if not filename:
                filename = "products_detailed_export.csv"
            rag_storage.export_to_csv(filename)
        
        elif choice == "6":
            confirm = input("⚠️  Are you sure you want to clear all data? (yes/no): ").strip().lower()
            if confirm == "yes":
                rag_storage.clear_storage()
                print("✓ Database cleared successfully.")
        
        elif choice == "7":
            print("\n👋 Exiting... Goodbye!")
            break
        
        else:
            print("❌ Invalid choice. Please enter 1-7.")