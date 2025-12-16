from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager

import pandas as pd
import time
import random
import re
import tkinter as tk
from tkinter import ttk, scrolledtext
from PIL import Image, ImageTk
import requests
from io import BytesIO
from threading import Thread
import webbrowser

# ==========================================================
# GLOBAL CONFIG
# ==========================================================

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
]


def setup_driver():
    """Create a Selenium Chrome driver with anti-bot tweaks."""
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument(f"user-agent={random.choice(USER_AGENTS)}")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_experimental_option(
        "excludeSwitches", ["enable-automation", "enable-logging"]
    )
    chrome_options.add_experimental_option("useAutomationExtension", False)
    chrome_options.add_experimental_option(
        "prefs", {"profile.default_content_setting_values.notifications": 2}
    )

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=chrome_options,
    )

    # Avoid basic bot detection
    try:
        driver.execute_cdp_cmd(
            "Network.setUserAgentOverride",
            {"userAgent": random.choice(USER_AGENTS)},
        )
    except Exception:
        pass

    try:
        driver.execute_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
        )
    except Exception:
        pass

    driver.set_page_load_timeout(60)
    return driver


def categorize_product(name, subcategory=""):
    """Very simple category tagging just for display."""
    text = (name + " " + subcategory).lower()

    shoe_keywords = [
        "shoe",
        "sneaker",
        "boot",
        "sandal",
        "slipper",
        "footwear",
        "nike air",
        "jordan",
        "running",
    ]
    clothing_keywords = [
        "shirt",
        "t-shirt",
        "pant",
        "jean",
        "jacket",
        "hoodie",
        "dress",
        "skirt",
        "shorts",
        "clothing",
        "apparel",
    ]

    if any(k in text for k in shoe_keywords):
        return "Shoes"
    if any(k in text for k in clothing_keywords):
        return "Clothing"
    return "Other"


# ==========================================================
# PRODUCT DETAIL SCRAPERS
# ==========================================================

def scrape_amazon_product_details(driver, product_url):
    """Scrape detailed information from Amazon product page - enhanced version"""
    try:
        if product_url:
            driver.get(product_url)
            time.sleep(random.uniform(3, 5))
        
        details = {
            'technical_details': {},
            'additional_info': {},
            'description': '',
            'features': [],
            'customer_reviews': [],
            'review_summary': '',
            'rating_breakdown': {}
        }
        
        print(f"      Searching for Amazon product details...")
        
        # ========== CUSTOMER REVIEWS EXTRACTION ==========
        print(f"      Extracting customer reviews...")
        
        # Get rating breakdown (5 star: 61%, 4 star: 22%, etc.)
        try:
            rating_table = driver.find_element(By.ID, "histogramTable")
            rating_rows = rating_table.find_elements(By.CSS_SELECTOR, "tr.a-histogram-row")
            for row in rating_rows:
                try:
                    star_text = row.find_element(By.CSS_SELECTOR, "td.aok-nowrap span").text.strip()
                    percent_text = row.find_element(By.CSS_SELECTOR, "td.a-text-right span").text.strip()
                    if star_text and percent_text:
                        details['rating_breakdown'][star_text] = percent_text
                except:
                    continue
            print(f"      Found rating breakdown: {details['rating_breakdown']}")
        except:
            # Alternative method
            try:
                histogram = driver.find_elements(By.CSS_SELECTOR, "#averageCustomerReviews .a-histogram-row, [data-hook='rating-histogram'] tr")
                for row in histogram:
                    text = row.text.strip()
                    if 'star' in text.lower() and '%' in text:
                        parts = text.split()
                        for i, p in enumerate(parts):
                            if 'star' in p.lower() and i > 0:
                                star = parts[i-1] + ' star'
                                for pct in parts:
                                    if '%' in pct:
                                        details['rating_breakdown'][star] = pct
                                        break
            except:
                pass
        
        # Get "Customers say" summary
        try:
            customers_say = driver.find_element(By.CSS_SELECTOR, "[data-hook='cr-summarization-attributes-list'], #cr-summarization-attributes-list")
            details['review_summary'] = customers_say.text.strip()
            print(f"      Found 'Customers say' summary")
        except:
            try:
                # Alternative: Look for insights section
                insights = driver.find_element(By.CSS_SELECTOR, ".cr-widget-FocalReviews, [data-hook='lighthut-terms-list']")
                details['review_summary'] = insights.text.strip()
            except:
                pass
        
        # Get top customer reviews
        try:
            review_elements = driver.find_elements(By.CSS_SELECTOR, "[data-hook='review'], .review, .a-section.review")[:5]
            for review in review_elements:
                try:
                    review_data = {}
                    
                    # Review title
                    try:
                        title = review.find_element(By.CSS_SELECTOR, "[data-hook='review-title'], .review-title").text.strip()
                        review_data['title'] = title
                    except:
                        pass
                    
                    # Review text
                    try:
                        body = review.find_element(By.CSS_SELECTOR, "[data-hook='review-body'], .review-text, .review-text-content").text.strip()
                        review_data['text'] = body
                    except:
                        pass
                    
                    # Review rating
                    try:
                        rating = review.find_element(By.CSS_SELECTOR, "[data-hook='review-star-rating'], .review-rating").text.strip()
                        review_data['rating'] = rating
                    except:
                        pass
                    
                    if review_data.get('text'):
                        details['customer_reviews'].append(review_data)
                except:
                    continue
            print(f"      Found {len(details['customer_reviews'])} customer reviews")
        except:
            pass
        
        # Alternative: Get reviews from review section
        if not details['customer_reviews']:
            try:
                review_texts = driver.find_elements(By.CSS_SELECTOR, ".reviewText, [data-hook='review-body'] span")[:5]
                for rt in review_texts:
                    text = rt.text.strip()
                    if text and len(text) > 20:
                        details['customer_reviews'].append({'text': text})
            except:
                pass
        
        # ========== ORIGINAL DETAIL EXTRACTION ==========
        # Keywords to search for in page elements
        detail_keywords = ['detail', 'product information', 'specification', 'technical', 'product details']
        
        # Method 1: Technical Details table by ID
        try:
            tech_table = driver.find_element(By.ID, "productDetails_techSpec_section_1")
            rows = tech_table.find_elements(By.TAG_NAME, "tr")
            print(f"      Found Technical Details table with {len(rows)} rows")
            for row in rows:
                try:
                    th = row.find_element(By.TAG_NAME, "th").text.strip()
                    td = row.find_element(By.TAG_NAME, "td").text.strip()
                    if th and td:
                        details['technical_details'][th] = td
                except:
                    continue
        except:
            print(f"      Technical Details table not found by ID, trying alternatives...")
        
        # Method 2: Search all tables for keyword matches
        if not details['technical_details']:
            try:
                all_tables = driver.find_elements(By.TAG_NAME, "table")
                print(f"      Found {len(all_tables)} tables, searching for details...")
                
                for table in all_tables:
                    try:
                        table_text = table.text.lower()
                        if any(keyword in table_text for keyword in detail_keywords):
                            rows = table.find_elements(By.TAG_NAME, "tr")
                            for row in rows:
                                try:
                                    cells = row.find_elements(By.TAG_NAME, "th")
                                    cells.extend(row.find_elements(By.TAG_NAME, "td"))
                                    if len(cells) >= 2:
                                        key = cells[0].text.strip()
                                        value = cells[1].text.strip()
                                        if key and value and len(key) < 100:
                                            details['technical_details'][key] = value
                                except:
                                    continue
                    except:
                        continue
            except:
                pass
        
        # Additional Information section
        try:
            additional_table = driver.find_element(By.ID, "productDetails_detailBullets_sections1")
            rows = additional_table.find_elements(By.TAG_NAME, "tr")
            print(f"      Found Additional Information with {len(rows)} rows")
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
        
        # Detail bullets (common format on Amazon)
        if not details['technical_details']:
            try:
                bullets = driver.find_elements(By.CSS_SELECTOR, "#detailBullets_feature_div li")
                for bullet in bullets:
                    text = bullet.text.strip()
                    if ':' in text:
                        parts = text.split(':', 1)
                        if len(parts) == 2:
                            key = parts[0].strip()
                            value = parts[1].strip()
                            if key and value:
                                details['technical_details'][key] = value
            except:
                pass
        
        # Product Features (bullet points)
        desc_found = False
        try:
            desc_element = driver.find_element(By.ID, "feature-bullets")
            features = desc_element.find_elements(By.TAG_NAME, "li")
            print(f"      Found {len(features)} product features")
            for feat in features:
                text = feat.text.strip()
                if text and len(text) > 5:
                    details['features'].append(text)
            if details['features']:
                details['description'] = ' | '.join(details['features'])
                desc_found = True
        except:
            pass
        
        # Alternative description from productDescription div
        if not desc_found:
            try:
                desc = driver.find_element(By.ID, "productDescription")
                details['description'] = desc.text.strip()
                print(f"      Found product description")
                desc_found = True
            except:
                pass
        
        # Method 3: Search divs for description keywords
        if not desc_found:
            try:
                all_divs = driver.find_elements(By.TAG_NAME, "div")
                for div in all_divs:
                    try:
                        div_class = div.get_attribute("class") or ""
                        div_id = div.get_attribute("id") or ""
                        
                        if any(kw in div_class.lower() or kw in div_id.lower() 
                               for kw in ['description', 'detail', 'product-info']):
                            text = div.text.strip()
                            if text and len(text) > 50 and len(text) < 2000:
                                details['description'] = text
                                print(f"      Found description in div: {div_id or div_class[:30]}")
                                break
                    except:
                        continue
            except:
                pass
        
        # Fallback: aplus description
        if not details['description']:
            try:
                aplus = driver.find_element(By.ID, "aplus")
                details['description'] = aplus.text.strip()[:500]
            except:
                pass
        
        print(f"      Amazon extraction complete: {len(details['technical_details'])} specs, "
              f"{len(details['additional_info'])} additional info, {len(details['features'])} features")
        
        return details
        
    except Exception as e:
        print(f"Error scraping Amazon details: {e}")
        import traceback
        traceback.print_exc()
        return None


def scrape_flipkart_product_details(driver, product_url):
    """Scrape detailed information from Flipkart product page - enhanced version"""
    try:
        if product_url:
            driver.get(product_url)
            time.sleep(random.uniform(3, 5))
        
        details = {
            'technical_details': {},
            'additional_info': {},
            'description': '',
            'features': [],
            'page_price': '',  # Price from product page
            'customer_reviews': [],
            'review_summary': '',
            'rating_breakdown': {}
        }
        
        print(f"      Searching for Flipkart product details...")
        
        # ========== CUSTOMER REVIEWS EXTRACTION ==========
        print(f"      Extracting Flipkart customer reviews...")
        
        # Get rating breakdown
        try:
            rating_bars = driver.find_elements(By.CSS_SELECTOR, "._5nfVE5, .BArk-j, ._1uJVNT")
            for bar in rating_bars:
                text = bar.text.strip()
                if 'star' in text.lower() or any(c.isdigit() for c in text):
                    # Parse "5 ★ 12,345" or similar formats
                    parts = text.split()
                    if len(parts) >= 2:
                        star = parts[0] + ' star'
                        count = parts[-1] if len(parts) > 1 else ''
                        if count:
                            details['rating_breakdown'][star] = count
        except:
            pass
        
        # Get review highlights/summary
        try:
            highlights = driver.find_elements(By.CSS_SELECTOR, "._3LsR7f, ._2o9D_1, .t-ZTKy")
            summary_parts = []
            for h in highlights[:5]:
                text = h.text.strip()
                if text and len(text) > 5:
                    summary_parts.append(text)
            if summary_parts:
                details['review_summary'] = ' | '.join(summary_parts)
                print(f"      Found review highlights")
        except:
            pass
        
        # Get customer reviews
        try:
            review_elements = driver.find_elements(By.CSS_SELECTOR, "div.ZmyHeo, div._27M-vq, div.t-ZTKy, div._6K-7Co")[:5]
            for review in review_elements:
                try:
                    review_data = {}
                    
                    # Review title
                    try:
                        title = review.find_element(By.CSS_SELECTOR, "p._2-N8zT, ._2sc7ZR").text.strip()
                        review_data['title'] = title
                    except:
                        pass
                    
                    # Review text
                    try:
                        body = review.find_element(By.CSS_SELECTOR, "div.t-ZTKy, div._6K-7Co, div.ZmyHeo").text.strip()
                        review_data['text'] = body
                    except:
                        try:
                            body = review.text.strip()
                            if len(body) > 20:
                                review_data['text'] = body
                        except:
                            pass
                    
                    # Review rating
                    try:
                        rating = review.find_element(By.CSS_SELECTOR, "div._3LWZlK, div._3LWZlK._1BLPMq").text.strip()
                        review_data['rating'] = rating
                    except:
                        pass
                    
                    if review_data.get('text') and len(review_data['text']) > 10:
                        details['customer_reviews'].append(review_data)
                except:
                    continue
            print(f"      Found {len(details['customer_reviews'])} customer reviews")
        except:
            pass
        
        # Alternative: Look for review text in common containers
        if not details['customer_reviews']:
            try:
                review_containers = driver.find_elements(By.CSS_SELECTOR, "[class*='review'], [class*='Review']")[:10]
                for container in review_containers:
                    text = container.text.strip()
                    if text and len(text) > 30 and len(text) < 1000:
                        details['customer_reviews'].append({'text': text})
                        if len(details['customer_reviews']) >= 5:
                            break
            except:
                pass
        
        # ========== ORIGINAL EXTRACTION ==========
        # Try to get price from product page (useful when search page doesn't show price)
        price_selectors = [
            "div.Nx9bqj._4b5DiR",  # Main price on product page
            "div._30jeq3._16Jk6d",  # Alternative price
            "div.Nx9bqj",
            "div._30jeq3",
            "div._16Jk6d"
        ]
        
        for sel in price_selectors:
            try:
                price_elem = driver.find_element(By.CSS_SELECTOR, sel)
                price_text = price_elem.text.strip()
                if price_text and '₹' in price_text:
                    details['page_price'] = price_text
                    print(f"      Found price on product page: {price_text}")
                    break
            except:
                continue
        
        # Keywords to search for specifications
        spec_keywords = ['specification', 'specifications', 'specs']
        desc_keywords = ['description', 'product description', 'about']
        detail_keywords = ['product detail', 'product details', 'details', 'product information']
        
        # Method 1: Find elements by searching text content for keywords
        try:
            all_headings = driver.find_elements(By.CSS_SELECTOR, "div, span, h1, h2, h3")
            spec_sections = []
            
            for heading in all_headings:
                try:
                    text = heading.text.strip().lower()
                    if any(keyword in text for keyword in spec_keywords + detail_keywords):
                        parent = heading
                        for _ in range(3):
                            try:
                                parent = parent.find_element(By.XPATH, "..")
                                rows = parent.find_elements(By.CSS_SELECTOR, "tr, li")
                                if rows:
                                    spec_sections.append(parent)
                                    print(f"      Found section by keyword '{text[:30]}': {len(rows)} rows")
                                    break
                            except:
                                break
                except:
                    continue
            
            # Extract from found sections
            for section in spec_sections:
                try:
                    section_title = "Specifications"
                    try:
                        title_elem = section.find_element(By.CSS_SELECTOR, "div._4BJ2V\\+, div._1AtVbE, div._3dtsli, span, h3")
                        section_title = title_elem.text.strip() or "Specifications"
                    except:
                        pass
                    
                    rows = section.find_elements(By.CSS_SELECTOR, "tr")
                    for row in rows:
                        try:
                            cells = row.find_elements(By.CSS_SELECTOR, "td")
                            if len(cells) >= 2:
                                key = cells[0].text.strip()
                                value = cells[1].text.strip()
                                if key and value and len(key) < 100:
                                    details['technical_details'][f"{section_title} - {key}"] = value
                        except:
                            continue
                except:
                    continue
        except Exception as e:
            print(f"      Error in keyword search: {e}")
        
        # Method 2: Try multiple CSS selectors for specification sections
        if not details['technical_details']:
            spec_section_selectors = [
                "div._9V0sS6", "div.GNDEQ-", "div._1s76Cw", "div._3dtsli",
                "div.aMraIH", "div[class*='spec']", "div[class*='detail']"
            ]
            
            for selector in spec_section_selectors:
                try:
                    sections = driver.find_elements(By.CSS_SELECTOR, selector)
                    if sections:
                        print(f"      Found {len(sections)} spec sections using selector: {selector}")
                        for section_idx, section in enumerate(sections):
                            section_title = "Specifications"
                            title_selectors = ["div._4BJ2V\\+", "div._1AtVbE", "div._3dtsli", "div._2RngUh", "span"]
                            
                            for title_sel in title_selectors:
                                try:
                                    title_elem = section.find_element(By.CSS_SELECTOR, title_sel)
                                    section_title = title_elem.text.strip()
                                    if section_title and len(section_title) > 0 and len(section_title) < 50:
                                        break
                                except:
                                    continue
                            
                            if not section_title:
                                section_title = f"Section {section_idx + 1}"
                            
                            row_selectors = ["tr.WJdYP6", "tr._2-N8s", "li.W5FkOm", "div.row", "tr"]
                            for row_sel in row_selectors:
                                rows = section.find_elements(By.CSS_SELECTOR, row_sel)
                                if rows:
                                    print(f"        Found {len(rows)} rows using selector: {row_sel}")
                                    for row in rows:
                                        try:
                                            cell_selectors = ["td.URwL2w", "td._7eSDEY", "td._2H-kL", "td._2vIOIi", "td", "li._21Ahn-"]
                                            cells = []
                                            for cell_sel in cell_selectors:
                                                try:
                                                    found_cells = row.find_elements(By.CSS_SELECTOR, cell_sel)
                                                    if found_cells:
                                                        cells.extend(found_cells)
                                                        break
                                                except:
                                                    continue
                                            
                                            if len(cells) >= 2:
                                                key = cells[0].text.strip()
                                                value = cells[1].text.strip()
                                                if key and value and len(key) < 100:
                                                    details['technical_details'][f"{section_title} - {key}"] = value
                                            elif len(cells) == 1:
                                                text = cells[0].text.strip()
                                                if text and len(text) > 3:
                                                    details['features'].append(text)
                                        except:
                                            continue
                                    break
                        if details['technical_details']:
                            break
                except:
                    continue
        
        # Method 3: Generic table search for phone specs
        if not details['technical_details']:
            try:
                tables = driver.find_elements(By.TAG_NAME, "table")
                for table in tables:
                    table_text = table.text.lower()
                    if any(kw in table_text for kw in ['ram', 'storage', 'display', 'battery', 'processor', 'camera', 'screen']):
                        rows = table.find_elements(By.TAG_NAME, "tr")
                        for row in rows:
                            try:
                                cells = row.find_elements(By.TAG_NAME, "td")
                                if len(cells) >= 2:
                                    key = cells[0].text.strip()
                                    value = cells[1].text.strip()
                                    if key and value and len(key) < 100:
                                        details['technical_details'][key] = value
                            except:
                                continue
                        if details['technical_details']:
                            break
            except:
                pass
        
        # Features/Highlights
        highlight_selectors = [
            "li._21Ahn-", "li.WJdYP6", "ul._1D2qrc li", 
            "div._2418kt ul li", "li[class*='highlight']", "li[class*='feature']"
        ]
        
        for sel in highlight_selectors:
            try:
                highlights = driver.find_elements(By.CSS_SELECTOR, sel)
                if highlights:
                    for h in highlights:
                        text = h.text.strip()
                        if text and len(text) > 5 and text not in details['features']:
                            details['features'].append(text)
                    if details['features']:
                        print(f"      Found {len(details['features'])} features using: {sel}")
                        break
            except:
                continue
        
        # Description
        desc_selectors = [
            "div._1mXcCf", "div._2418kt", "div.yN\\+eNk", 
            "div._2RngUh p", "div.product-description", "div[class*='description']", "div[class*='desc']"
        ]
        
        for sel in desc_selectors:
            try:
                desc_el = driver.find_element(By.CSS_SELECTOR, sel)
                text = desc_el.text.strip()
                if text and len(text) > 20:
                    details['description'] = text[:1000]
                    print(f"      Found description using: {sel}")
                    break
            except:
                continue
        
        # If no description found, search by text
        if not details['description']:
            try:
                all_divs = driver.find_elements(By.TAG_NAME, "div")
                for div in all_divs:
                    try:
                        div_class = div.get_attribute("class") or ""
                        if any(kw in div_class.lower() for kw in desc_keywords):
                            text = div.text.strip()
                            if text and len(text) > 50 and len(text) < 2000:
                                details['description'] = text
                                print(f"      Found description by keyword in: {div_class[:30]}")
                                break
                    except:
                        continue
            except:
                pass
        
        if not details['description'] and details['features']:
            details['description'] = ' | '.join(details['features'])
        
        print(f"      Flipkart extraction complete: {len(details['technical_details'])} specs, {len(details['features'])} features")
        
        return details
        
    except Exception as e:
        print(f"Error scraping Flipkart details: {e}")
        import traceback
        traceback.print_exc()
        return None


# ==========================================================
# SCRAPERS
# ==========================================================

def scrape_amazon_in(driver, product_name, max_products=5, fetch_details=True):
    """
    Scrape Amazon.in search results.
    
    Args:
        fetch_details: If True, open each product page for detailed info (slower).
                      If False, only scrape search page data (faster).
    """
    try:
        print("Opening Amazon...")
        driver.get("https://www.amazon.in")
        time.sleep(random.uniform(4, 6))

        url = f"https://www.amazon.in/s?k={product_name.replace(' ', '+')}"
        print("Amazon search URL:", url)
        driver.get(url)
        time.sleep(random.uniform(5, 7))

        WebDriverWait(driver, 35).until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, "[data-component-type='s-search-result']")
            )
        )
        time.sleep(3)
    except TimeoutException:
        print("Amazon: results took too long to load.")
        return []
    except Exception as e:
        print(f"Amazon error: {e}")
        return []

    products = []
    items = driver.find_elements(
        By.CSS_SELECTOR, "[data-component-type='s-search-result']"
    )

    print(f"Amazon: located {len(items)} search result containers, scraping top {max_products}...")

    # Skip top 2 results (usually ads), then take max_products
    items = items[2:2+max_products]

    for item in items:
        try:
            # Product Link
            product_link = ""
            for sel in ["h2 a", "a.a-link-normal[href*='/dp/']", "a[href*='/dp/']"]:
                try:
                    link_el = item.find_element(By.CSS_SELECTOR, sel)
                    href = link_el.get_attribute("href")
                    if href and '/dp/' in href:
                        product_link = href
                        break
                except NoSuchElementException:
                    continue

            # Name
            name_el = None
            for sel in ["h2 a span", "h2 span"]:
                try:
                    name_el = item.find_element(By.CSS_SELECTOR, sel)
                    break
                except NoSuchElementException:
                    continue
            if not name_el:
                continue
            name = name_el.text.strip()
            if not name:
                continue

            # Price
            price_el = None
            for sel in [".a-price-whole", ".a-price .a-offscreen"]:
                try:
                    price_el = item.find_element(By.CSS_SELECTOR, sel)
                    break
                except NoSuchElementException:
                    continue
            if not price_el:
                continue
            price_text = price_el.text.strip()
            if not price_text:
                continue

            # Image
            image_url = ""
            try:
                img_el = item.find_element(By.CSS_SELECTOR, "img.s-image")
                image_url = img_el.get_attribute("src")
            except Exception:
                pass

            # Rating
            rating = ""
            try:
                r_el = item.find_element(By.CSS_SELECTOR, ".a-icon-alt")
                rating = r_el.get_attribute("textContent").strip()
            except Exception:
                pass

            # Reviews
            reviews = ""
            try:
                rev_el = item.find_element(
                    By.CSS_SELECTOR, ".a-size-base.s-underline-text"
                )
                reviews = rev_el.text.strip()
            except Exception:
                pass

            # Subcategory
            subcategory = ""
            try:
                sub_el = item.find_element(
                    By.CSS_SELECTOR, ".a-size-base-plus.a-color-base"
                )
                subcategory = sub_el.text.strip()
            except Exception:
                pass

            category = categorize_product(name, subcategory)

            product_data = {
                "name": name,
                "subcategory": subcategory,
                "price": price_text,
                "rating": rating,
                "reviews": reviews,
                "image_url": image_url,
                "category": category,
                "source": "Amazon.in",
                "product_link": product_link,
                "technical_details": {},
                "features": [],
                "description": "",
            }
            
            # Fetch detailed info from product page (only if fetch_details is True)
            if fetch_details and product_link:
                try:
                    print(f"  Fetching details for: {name[:50]}...")
                    # Open product in new tab
                    driver.execute_script("window.open(arguments[0]);", product_link)
                    time.sleep(2)
                    driver.switch_to.window(driver.window_handles[1])
                    time.sleep(random.uniform(2, 4))
                    
                    # Scrape detailed info
                    detailed_info = scrape_amazon_product_details(driver, None)
                    
                    # Close tab and switch back
                    driver.close()
                    driver.switch_to.window(driver.window_handles[0])
                    time.sleep(1)
                    
                    # Add detailed info to product
                    if detailed_info:
                        product_data.update(detailed_info)
                        print(f"    ✓ Got {len(detailed_info.get('technical_details', {}))} specs, {len(detailed_info.get('features', []))} features")
                    
                except Exception as e:
                    print(f"    ✗ Error fetching details: {e}")
                    # Make sure we're back to main window
                    try:
                        while len(driver.window_handles) > 1:
                            driver.switch_to.window(driver.window_handles[-1])
                            driver.close()
                        driver.switch_to.window(driver.window_handles[0])
                    except:
                        pass
            
            products.append(product_data)

        except Exception:
            continue

    print(f"Found {len(products)} products on Amazon")
    return products


def scrape_flipkart(driver, product_name, max_products=5, fetch_details=True):
    """
    Scrape Flipkart search results.
    
    Args:
        fetch_details: If True, open each product page for detailed info (slower).
                      If False, only scrape search page data (faster).
    """
    """
    More tolerant Flipkart scraper:
    - uses sleep + multiple selector fallbacks
    - falls back to parsing text (name/price) from item.text
    """
    try:
        url = f"https://www.flipkart.com/search?q={product_name.replace(' ', '+')}"
        print("Opening Flipkart search URL:", url)
        driver.get(url)
        time.sleep(random.uniform(7, 9))

        # Close login popup if present
        try:
            close_btn = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable(
                    (By.XPATH, "//button[contains(text(), '✕')]")
                )
            )
            close_btn.click()
            time.sleep(1)
        except Exception:
            pass

        time.sleep(random.uniform(5, 7))

    except TimeoutException:
        print("Flipkart: page load timed out.")
        return []
    except Exception as e:
        print(f"Flipkart error while loading page: {e}")
        return []

    products = []
    items = []

    possible_selectors = [
        "._1AtVbE",
        "._13oc-S",
        "._tUxRFH",
        "._CGtC98",
        "div[data-id]",  # generic product tiles
    ]

    for sel in possible_selectors:
        items = driver.find_elements(By.CSS_SELECTOR, sel)
        if items:
            print(f"Flipkart: found {len(items)} containers, scraping top {max_products}...")
            break

    if not items:
        print("Flipkart: no product containers found.")
        return []

    for item in items[:max_products]:
        try:
            full_text = item.text.strip()
            if not full_text:
                continue

            # Skip sold out / unavailable products early
            full_text_lower = full_text.lower()
            if any(skip in full_text_lower for skip in ['sold out', 'currently unavailable', 'out of stock', 'coming soon']):
                print(f"  Skipping unavailable product")
                continue

            # ---------------- Product Link ----------------
            product_link = ""
            try:
                link_el = item.find_element(By.CSS_SELECTOR, "a[href*='/p/']")
                href = link_el.get_attribute("href")
                if href:
                    product_link = href if href.startswith('http') else f"https://www.flipkart.com{href}"
            except NoSuchElementException:
                pass
            
            if not product_link:
                try:
                    link_el = item.find_element(By.CSS_SELECTOR, "a[title]")
                    href = link_el.get_attribute("href")
                    if href:
                        product_link = href if href.startswith('http') else f"https://www.flipkart.com{href}"
                except NoSuchElementException:
                    pass

            # ---------------- Name ----------------
            name = ""
            
            # First, try to get title from anchor tag (most reliable)
            try:
                title_el = item.find_element(By.CSS_SELECTOR, "a[title]")
                name = title_el.get_attribute("title") or ""
            except NoSuchElementException:
                pass
            
            # Try specific class selectors
            if not name:
                for sel in ["._4rR01T", ".KzDlHZ", ".s1Q9rs", ".WKTcLC", "._2WkVRV"]:
                    try:
                        name_el = item.find_element(By.CSS_SELECTOR, sel)
                        name = name_el.text.strip()
                        if name and len(name) > 15:  # Valid product name
                            break
                    except NoSuchElementException:
                        continue
            
            # Fallback: parse from full text, skip junk lines
            if not name or len(name) < 15:
                lines = [l.strip() for l in full_text.split("\n") if l.strip()]
                for line in lines:
                    # Skip junk
                    if "₹" in line:
                        continue
                    if "%" in line:
                        continue
                    if "OFF" in line.upper():
                        continue
                    if "Add to Compare" in line:
                        continue
                    if "Currently unavailable" in line:
                        continue
                    # Skip very short lines (likely ratings like "4.5")
                    if len(line) < 15:
                        continue
                    name = line
                    break

            if not name or len(name) < 10:
                continue
            
            # Skip if the name is obviously not a product
            if name.lower() in ["add to compare", "currently unavailable", "out of stock"]:
                continue

            # ---------------- Price ----------------
            price_el = None
            for sel in ["._30jeq3", "._1_WHN1", "._Nx9bqj", "._4b5DiR"]:
                try:
                    price_el = item.find_element(By.CSS_SELECTOR, sel)
                    break
                except NoSuchElementException:
                    continue

            price_text = ""
            if price_el:
                price_text = price_el.text.strip()

            if not price_text:
                m = re.search(r"₹\s*[\d,]+", full_text)
                if m:
                    price_text = m.group(0)

            # Note: Don't skip here if no price - we'll try to get it from product page later

            # ---------------- Image ----------------
            image_url = ""
            try:
                img_el = item.find_element(By.CSS_SELECTOR, "img")
                image_url = img_el.get_attribute("src")
            except Exception:
                pass

            # ---------------- Rating ----------------
            rating = ""
            try:
                r_el = item.find_element(By.CSS_SELECTOR, "div[class*='_3LWZlK']")
                rating = r_el.text.strip()
            except Exception:
                pass

            # ---------------- Reviews ----------------
            reviews = ""
            try:
                rev_el = item.find_element(
                    By.CSS_SELECTOR, "span[class*='_2_R_DZ']"
                )
                reviews = rev_el.text.strip()
            except Exception:
                pass

            # ---------------- Subcategory / brand text ----------------
            subcategory = ""
            try:
                sub_el = item.find_element(
                    By.CSS_SELECTOR, "._2WkVRV, ._NqpwHC"
                )
                subcategory = sub_el.text.strip()
            except Exception:
                pass

            category = categorize_product(name, subcategory)

            product_data = {
                "name": name,
                "subcategory": subcategory,
                "price": price_text,
                "rating": rating,
                "reviews": reviews,
                "image_url": image_url,
                "category": category,
                "source": "Flipkart",
                "product_link": product_link,
                "technical_details": {},
                "features": [],
                "description": "",
            }
            
            # Fetch detailed info from product page (only if fetch_details is True)
            if fetch_details and product_link:
                try:
                    print(f"  Fetching details for: {name[:50]}...")
                    # Open product in new tab
                    driver.execute_script("window.open(arguments[0]);", product_link)
                    time.sleep(2)
                    driver.switch_to.window(driver.window_handles[1])
                    time.sleep(random.uniform(2, 4))
                    
                    # Scrape detailed info
                    detailed_info = scrape_flipkart_product_details(driver, None)
                    
                    # Close tab and switch back
                    driver.close()
                    driver.switch_to.window(driver.window_handles[0])
                    time.sleep(1)
                    
                    # Add detailed info to product
                    if detailed_info:
                        product_data.update(detailed_info)
                        # Use page_price as fallback if no price from search page
                        if not product_data.get('price') and detailed_info.get('page_price'):
                            product_data['price'] = detailed_info['page_price']
                            print(f"    → Using price from product page: {detailed_info['page_price']}")
                        print(f"    ✓ Got {len(detailed_info.get('technical_details', {}))} specs, {len(detailed_info.get('features', []))} features")
                    
                except Exception as e:
                    print(f"    ✗ Error fetching details: {e}")
                    # Make sure we're back to main window
                    try:
                        while len(driver.window_handles) > 1:
                            driver.switch_to.window(driver.window_handles[-1])
                            driver.close()
                        driver.switch_to.window(driver.window_handles[0])
                    except:
                        pass
            
            # Skip product if still no price (only strict when not fetching details)
            if not product_data.get('price'):
                if fetch_details:
                    print(f"  Skipping product without price: {name[:50]}...")
                    continue
                # In fast mode, keep products even without price from search page
            
            products.append(product_data)

        except Exception:
            continue

    print(f"Found {len(products)} products on Flipkart")
    return products


# ==========================================================
# FILTERING + GUI HELPERS
# ==========================================================

def clean_price(price_text: str) -> float:
    if not price_text:
        return 0.0
    cleaned = re.sub(r"[₹,\s]", "", price_text)
    nums = re.findall(r"\d+\.?\d*", cleaned)
    if nums:
        try:
            return float(nums[0])
        except ValueError:
            return 0.0
    return 0.0


def load_image_from_url(url, size=(130, 130)):
    try:
        response = requests.get(url, timeout=4)
        img = Image.open(BytesIO(response.content))
        img = img.resize(size, Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(img)
    except Exception:
        return None


def load_image_async(img_label, url):
    try:
        img = load_image_from_url(url)
        if img:
            img_label.config(image=img, text="", bg="white")
            img_label.image = img
    except Exception:
        pass


def display_results_gui(df: pd.DataFrame):
    root = tk.Tk()
    root.title("Product Price Comparison")
    root.geometry("1200x800")

    main_frame = ttk.Frame(root)
    main_frame.pack(fill=tk.BOTH, expand=True)

    canvas = tk.Canvas(main_frame)
    scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=canvas.yview)
    scrollable_frame = ttk.Frame(canvas)

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    
    # Enable mouse wheel scrolling
    def on_mousewheel(event):
        canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    canvas.bind_all("<MouseWheel>", on_mousewheel)

    header = tk.Label(
        scrollable_frame,
        text="Product Price Comparison Results",
        font=("Arial", 18, "bold"),
        pady=10,
    )
    header.pack()

    for _, row in df.iterrows():
        frame = tk.Frame(
            scrollable_frame, relief=tk.RAISED, borderwidth=2, padx=10, pady=10
        )
        frame.pack(fill=tk.X, padx=10, pady=5)

        img_label = tk.Label(frame, text="Loading...", width=130, height=130, bg="lightgray")
        img_label.grid(row=0, column=0, rowspan=6, padx=10)

        if row.get("image_url"):
            Thread(
                target=load_image_async, args=(img_label, row["image_url"]),
            ).start()

        name_label = tk.Label(
            frame,
            text=row["name"],
            font=("Arial", 12, "bold"),
            wraplength=600,
            justify=tk.LEFT,
        )
        name_label.grid(row=0, column=1, sticky=tk.W, padx=10)

        if row.get("subcategory"):
            subcat_label = tk.Label(
                frame,
                text=row["subcategory"],
                font=("Arial", 9),
                fg="gray",
            )
            subcat_label.grid(row=1, column=1, sticky=tk.W, padx=10)

        price_label = tk.Label(
            frame,
            text=f"Price: ₹{row['price_numeric']:.0f}",
            font=("Arial", 14, "bold"),
            fg="#27ae60",
        )
        price_label.grid(row=2, column=1, sticky=tk.W, padx=10)

        info = f"Source: {row['source']} | Category: {row['category']}"
        if row.get("rating"):
            info += f" | Rating: {row['rating']}"
        if row.get("reviews"):
            info += f" | Reviews: {row['reviews']}"

        info_label = tk.Label(frame, text=info, font=("Arial", 9))
        info_label.grid(row=3, column=1, sticky=tk.W, padx=10)
        
        next_row = 4
        
        # View Details button
        product_row = row.to_dict()
        
        def show_details(product_data=product_row):
            details_window = tk.Toplevel(root)
            details_window.title(f"Product Details - {product_data['name'][:50]}...")
            details_window.geometry("700x500")
            
            text_widget = scrolledtext.ScrolledText(details_window, wrap=tk.WORD, 
                                                     font=("Consolas", 10))
            text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            details_text = "="*70 + "\n"
            details_text += f"📱 {product_data['name']}\n"
            details_text += "="*70 + "\n\n"
            
            details_text += f"💰 Price: ₹{product_data['price_numeric']:.0f}\n"
            details_text += f"🏪 Source: {product_data['source']}\n"
            details_text += f"📂 Category: {product_data['category']}\n"
            
            if product_data.get('rating'):
                details_text += f"⭐ Rating: {product_data['rating']}\n"
            if product_data.get('reviews'):
                details_text += f"💬 Reviews: {product_data['reviews']}\n"
            
            details_text += "\n"
            
            # Rating Breakdown (5 star: 61%, etc.)
            if product_data.get('rating_breakdown'):
                details_text += "⭐ RATING BREAKDOWN:\n" + "-"*50 + "\n"
                for star, percent in product_data['rating_breakdown'].items():
                    details_text += f"  {star}: {percent}\n"
                details_text += "\n"
            
            # Customer Reviews
            if product_data.get('customer_reviews'):
                reviews = product_data['customer_reviews']
                details_text += f"💬 CUSTOMER REVIEWS ({len(reviews)} shown):\n" + "-"*50 + "\n"
                for idx, review in enumerate(reviews[:5], 1):
                    if isinstance(review, dict):
                        if review.get('rating'):
                            details_text += f"  ⭐ {review['rating']}\n"
                        if review.get('title'):
                            details_text += f"  📌 {review['title']}\n"
                        if review.get('text'):
                            review_text = review['text'][:300] + "..." if len(review['text']) > 300 else review['text']
                            details_text += f"  {review_text}\n"
                    else:
                        details_text += f"  {str(review)[:300]}\n"
                    details_text += "\n"
            
            # Review Summary (Customers say...)
            if product_data.get('review_summary'):
                details_text += "📊 CUSTOMERS SAY:\n" + "-"*50 + "\n"
                details_text += f"  {product_data['review_summary']}\n\n"
            
            # Technical Details
            if product_data.get('technical_details'):
                details_text += "📋 TECHNICAL DETAILS:\n" + "-"*50 + "\n"
                for key, value in product_data['technical_details'].items():
                    details_text += f"  {key}: {value}\n"
                details_text += "\n"
            
            # Features
            if product_data.get('features') and isinstance(product_data['features'], list):
                if len(product_data['features']) > 0:
                    details_text += "✨ FEATURES:\n" + "-"*50 + "\n"
                    for idx, feat in enumerate(product_data['features'], 1):
                        details_text += f"  {idx}. {feat}\n"
                    details_text += "\n"
            
            # Description
            if product_data.get('description'):
                details_text += "📝 DESCRIPTION:\n" + "-"*50 + "\n"
                details_text += product_data['description'] + "\n\n"
            
            # Product Link
            if product_data.get('product_link'):
                details_text += "🔗 PRODUCT LINK:\n" + "-"*50 + "\n"
                details_text += product_data['product_link'] + "\n"
            
            details_text += "\n" + "="*70 + "\n"
            
            text_widget.insert(tk.END, details_text)
            text_widget.config(state=tk.DISABLED)
            
            # Buttons frame
            btn_frame = tk.Frame(details_window)
            btn_frame.pack(pady=5)
            
            if product_data.get('product_link'):
                open_btn = tk.Button(btn_frame, text="🌐 Open in Browser", 
                                    command=lambda: webbrowser.open(product_data['product_link']),
                                    bg="#3498db", fg="white", font=("Arial", 10, "bold"),
                                    padx=15, pady=5)
                open_btn.pack(side=tk.LEFT, padx=5)
            
            def copy_to_clipboard():
                details_window.clipboard_clear()
                details_window.clipboard_append(details_text)
            
            copy_btn = tk.Button(btn_frame, text="📋 Copy Details", 
                                command=copy_to_clipboard, bg="#95a5a6", fg="white",
                                font=("Arial", 10, "bold"), padx=15, pady=5)
            copy_btn.pack(side=tk.LEFT, padx=5)
        
        details_btn = tk.Button(frame, text="📄 View Details", 
                               command=show_details, bg="#3498db", fg="white",
                               font=("Arial", 9, "bold"), cursor="hand2", padx=10, pady=3)
        details_btn.grid(row=5, column=1, sticky=tk.W, padx=10, pady=5)
        
        # Product link - clickable
        if row.get('product_link'):
            link_label = tk.Label(frame, text="🔗 View on Website", 
                                font=("Arial", 9, "underline"), fg="blue",
                                cursor="hand2")
            link_label.grid(row=6, column=1, sticky=tk.W, padx=10, pady=3)
            link_label.bind("<Button-1>", lambda e, url=row['product_link']: webbrowser.open(url))

    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    root.mainloop()


# ==========================================================
# AGENTS
# ==========================================================

class BrowserAgent:
    """Manages a single Selenium driver (one per site/thread)."""

    def __init__(self):
        self.driver = None

    def start(self):
        self.driver = setup_driver()
        return self.driver

    def stop(self):
        if self.driver:
            self.driver.quit()
            self.driver = None


class AmazonAgent:
    def __init__(self, browser_agent: BrowserAgent):
        self.browser_agent = browser_agent

    def search(self, product_name: str, max_products: int = 5, fetch_details: bool = True):
        return scrape_amazon_in(self.browser_agent.driver, product_name, max_products, fetch_details)


class FlipkartAgent:
    def __init__(self, browser_agent: BrowserAgent):
        self.browser_agent = browser_agent

    def search(self, product_name: str, max_products: int = 5, fetch_details: bool = True):
        return scrape_flipkart(self.browser_agent.driver, product_name, max_products, fetch_details)


class FilterAgent:
    def __init__(self):
        # for phone queries
        self.phone_related_keywords = {
            "phone", "mobile", "smartphone", "iphone", "galaxy",
            "pixel", "oneplus", "redmi", "realme", "vivo", "oppo", "poco"
        }
        self.generic_terms = {"mobile", "phone", "phones", "smartphone", "smartphones", "cell"}
        self.fallback_include = {"phone", "mobile", "smartphone", "iphone"}
        self.exclude_keywords = [
            "case ",
            "cases ",
            " case",
            "cover ",
            "covers ",
            " cover",
            "back cover",
            "bumper",
            "protective case",
            "protective cover",
            "charger",
            "charging cable",
            "cable for",
            "adapter",
            "tempered glass",
            "tempered ",
            "screen guard",
            "screen protector",
            "glass protector",
            "protector for",
            "pouch",
            "strap",
            "phone stand",
            "mobile stand",
            "skin for",
            "phone holder",
            "car holder",
            "battery pack",
            "lens protector",
            "camera protector",
            "camera lens",
            "powerbank",
            "power bank",
            "data cable",
            "ring holder",
            "popsocket",
            "screen film",
            "keyboard for",
            "car mount",
            "phone mount",
            "charging dock",
        ]

        # for headphone queries
        self.headphone_words = {"headphone", "headphones", "headset", "headsets"}
        self.ear_exclude_for_headphones = {
            "earphone", "earphones",
            "earbud", "earbuds",
            "ear pod", "ear pods", "earpod", "earpods",
            "neckband", "neck band",
            "tws", "true wireless",
            "in-ear", "in ear", "ear stick", "earsticks"
        }

        # generic search stopwords – not used as brand/model tokens
        self.generic_search_words = {
            "with", "for", "and", "or", "the", "a", "an",
            "wireless", "bluetooth", "over", "on", "ear", "ears",
            "mic", "microphone", "bass", "deep", "extra",
            "black", "white", "blue", "red", "green", "grey", "gray",
            "playtime", "hours", "upto", "up", "to", "noise", "cancelling",
            "wired", "without", "type", "c", "charging", "fast",
            "pro", "max", "plus", "ultra", "edition", "series"
        }

    # ---------- QUERY TYPE CHECKS ----------

    def is_phone_query(self, search_term: str) -> bool:
        """Check if the query is about a phone."""
        words = re.findall(r"\b\w+\b", search_term.lower())
        return any(word in self.phone_related_keywords for word in words)

    def is_headphone_query(self, search_term: str) -> bool:
        """Check if the query is about headphones / headsets."""
        words = re.findall(r"\b\w+\b", search_term.lower())
        return any(word in self.headphone_words for word in words)

    # ---------- HELPER: important tokens ----------

    def extract_important_tokens(self, search_term: str):
        tokens = re.findall(r"\b\w+\b", search_term.lower())
        important = [
            t for t in tokens
            if t not in self.generic_search_words
            and t not in self.headphone_words
            and t not in self.generic_terms
            and (len(t) > 2 or t.isdigit())
        ]
        return important

    # ---------- PHONE FILTERING ----------

    def clean_price(self, price_text: str) -> float:
        return clean_price(price_text)

    def filter_only_phones(self, products, search_term):
        if not products:
            return products

        tokens = [
            t
            for t in re.split(r"\W+", search_term.lower())
            if t and t not in self.generic_terms
        ]
        
        # Separate brand tokens from model tokens (numbers)
        brand_tokens = [t for t in tokens if not t.isdigit()]
        
        filtered = []
        for p in products:
            title = p.get("name", "").lower()
            if not title:
                continue

            if any(ex in title for ex in self.exclude_keywords):
                continue

            # For phone queries, require brand match but be flexible on model number
            if brand_tokens:
                # At least one brand token must match
                if any(tok in title for tok in brand_tokens):
                    filtered.append(p)
            elif tokens:
                # If only numbers in search, use original logic
                if any(tok in title for tok in tokens):
                    filtered.append(p)
            else:
                if any(f in title for f in self.fallback_include):
                    filtered.append(p)

        return filtered

    # ---------- HEADPHONE FILTERING ----------

    def filter_only_headphones(self, products):
        """Keep only true headphones / headsets."""
        if not products:
            return products

        filtered = []
        for p in products:
            title = (p.get("name", "") + " " + p.get("subcategory", "")).lower()
            if not title:
                continue

            if not any(w in title for w in self.headphone_words):
                continue

            if any(ex in title for ex in self.ear_exclude_for_headphones):
                continue

            filtered.append(p)

        return filtered

    # ---------- SEARCH-TERM RELEVANCE FILTER ----------

    def verify_by_search_term(self, products, search_term):
        """
        Second-stage verification:
        - Extract brand/model tokens from search term
        - For phone queries, be more flexible (match brand, not model number)
        - If such tokens exist, keep only products whose title contains
          the important non-numeric tokens (removes unrelated 'advertised' items).
        - If no important tokens (e.g. search is just 'headphones'), return as-is.
        """
        if not products:
            return products

        important = self.extract_important_tokens(search_term)
        if not important:
            return products

        # Separate brand/text tokens from numeric tokens (model numbers)
        brand_tokens = [t for t in important if not t.isdigit()]
        
        # If no brand tokens, return as-is (search was just numbers)
        if not brand_tokens:
            return products

        filtered = []
        for p in products:
            text = (p.get("name", "") + " " + p.get("subcategory", "")).lower()
            # Require all brand tokens to match (but not model numbers)
            if all(tok in text for tok in brand_tokens):
                filtered.append(p)

        return filtered if filtered else products

    # ---------- DATAFRAME BUILD ----------

    def build_dataframe(self, products):
        if not products:
            return pd.DataFrame()

        df = pd.DataFrame(products)
        if "price" not in df.columns:
            return pd.DataFrame()

        df["price_numeric"] = df["price"].apply(self.clean_price)
        df = df[df["price_numeric"] > 0]
        if df.empty:
            return df

        return df.sort_values(by="price_numeric")


class GUIAgent:
    @staticmethod
    def show(df: pd.DataFrame):
        display_results_gui(df)


# ==========================================================
# COORDINATOR – MULTI-AGENT WORKFLOW
# ==========================================================

def multi_agent_compare_prices(product_name: str, max_products: int = 5, fetch_details: bool = True):
    """
    Multi-agent price comparison with parallel scraping.
    
    Args:
        product_name: Product to search for
        max_products: Maximum products per site
        fetch_details: If True, scrape detailed product info (slower). 
                      If False, only scrape search page (faster).
    """
    amazon_browser = BrowserAgent()
    flipkart_browser = BrowserAgent()

    amazon_browser.start()
    flipkart_browser.start()

    amazon_agent = AmazonAgent(amazon_browser)
    flipkart_agent = FlipkartAgent(flipkart_browser)
    filter_agent = FilterAgent()
    gui_agent = GUIAgent()

    amazon_products = []
    flipkart_products = []

    def run_amazon():
        nonlocal amazon_products
        amazon_products = amazon_agent.search(product_name, max_products, fetch_details)

    def run_flipkart():
        nonlocal flipkart_products
        flipkart_products = flipkart_agent.search(product_name, max_products, fetch_details)

    t1 = Thread(target=run_amazon)
    t2 = Thread(target=run_flipkart)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    amazon_browser.stop()
    flipkart_browser.stop()

    print(f"Amazon products scraped: {len(amazon_products)}")
    print(f"Flipkart products scraped: {len(flipkart_products)}")

    all_products = amazon_products + flipkart_products
    if not all_products:
        print("No products scraped from either site.")
        return

    # Type-based filters
    if filter_agent.is_phone_query(product_name):
        all_products = filter_agent.filter_only_phones(all_products, product_name)
        if not all_products:
            print("No phone-like products found after filtering.")
            return
    elif filter_agent.is_headphone_query(product_name):
        all_products = filter_agent.filter_only_headphones(all_products)
        if not all_products:
            print("No headphone-only products found after filtering.")
            return

    # Verification filter against search term (removes off-brand ads)
    all_products = filter_agent.verify_by_search_term(all_products, product_name)

    df = filter_agent.build_dataframe(all_products)
    if df.empty:
        print("No products with valid prices found.")
        return

    print(f"\nFound {len(df)} products after cleaning")
    print(
        f"Price range: ₹{df['price_numeric'].min():.0f} - ₹{df['price_numeric'].max():.0f}"
    )
    print("\nOpening GUI...")
    gui_agent.show(df)


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":
    product_name = input("Enter product name to search: ")
    max_products_input = input("How many products per source? (default: 5): ").strip()
    max_products = int(max_products_input) if max_products_input.isdigit() else 5
    
    fetch_details_input = input("Fetch detailed product info? (y/n, default: y): ").strip().lower()
    fetch_details = fetch_details_input != 'n'
    
    if not fetch_details:
        print("⚡ Fast mode: Skipping detailed product page scraping for faster results")
    else:
        print("📋 Full mode: Will scrape detailed specs from product pages (slower)")
    
    multi_agent_compare_prices(product_name, max_products, fetch_details)
