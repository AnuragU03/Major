"""
Comprehensive Comparison Test: Single Agent vs Multi-Agent E-Commerce Scraper

This script performs a complete comparison test for the Major Project Results section.
It collects detailed metrics on time, bandwidth, accuracy, and energy efficiency.

Usage:
    python comparison_test.py

Output:
    - scraped_single_agent.pkl
    - scraped_multi_agent.pkl
"""

import pickle
import time
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys
from threading import Thread
from typing import Dict, List, Any, Optional, Tuple

# Test Configuration - 10 Categories with 4 queries each
CATEGORIES = [
    ("Smartphones", ["iPhone 15", "Samsung Galaxy S24", "OnePlus 12", "Pixel 8"]),
    ("Laptops", ["Dell Inspiron", "HP Pavilion", "Lenovo ThinkPad", "MacBook Air"]),
    ("Smartwatches", ["Apple Watch Series 9", "Samsung Galaxy Watch", "Noise ColorFit", "Boat Storm"]),
    ("Tablets", ["iPad", "Samsung Galaxy Tab", "Lenovo Tab", "Redmi Pad"]),
    ("Wireless Earbuds", ["AirPods Pro", "Samsung Galaxy Buds", "Boat Airdopes", "Noise Buds"]),
    ("Headphones", ["Sony WH-1000XM5", "Bose QuietComfort", "JBL Tune", "Sennheiser"]),
    ("Smart TVs", ["Samsung Smart TV 55", "LG OLED TV", "Sony Bravia", "Mi TV"]),
    ("Cameras", ["Canon EOS", "Nikon D", "Sony Alpha", "GoPro Hero"]),
    ("Gaming Consoles", ["PlayStation 5", "Xbox Series X", "Nintendo Switch", "Gaming Laptop"]),
    ("Smart Speakers", ["Amazon Echo", "Google Nest", "Apple HomePod", "JBL Flip"]),
]

PRODUCTS_PER_QUERY = 1
TOTAL_QUERIES = len(CATEGORIES) * 4

# India electricity carbon factor
CO2_FACTOR_INDIA = 0.820  # kg CO2/kWh

print("\n" + "=" * 70)
print("   COMPREHENSIVE COMPARISON TEST")
print("   Single Agent (Try.py) vs Multi-Agent (multi_agent_scraper.py)")
print("=" * 70)
print(f"\n📋 Configuration:")
print(f"   • Categories: {len(CATEGORIES)}")
print(f"   • Queries/Category: 4")
print(f"   • Total Queries: {TOTAL_QUERIES}")
print(f"\n⏱️  Estimated Time: 2-4 hours (both methods)")

# Import analyzers
try:
    from umap_rag_analyzer import UMAPAnalyzer, UMAP_AVAILABLE
    UMAP_ANALYZER_AVAILABLE = UMAP_AVAILABLE
except ImportError:
    UMAP_ANALYZER_AVAILABLE = False
    print("⚠️ UMAP analyzer not available")

try:
    from power_monitor import PowerMonitor
    POWER_MONITOR_AVAILABLE = True
except ImportError:
    POWER_MONITOR_AVAILABLE = False
    print("⚠️ Power monitor not available")


class ComprehensiveMetrics:
    """Collect comprehensive metrics for comparison"""
    
    def __init__(self, method_name: str):
        self.method = method_name
        self.start_time = None
        self.end_time = None
        self.queries = []
        self.products_per_source = {'Amazon': 0, 'Flipkart': 0, 'Croma': 0, 'Reliance Digital': 0}
        self.products_per_category = {}
        self.errors = 0
        self.error_details = []
        self.power_data = {}
        self.umap_data = {}
        
    def start(self):
        self.start_time = datetime.now()
        
    def end(self):
        self.end_time = datetime.now()
        
    def add_query_result(self, query: str, category: str, products: List[Dict], 
                         query_time: float, sources: Dict[str, int]):
        self.queries.append({
            'query': query,
            'category': category,
            'products': len(products),
            'time': query_time,
            'sources': sources
        })
        
        if category not in self.products_per_category:
            self.products_per_category[category] = 0
        self.products_per_category[category] += len(products)
        
        for source, count in sources.items():
            if source in self.products_per_source:
                self.products_per_source[source] += count
                
    def add_error(self, query: str, error: str):
        self.errors += 1
        self.error_details.append({'query': query, 'error': str(error)})
        
    def get_summary(self) -> Dict:
        total_time = (self.end_time - self.start_time).total_seconds() if self.end_time else 0
        total_products = sum(self.products_per_source.values())
        query_times = [q['time'] for q in self.queries if q['time'] > 0]
        
        return {
            'method': self.method,
            'total_products': total_products,
            'total_time': total_time,
            'total_time_minutes': total_time / 60,
            'total_queries': len(self.queries),
            'avg_time_per_query': np.mean(query_times) if query_times else 0,
            'avg_products_per_query': total_products / max(1, len(self.queries)),
            'throughput_products_per_second': total_products / max(1, total_time),
            'avg_time_per_product': total_time / max(1, total_products),
            'products_per_source': self.products_per_source,
            'products_per_category': self.products_per_category,
            'errors': self.errors,
            'success_rate': (1 - self.errors / max(1, len(self.queries))) * 100,
            'queries': self.queries,
            'power': self.power_data,
            'umap': self.umap_data,
            'latency_mean': np.mean(query_times) if query_times else 0,
            'latency_median': np.median(query_times) if query_times else 0,
            'latency_min': np.min(query_times) if query_times else 0,
            'latency_max': np.max(query_times) if query_times else 0,
            'latency_std': np.std(query_times) if query_times else 0,
            'latency_p95': np.percentile(query_times, 95) if query_times else 0,
            'latency_p99': np.percentile(query_times, 99) if query_times else 0,
            'latency_variance': np.var(query_times) if query_times else 0,
        }


def run_single_agent_test() -> Tuple[List[Dict], Dict]:
    """Run comprehensive single-agent test using Try.py"""
    print("\n" + "=" * 70)
    print("   TEST 1: SINGLE AGENT (Try.py)")
    print("=" * 70)
    
    metrics = ComprehensiveMetrics("Single Agent (Try.py)")
    
    power_monitor = None
    if POWER_MONITOR_AVAILABLE:
        power_monitor = PowerMonitor()
        power_monitor.start_monitoring()
    
    try:
        from Try import unified_rag_search, ProductRAGStorage
    except ImportError as e:
        print(f"❌ Could not import Try.py: {e}")
        return [], {}
    
    test_storage = ProductRAGStorage('test_single_agent_rag.pkl')
    all_products = []
    
    metrics.start()
    query_count = 0
    
    for category, queries in CATEGORIES:
        print(f"\n📂 Category: {category}")
        
        for query in queries:
            query_count += 1
            print(f"\n   [{query_count}/{TOTAL_QUERIES}] Searching: {query}")
            
            query_start = time.time()
            sources = {'Amazon': 0, 'Flipkart': 0, 'Croma': 0, 'Reliance Digital': 0}
            
            try:
                df = unified_rag_search(query, test_storage, max_products=PRODUCTS_PER_QUERY)
                query_time = time.time() - query_start
                
                if df is not None and not df.empty:
                    products = df.to_dict('records')
                    for p in products:
                        p['category'] = category
                        p['search_query'] = query
                        p['scrape_time'] = query_time
                        
                        source = p.get('source', 'Unknown')
                        if 'Amazon' in source:
                            sources['Amazon'] += 1
                        elif 'Flipkart' in source:
                            sources['Flipkart'] += 1
                        elif 'Croma' in source:
                            sources['Croma'] += 1
                        elif 'Reliance' in source:
                            sources['Reliance Digital'] += 1
                    
                    all_products.extend(products)
                    metrics.add_query_result(query, category, products, query_time, sources)
                    print(f"      ✓ Found {len(products)} products in {query_time:.1f}s")
                else:
                    metrics.add_query_result(query, category, [], query_time, sources)
                    print(f"      ⚠️ No products found")
                    
            except Exception as e:
                query_time = time.time() - query_start
                metrics.add_error(query, str(e))
                metrics.add_query_result(query, category, [], query_time, sources)
                print(f"      ❌ Error: {e}")
    
    metrics.end()
    
    if power_monitor:
        try:
            power_monitor.record_measurement("Single Agent Complete")
            power_report = power_monitor.generate_report()
            metrics.power_data = {
                'avg_cpu_percent': power_report.get('resource_utilization', {}).get('average_cpu_usage_percent', 0),
                'avg_cpu_power_watts': power_report.get('power_consumption', {}).get('average_cpu_power_watts', 0),
                'avg_memory_percent': power_report.get('resource_utilization', {}).get('average_memory_usage_percent', 0),
                'total_energy_kwh': power_report.get('energy_consumption', {}).get('total_energy_kwh', 0),
                'co2_emissions_grams': power_report.get('co2_emissions_grams', {}).get('india', 0),
            }
            print(f"\n⚡ Power: {metrics.power_data['avg_cpu_power_watts']:.2f}W, {metrics.power_data['total_energy_kwh']:.6f} kWh")
        except Exception as e:
            print(f"⚠️ Power monitoring error: {e}")
    
    if UMAP_ANALYZER_AVAILABLE and len(all_products) >= 10:
        try:
            print("\n🗺️ Running UMAP clustering...")
            analyzer = UMAPAnalyzer(all_products)
            analyzer.prepare_features()
            analyzer.run_umap()
            umap_metrics = analyzer.calculate_clustering_metrics()
            metrics.umap_data = {
                'silhouette_score': umap_metrics.get('silhouette_score', 0),
                'davies_bouldin_index': umap_metrics.get('davies_bouldin_index', 0),
                'cluster_purity': umap_metrics.get('cluster_purity', 0),
                'n_clusters': umap_metrics.get('n_clusters', 0)
            }
            analyzer.create_visualization('umap_single_agent.png')
            print(f"✓ UMAP saved: umap_single_agent.png")
        except Exception as e:
            print(f"⚠️ UMAP error: {e}")
    
    summary = metrics.get_summary()
    with open('scraped_single_agent.pkl', 'wb') as f:
        pickle.dump({
            'products': all_products,
            'metrics': summary,
            'scraped_at': datetime.now().isoformat()
        }, f)
    
    print(f"\n✅ Single Agent Complete!")
    print(f"   Products: {len(all_products)}")
    print(f"   Time: {summary['total_time_minutes']:.1f} min")
    print(f"   Avg/Query: {summary['avg_time_per_query']:.1f}s")
    
    return all_products, summary


def run_multi_agent_test() -> Tuple[List[Dict], Dict]:
    """Run comprehensive multi-agent test"""
    print("\n" + "=" * 70)
    print("   TEST 2: MULTI-AGENT (multi_agent_scraper.py)")
    print("=" * 70)
    
    metrics = ComprehensiveMetrics("Multi-Agent (Parallel)")
    
    power_monitor = None
    if POWER_MONITOR_AVAILABLE:
        power_monitor = PowerMonitor()
        power_monitor.start_monitoring()
    
    try:
        from multi_agent_scraper import (
            BrowserAgent, AmazonAgent, FlipkartAgent, CromaAgent, RelianceAgent
        )
    except ImportError as e:
        print(f"❌ Could not import multi_agent_scraper.py: {e}")
        return [], {}
    
    all_products = []
    metrics.start()
    query_count = 0
    
    for category, queries in CATEGORIES:
        print(f"\n📂 Category: {category}")
        
        for query in queries:
            query_count += 1
            print(f"\n   [{query_count}/{TOTAL_QUERIES}] Searching: {query}")
            
            query_start = time.time()
            sources = {'Amazon': 0, 'Flipkart': 0, 'Croma': 0, 'Reliance Digital': 0}
            
            try:
                amazon_browser = BrowserAgent()
                flipkart_browser = BrowserAgent()
                croma_browser = BrowserAgent()
                reliance_browser = BrowserAgent()
                
                amazon_browser.start()
                flipkart_browser.start()
                croma_browser.start()
                reliance_browser.start()
                
                amazon_agent = AmazonAgent(amazon_browser)
                flipkart_agent = FlipkartAgent(flipkart_browser)
                croma_agent = CromaAgent(croma_browser)
                reliance_agent = RelianceAgent(reliance_browser)
                
                amazon_products = []
                flipkart_products = []
                croma_products = []
                reliance_products = []
                
                def run_amazon():
                    nonlocal amazon_products
                    amazon_products = amazon_agent.search(query, PRODUCTS_PER_QUERY, True)
                
                def run_flipkart():
                    nonlocal flipkart_products
                    flipkart_products = flipkart_agent.search(query, PRODUCTS_PER_QUERY, True)
                
                def run_croma():
                    nonlocal croma_products
                    croma_products = croma_agent.search(query, PRODUCTS_PER_QUERY, True)
                
                def run_reliance():
                    nonlocal reliance_products
                    reliance_products = reliance_agent.search(query, PRODUCTS_PER_QUERY, True)
                
                threads = [
                    Thread(target=run_amazon),
                    Thread(target=run_flipkart),
                    Thread(target=run_croma),
                    Thread(target=run_reliance)
                ]
                
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()
                
                amazon_browser.stop()
                flipkart_browser.stop()
                croma_browser.stop()
                reliance_browser.stop()
                
                query_time = time.time() - query_start
                products = amazon_products + flipkart_products + croma_products + reliance_products
                
                for p in products:
                    p['category'] = category
                    p['search_query'] = query
                    p['scrape_time'] = query_time
                
                sources['Amazon'] = len(amazon_products)
                sources['Flipkart'] = len(flipkart_products)
                sources['Croma'] = len(croma_products)
                sources['Reliance Digital'] = len(reliance_products)
                
                all_products.extend(products)
                metrics.add_query_result(query, category, products, query_time, sources)
                
                print(f"      ✓ Found {len(products)} products in {query_time:.1f}s (parallel)")
                print(f"        A:{len(amazon_products)} F:{len(flipkart_products)} C:{len(croma_products)} R:{len(reliance_products)}")
                
            except Exception as e:
                query_time = time.time() - query_start
                metrics.add_error(query, str(e))
                metrics.add_query_result(query, category, [], query_time, sources)
                print(f"      ❌ Error: {e}")
                
                try:
                    amazon_browser.stop()
                    flipkart_browser.stop()
                    croma_browser.stop()
                    reliance_browser.stop()
                except:
                    pass
    
    metrics.end()
    
    if power_monitor:
        try:
            power_monitor.record_measurement("Multi Agent Complete")
            power_report = power_monitor.generate_report()
            metrics.power_data = {
                'avg_cpu_percent': power_report.get('resource_utilization', {}).get('average_cpu_usage_percent', 0),
                'avg_cpu_power_watts': power_report.get('power_consumption', {}).get('average_cpu_power_watts', 0),
                'avg_memory_percent': power_report.get('resource_utilization', {}).get('average_memory_usage_percent', 0),
                'total_energy_kwh': power_report.get('energy_consumption', {}).get('total_energy_kwh', 0),
                'co2_emissions_grams': power_report.get('co2_emissions_grams', {}).get('india', 0),
            }
            print(f"\n⚡ Power: {metrics.power_data['avg_cpu_power_watts']:.2f}W, {metrics.power_data['total_energy_kwh']:.6f} kWh")
        except Exception as e:
            print(f"⚠️ Power monitoring error: {e}")
    
    if UMAP_ANALYZER_AVAILABLE and len(all_products) >= 10:
        try:
            print("\n🗺️ Running UMAP clustering...")
            analyzer = UMAPAnalyzer(all_products)
            analyzer.prepare_features()
            analyzer.run_umap()
            umap_metrics = analyzer.calculate_clustering_metrics()
            metrics.umap_data = {
                'silhouette_score': umap_metrics.get('silhouette_score', 0),
                'davies_bouldin_index': umap_metrics.get('davies_bouldin_index', 0),
                'cluster_purity': umap_metrics.get('cluster_purity', 0),
                'n_clusters': umap_metrics.get('n_clusters', 0)
            }
            analyzer.create_visualization('umap_multi_agent.png')
            print(f"✓ UMAP saved: umap_multi_agent.png")
        except Exception as e:
            print(f"⚠️ UMAP error: {e}")
    
    summary = metrics.get_summary()
    with open('scraped_multi_agent.pkl', 'wb') as f:
        pickle.dump({
            'products': all_products,
            'metrics': summary,
            'scraped_at': datetime.now().isoformat()
        }, f)
    
    print(f"\n✅ Multi-Agent Complete!")
    print(f"   Products: {len(all_products)}")
    print(f"   Time: {summary['total_time_minutes']:.1f} min")
    print(f"   Avg/Query: {summary['avg_time_per_query']:.1f}s")
    
    return all_products, summary


def main():
    print(f"\n⏱️  Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    choice = input("\nSelect test mode:\n  1. Single Agent only\n  2. Multi-Agent only\n  3. Both (Full Comparison)\n\nChoice (1/2/3): ").strip()
    
    if choice in ['1', '3']:
        run_single_agent_test()
    
    if choice in ['2', '3']:
        run_multi_agent_test()
    
    print("\n" + "=" * 70)
    print("   TEST COMPLETE!")
    print("=" * 70)
    
    print(f"\n📁 Generated Files:")
    for f in ['scraped_single_agent.pkl', 'scraped_multi_agent.pkl', 
              'umap_single_agent.png', 'umap_multi_agent.png']:
        if os.path.exists(f):
            print(f"   ✓ {f}")
    
    print(f"\n📊 Run 'python analyze_pkl_results.py' for detailed Chapter 5 report")
    print(f"\n⏱️  End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
