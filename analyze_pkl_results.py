"""
Analyze test.pkl and multi.pkl results - Generate UMAP visualizations and comparison
Includes: Latency, Bandwidth, Accuracy, Website Comparison, Query Analysis
"""

import pickle
import pandas as pd
import numpy as np
import sys

# Try importing UMAP analyzer
try:
    from umap_rag_analyzer import UMAPAnalyzer, UMAP_AVAILABLE
    UMAP_ANALYZER_AVAILABLE = True
except ImportError:
    UMAP_ANALYZER_AVAILABLE = False
    UMAP_AVAILABLE = False
    print("⚠ UMAPAnalyzer not available")

def load_pickle_data(filepath):
    """Load data from pickle file"""
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        print(f"✓ Loaded: {filepath}")
        return data
    except Exception as e:
        print(f"✗ Error loading {filepath}: {e}")
        return None

def analyze_latency(metrics, name):
    """Analyze latency (time per query) in detail"""
    print(f"\n{'='*60}")
    print(f"⏱️ LATENCY ANALYSIS - {name}")
    print(f"{'='*60}")
    
    queries = metrics.get('queries', [])
    if not queries:
        print("   No query-level timing data available")
        return {}
    
    # Extract timing data
    times = [q.get('time', 0) for q in queries]
    
    latency_stats = {
        'total_queries': len(queries),
        'avg_latency': np.mean(times) if times else 0,
        'min_latency': np.min(times) if times else 0,
        'max_latency': np.max(times) if times else 0,
        'median_latency': np.median(times) if times else 0,
        'std_latency': np.std(times) if times else 0,
        'p95_latency': np.percentile(times, 95) if times else 0,
        'p99_latency': np.percentile(times, 99) if times else 0,
    }
    
    print(f"\n📊 Latency Statistics:")
    print(f"   • Total Queries: {latency_stats['total_queries']}")
    print(f"   • Average Latency: {latency_stats['avg_latency']:.2f}s")
    print(f"   • Min Latency: {latency_stats['min_latency']:.2f}s")
    print(f"   • Max Latency: {latency_stats['max_latency']:.2f}s")
    print(f"   • Median Latency: {latency_stats['median_latency']:.2f}s")
    print(f"   • Std Deviation: {latency_stats['std_latency']:.2f}s")
    print(f"   • P95 Latency: {latency_stats['p95_latency']:.2f}s")
    print(f"   • P99 Latency: {latency_stats['p99_latency']:.2f}s")
    
    return latency_stats

def analyze_bandwidth(metrics, products, name):
    """Analyze bandwidth (data throughput)"""
    print(f"\n{'='*60}")
    print(f"📡 BANDWIDTH ANALYSIS - {name}")
    print(f"{'='*60}")
    
    total_time = metrics.get('total_time', 0)
    total_products = len(products) if products else 0
    
    # Estimate data size (rough calculation based on product fields)
    total_data_bytes = 0
    for p in products:
        # Estimate bytes per product based on content
        product_size = len(str(p))
        total_data_bytes += product_size
    
    total_data_kb = total_data_bytes / 1024
    total_data_mb = total_data_kb / 1024
    
    bandwidth_stats = {
        'total_data_kb': total_data_kb,
        'total_data_mb': total_data_mb,
        'total_time': total_time,
        'products_per_second': total_products / total_time if total_time > 0 else 0,
        'kb_per_second': total_data_kb / total_time if total_time > 0 else 0,
        'mb_per_minute': (total_data_mb / total_time) * 60 if total_time > 0 else 0,
        'avg_product_size_bytes': total_data_bytes / total_products if total_products > 0 else 0,
    }
    
    print(f"\n📊 Bandwidth Statistics:")
    print(f"   • Total Data Scraped: {bandwidth_stats['total_data_kb']:.2f} KB ({bandwidth_stats['total_data_mb']:.2f} MB)")
    print(f"   • Total Time: {bandwidth_stats['total_time']:.2f}s")
    print(f"   • Throughput: {bandwidth_stats['products_per_second']:.4f} products/sec")
    print(f"   • Data Rate: {bandwidth_stats['kb_per_second']:.2f} KB/sec")
    print(f"   • Avg Product Size: {bandwidth_stats['avg_product_size_bytes']:.0f} bytes")
    
    return bandwidth_stats

def analyze_accuracy(products, name):
    """Analyze product accuracy and data quality"""
    print(f"\n{'='*60}")
    print(f"🎯 ACCURACY ANALYSIS - {name}")
    print(f"{'='*60}")
    
    if not products:
        print("   No products to analyze")
        return {}
    
    df = pd.DataFrame(products)
    total = len(df)
    
    # Check completeness of key fields
    accuracy_stats = {
        'total_products': total,
        'has_name': (df['name'].notna() & (df['name'] != '')).sum() if 'name' in df.columns else 0,
        'has_price': (df['price'].notna() & (df['price'] != '')).sum() if 'price' in df.columns else 0,
        'has_rating': (df['rating'].notna() & (df['rating'] != '')).sum() if 'rating' in df.columns else 0,
        'has_reviews': (df['reviews'].notna() & (df['reviews'] != '')).sum() if 'reviews' in df.columns else 0,
        'has_image': (df['image_url'].notna() & (df['image_url'] != '')).sum() if 'image_url' in df.columns else 0,
        'has_link': (df['product_link'].notna() & (df['product_link'] != '')).sum() if 'product_link' in df.columns else 0,
        'has_description': (df['description'].notna() & (df['description'] != '')).sum() if 'description' in df.columns else 0,
        'has_technical_details': 0,
    }
    
    # Check technical details
    if 'technical_details' in df.columns:
        accuracy_stats['has_technical_details'] = df['technical_details'].apply(
            lambda x: x is not None and len(x) > 0 if isinstance(x, (dict, list)) else bool(x)
        ).sum()
    
    # Calculate percentages
    for key in ['has_name', 'has_price', 'has_rating', 'has_reviews', 'has_image', 'has_link', 'has_description', 'has_technical_details']:
        accuracy_stats[f'{key}_pct'] = (accuracy_stats[key] / total * 100) if total > 0 else 0
    
    # Overall accuracy score
    key_fields = ['has_name', 'has_price', 'has_rating', 'has_link']
    accuracy_stats['overall_accuracy'] = np.mean([accuracy_stats[f'{k}_pct'] for k in key_fields])
    
    print(f"\n📊 Data Completeness:")
    print(f"   • Name: {accuracy_stats['has_name']}/{total} ({accuracy_stats['has_name_pct']:.1f}%)")
    print(f"   • Price: {accuracy_stats['has_price']}/{total} ({accuracy_stats['has_price_pct']:.1f}%)")
    print(f"   • Rating: {accuracy_stats['has_rating']}/{total} ({accuracy_stats['has_rating_pct']:.1f}%)")
    print(f"   • Reviews: {accuracy_stats['has_reviews']}/{total} ({accuracy_stats['has_reviews_pct']:.1f}%)")
    print(f"   • Image URL: {accuracy_stats['has_image']}/{total} ({accuracy_stats['has_image_pct']:.1f}%)")
    print(f"   • Product Link: {accuracy_stats['has_link']}/{total} ({accuracy_stats['has_link_pct']:.1f}%)")
    print(f"   • Description: {accuracy_stats['has_description']}/{total} ({accuracy_stats['has_description_pct']:.1f}%)")
    print(f"   • Technical Details: {accuracy_stats['has_technical_details']}/{total} ({accuracy_stats['has_technical_details_pct']:.1f}%)")
    print(f"\n   🏆 Overall Accuracy Score: {accuracy_stats['overall_accuracy']:.1f}%")
    
    return accuracy_stats

def analyze_websites(products, metrics, name):
    """Analyze which website performs better"""
    print(f"\n{'='*60}")
    print(f"🌐 WEBSITE COMPARISON - {name}")
    print(f"{'='*60}")
    
    if not products:
        print("   No products to analyze")
        return {}
    
    df = pd.DataFrame(products)
    
    website_stats = {}
    sources = df['source'].unique() if 'source' in df.columns else []
    
    for source in sources:
        source_df = df[df['source'] == source]
        count = len(source_df)
        
        # Calculate accuracy for this source
        has_price = (source_df['price'].notna() & (source_df['price'] != '')).sum() if 'price' in source_df.columns else 0
        has_rating = (source_df['rating'].notna() & (source_df['rating'] != '')).sum() if 'rating' in source_df.columns else 0
        has_reviews = (source_df['reviews'].notna() & (source_df['reviews'] != '')).sum() if 'reviews' in source_df.columns else 0
        
        # Price statistics
        prices = pd.to_numeric(source_df['price'].astype(str).str.replace('[₹,]', '', regex=True), errors='coerce')
        valid_prices = prices.dropna()
        
        website_stats[source] = {
            'product_count': count,
            'price_completeness': (has_price / count * 100) if count > 0 else 0,
            'rating_completeness': (has_rating / count * 100) if count > 0 else 0,
            'review_completeness': (has_reviews / count * 100) if count > 0 else 0,
            'avg_price': valid_prices.mean() if len(valid_prices) > 0 else 0,
            'min_price': valid_prices.min() if len(valid_prices) > 0 else 0,
            'max_price': valid_prices.max() if len(valid_prices) > 0 else 0,
        }
        
        # Calculate overall score (products + accuracy)
        website_stats[source]['score'] = (
            website_stats[source]['product_count'] * 0.4 +
            website_stats[source]['price_completeness'] * 0.3 +
            website_stats[source]['rating_completeness'] * 0.3
        )
    
    # Rank websites
    ranked = sorted(website_stats.items(), key=lambda x: x[1]['score'], reverse=True)
    
    print(f"\n📊 Website Performance Ranking:")
    for i, (source, stats) in enumerate(ranked, 1):
        print(f"\n   #{i} {source}")
        print(f"      Products: {stats['product_count']}")
        print(f"      Price Data: {stats['price_completeness']:.1f}%")
        print(f"      Rating Data: {stats['rating_completeness']:.1f}%")
        print(f"      Avg Price: ₹{stats['avg_price']:,.0f}")
        print(f"      Score: {stats['score']:.1f}")
    
    if ranked:
        print(f"\n   🏆 BEST WEBSITE: {ranked[0][0]} (Score: {ranked[0][1]['score']:.1f})")
    
    return website_stats

def analyze_queries(metrics, products, name):
    """Analyze which queries give most/least results"""
    print(f"\n{'='*60}")
    print(f"🔍 QUERY ANALYSIS - {name}")
    print(f"{'='*60}")
    
    queries = metrics.get('queries', [])
    if not queries:
        # Try to extract from products
        if products and 'search_query' in products[0]:
            df = pd.DataFrame(products)
            query_counts = df['search_query'].value_counts()
            queries = [{'query': q, 'products': c, 'time': 0} for q, c in query_counts.items()]
    
    if not queries:
        print("   No query-level data available")
        return {}
    
    # Sort by products
    by_products = sorted(queries, key=lambda x: x.get('products', 0), reverse=True)
    by_time = sorted(queries, key=lambda x: x.get('time', 0), reverse=True)
    
    query_stats = {
        'total_queries': len(queries),
        'total_products': sum(q.get('products', 0) for q in queries),
        'avg_products_per_query': np.mean([q.get('products', 0) for q in queries]),
        'max_products_query': by_products[0] if by_products else None,
        'min_products_query': by_products[-1] if by_products else None,
        'slowest_query': by_time[0] if by_time else None,
        'fastest_query': by_time[-1] if by_time else None,
    }
    
    print(f"\n📊 Query Statistics:")
    print(f"   • Total Queries: {query_stats['total_queries']}")
    print(f"   • Total Products: {query_stats['total_products']}")
    print(f"   • Avg Products/Query: {query_stats['avg_products_per_query']:.2f}")
    
    print(f"\n   📈 TOP 5 QUERIES (Most Products):")
    for i, q in enumerate(by_products[:5], 1):
        print(f"      {i}. \"{q.get('query', 'N/A')}\" → {q.get('products', 0)} products ({q.get('time', 0):.1f}s)")
    
    print(f"\n   📉 BOTTOM 5 QUERIES (Least Products):")
    for i, q in enumerate(by_products[-5:], 1):
        print(f"      {i}. \"{q.get('query', 'N/A')}\" → {q.get('products', 0)} products ({q.get('time', 0):.1f}s)")
    
    print(f"\n   🐢 SLOWEST QUERIES:")
    for i, q in enumerate(by_time[:3], 1):
        print(f"      {i}. \"{q.get('query', 'N/A')}\" → {q.get('time', 0):.1f}s ({q.get('products', 0)} products)")
    
    print(f"\n   ⚡ FASTEST QUERIES:")
    for i, q in enumerate(by_time[-3:], 1):
        print(f"      {i}. \"{q.get('query', 'N/A')}\" → {q.get('time', 0):.1f}s ({q.get('products', 0)} products)")
    
    # Category analysis
    categories = {}
    for q in queries:
        cat = q.get('category', 'Unknown')
        if cat not in categories:
            categories[cat] = {'products': 0, 'queries': 0, 'time': 0}
        categories[cat]['products'] += q.get('products', 0)
        categories[cat]['queries'] += 1
        categories[cat]['time'] += q.get('time', 0)
    
    print(f"\n   📂 CATEGORY BREAKDOWN:")
    for cat, stats in sorted(categories.items(), key=lambda x: x[1]['products'], reverse=True):
        avg_time = stats['time'] / stats['queries'] if stats['queries'] > 0 else 0
        print(f"      • {cat}: {stats['products']} products, {stats['queries']} queries, {avg_time:.1f}s avg")
    
    query_stats['categories'] = categories
    return query_stats


def analyze_products(products, name):
    """Analyze product data and print summary"""
    print(f"\n{'='*60}")
    print(f"📊 {name} PRODUCT SUMMARY")
    print(f"{'='*60}")
    
    if not products:
        print("No products found!")
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(products)
    
    print(f"\n📦 Total Products: {len(df)}")
    
    # Source distribution
    if 'source' in df.columns:
        print(f"\n📍 Products by Source:")
        source_counts = df['source'].value_counts()
        for source, count in source_counts.items():
            print(f"   • {source}: {count}")
    
    # Category distribution
    if 'category' in df.columns:
        print(f"\n📂 Products by Category:")
        cat_counts = df['category'].value_counts()
        for cat, count in cat_counts.items():
            print(f"   • {cat}: {count}")
    
    return df


def generate_umap(products, output_file, title):
    """Generate UMAP visualization"""
    if not UMAP_ANALYZER_AVAILABLE:
        print(f"⚠ Cannot generate UMAP - UMAPAnalyzer not available")
        return None
    
    if not UMAP_AVAILABLE:
        print(f"⚠ Cannot generate UMAP - umap-learn not installed")
        return None
    
    if not products or len(products) < 10:
        print(f"⚠ Not enough products for UMAP (need at least 10, have {len(products) if products else 0})")
        return None
    
    print(f"\n🔮 Generating UMAP for {title}...")
    
    try:
        # Ensure products have required fields
        for p in products:
            if 'name' not in p:
                p['name'] = p.get('product_name', 'Unknown Product')
            if 'price' not in p or p['price'] is None:
                p['price'] = 0
            else:
                # Clean price - extract numeric value
                price_str = str(p['price']).replace('₹', '').replace(',', '').replace('₹', '').strip()
                try:
                    p['price'] = float(price_str.split()[0]) if price_str else 0
                except:
                    p['price'] = 0
        
        # Initialize UMAPAnalyzer with products
        analyzer = UMAPAnalyzer(products)
        
        # Prepare features and run UMAP
        analyzer.prepare_features()
        analyzer.run_umap()
        
        # Calculate metrics
        analyzer.calculate_clustering_metrics()
        
        # Create visualization
        analyzer.create_visualization(save_path=output_file)
        
        # Get metrics
        metrics = analyzer.metrics if hasattr(analyzer, 'metrics') else {}
        
        print(f"✓ Saved UMAP: {output_file}")
        if metrics:
            print(f"\n📊 UMAP Metrics:")
            print(f"   • Silhouette Score: {metrics.get('silhouette_score', 0):.4f}")
            print(f"   • Davies-Bouldin Index: {metrics.get('davies_bouldin_index', 0):.4f}")
            print(f"   • Cluster Purity: {metrics.get('cluster_purity', 0):.2f}%")
            print(f"   • Number of Clusters: {metrics.get('n_clusters', 0)}")
        
        return metrics
        
    except Exception as e:
        print(f"✗ UMAP Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_comparison_table(single_df, multi_df, single_umap, multi_umap, single_metrics, multi_metrics):
    """Generate comparison table with power, time, and accuracy"""
    print(f"\n{'='*60}")
    print("📊 COMPARISON TABLE")
    print(f"{'='*60}")
    
    comparison = []
    
    # Product counts
    comparison.append(["Total Products", len(single_df), len(multi_df)])
    
    # Source counts
    for source in ['Amazon.in', 'Flipkart', 'Croma', 'Reliance Digital']:
        s_count = len(single_df[single_df['source'] == source]) if 'source' in single_df.columns else 0
        m_count = len(multi_df[multi_df['source'] == source]) if 'source' in multi_df.columns else 0
        comparison.append([f"{source} Products", s_count, m_count])
    
    # UMAP metrics
    if single_umap and multi_umap:
        comparison.append(["Silhouette Score", f"{single_umap.get('silhouette_score', 0):.4f}", f"{multi_umap.get('silhouette_score', 0):.4f}"])
        comparison.append(["Davies-Bouldin Index", f"{single_umap.get('davies_bouldin_index', 0):.4f}", f"{multi_umap.get('davies_bouldin_index', 0):.4f}"])
        comparison.append(["Cluster Purity (%)", f"{single_umap.get('cluster_purity', 0):.2f}", f"{multi_umap.get('cluster_purity', 0):.2f}"])
        comparison.append(["N Categories", single_umap.get('n_categories', 0), multi_umap.get('n_categories', 0)])
    
    # Print table manually (no tabulate dependency)
    print("\n" + "-"*60)
    print(f"{'Metric':<25} {'Single Agent':<15} {'Multi-Agent':<15}")
    print("-"*60)
    for row in comparison:
        print(f"{row[0]:<25} {str(row[1]):<15} {str(row[2]):<15}")
    print("-"*60)
    
    # Time metrics table
    time_table = []
    s_time = single_metrics.get('total_time', 0)
    m_time = multi_metrics.get('total_time', 0)
    time_table.append(["Total Time (seconds)", f"{s_time:.2f}", f"{m_time:.2f}"])
    
    s_avg_time = single_metrics.get('avg_time_per_query', 0)
    m_avg_time = multi_metrics.get('avg_time_per_query', 0)
    time_table.append(["Avg Time/Query (sec)", f"{s_avg_time:.2f}", f"{m_avg_time:.2f}"])
    
    s_prod_rate = len(single_df) / s_time if s_time > 0 else 0
    m_prod_rate = len(multi_df) / m_time if m_time > 0 else 0
    time_table.append(["Products/Second", f"{s_prod_rate:.4f}", f"{m_prod_rate:.4f}"])
    
    speedup = s_time / m_time if m_time > 0 else 0
    time_table.append(["Speedup Factor", "1.00x", f"{speedup:.2f}x"])
    
    print("\n⏱️ TIME PERFORMANCE:")
    print("-"*60)
    for row in time_table:
        print(f"{row[0]:<25} {str(row[1]):<15} {str(row[2]):<15}")
    print("-"*60)
    
    # Power metrics table
    power_table = []
    s_power = single_metrics.get('power', {})
    m_power = multi_metrics.get('power', {})
    
    s_cpu = s_power.get('avg_cpu_percent', 0)
    m_cpu = m_power.get('avg_cpu_percent', 0)
    power_table.append(["Avg CPU Usage (%)", f"{s_cpu:.1f}", f"{m_cpu:.1f}"])
    
    s_watts = s_power.get('avg_cpu_power_watts', 0)
    m_watts = m_power.get('avg_cpu_power_watts', 0)
    power_table.append(["Avg CPU Power (W)", f"{s_watts:.2f}", f"{m_watts:.2f}"])
    
    s_mem = s_power.get('avg_memory_percent', 0)
    m_mem = m_power.get('avg_memory_percent', 0)
    power_table.append(["Avg Memory (%)", f"{s_mem:.1f}", f"{m_mem:.1f}"])
    
    s_energy = s_power.get('total_energy_kwh', 0)
    m_energy = m_power.get('total_energy_kwh', 0)
    power_table.append(["Total Energy (kWh)", f"{s_energy:.6f}", f"{m_energy:.6f}"])
    
    s_co2 = s_power.get('co2_emissions_kg', 0)
    m_co2 = m_power.get('co2_emissions_kg', 0)
    power_table.append(["CO₂ Emissions (kg)", f"{s_co2:.6f}", f"{m_co2:.6f}"])
    
    print("\n⚡ POWER CONSUMPTION:")
    print("-"*60)
    for row in power_table:
        print(f"{row[0]:<25} {str(row[1]):<15} {str(row[2]):<15}")
    print("-"*60)
    
    # Accuracy metrics
    accuracy_table = []
    s_errors = single_metrics.get('errors', 0)
    m_errors = multi_metrics.get('errors', 0)
    accuracy_table.append(["Total Errors", s_errors, m_errors])
    
    s_success = len(single_df) / (len(single_df) + s_errors) * 100 if (len(single_df) + s_errors) > 0 else 0
    m_success = len(multi_df) / (len(multi_df) + m_errors) * 100 if (len(multi_df) + m_errors) > 0 else 0
    accuracy_table.append(["Success Rate (%)", f"{s_success:.1f}", f"{m_success:.1f}"])
    
    s_avg_prod = single_metrics.get('avg_products_per_query', 0)
    m_avg_prod = multi_metrics.get('avg_products_per_query', 0)
    accuracy_table.append(["Avg Products/Query", f"{s_avg_prod:.2f}", f"{m_avg_prod:.2f}"])
    
    # Source coverage
    s_sources = len([s for s in ['Amazon.in', 'Flipkart', 'Croma', 'Reliance Digital'] 
                     if len(single_df[single_df['source'] == s]) > 0]) if 'source' in single_df.columns else 0
    m_sources = len([s for s in ['Amazon.in', 'Flipkart', 'Croma', 'Reliance Digital'] 
                     if len(multi_df[multi_df['source'] == s]) > 0]) if 'source' in multi_df.columns else 0
    accuracy_table.append(["Source Coverage", f"{s_sources}/4", f"{m_sources}/4"])
    
    print("\n🎯 ACCURACY & RELIABILITY:")
    print("-"*60)
    for row in accuracy_table:
        print(f"{row[0]:<25} {str(row[1]):<15} {str(row[2]):<15}")
    print("-"*60)
    
    # Save to markdown
    with open("PKL_COMPARISON_REPORT.md", "w", encoding="utf-8") as f:
        f.write("# PKL Results Comparison Report\n\n")
        f.write("## 1. Product Comparison\n\n")
        f.write("| Metric | Single Agent | Multi-Agent |\n")
        f.write("|--------|--------------|-------------|\n")
        for row in comparison:
            f.write(f"| {row[0]} | {row[1]} | {row[2]} |\n")
        
        f.write("\n## 2. Time Performance\n\n")
        f.write("| Metric | Single Agent | Multi-Agent |\n")
        f.write("|--------|--------------|-------------|\n")
        for row in time_table:
            f.write(f"| {row[0]} | {row[1]} | {row[2]} |\n")
        
        f.write("\n## 3. Power Consumption\n\n")
        f.write("| Metric | Single Agent | Multi-Agent |\n")
        f.write("|--------|--------------|-------------|\n")
        for row in power_table:
            f.write(f"| {row[0]} | {row[1]} | {row[2]} |\n")
        
        f.write("\n## 4. Accuracy & Reliability\n\n")
        f.write("| Metric | Single Agent | Multi-Agent |\n")
        f.write("|--------|--------------|-------------|\n")
        for row in accuracy_table:
            f.write(f"| {row[0]} | {row[1]} | {row[2]} |\n")
        
        f.write("\n## 5. UMAP Visualizations\n\n")
        f.write("- Single Agent: `umap_single_agent_pkl.png`\n")
        f.write("- Multi-Agent: `umap_multi_agent_pkl.png`\n")
        
        # Conclusion
        f.write("\n## 6. Conclusion\n\n")
        if speedup > 1:
            f.write(f"**Multi-Agent is {speedup:.2f}x faster** than Single Agent.\n\n")
        elif speedup > 0:
            f.write(f"**Single Agent is {1/speedup:.2f}x faster** than Multi-Agent.\n\n")
        else:
            f.write("**Time comparison not available** (single agent timing data missing).\n\n")
        
        f.write(f"- **Products Scraped:** Multi-Agent got {len(multi_df) - len(single_df)} more products\n")
        f.write(f"- **Source Coverage:** Multi-Agent covers {m_sources}/4 sources vs Single Agent {s_sources}/4\n")
        f.write(f"- **Success Rate:** Single={s_success:.1f}% vs Multi={m_success:.1f}%\n")
    
    print("\n✓ Saved: PKL_COMPARISON_REPORT.md")

def main():
    print("="*60)
    print("🔍 PKL RESULTS ANALYZER - COMPREHENSIVE REPORT")
    print("="*60)
    
    # Load pickle files
    single_data = load_pickle_data("test_single_agent_rag.pkl")
    multi_data = load_pickle_data("scraped_multi_agent.pkl")
    
    # Extract products and metrics
    single_products = []
    multi_products = []
    single_metrics = {}
    multi_metrics = {}
    
    # Handle different data formats
    if single_data:
        if isinstance(single_data, list):
            single_products = single_data
        elif isinstance(single_data, dict):
            single_products = single_data.get('products', single_data.get('results', []))
            single_metrics = single_data.get('metrics', {})
            if not single_products:
                # Try to flatten dict values
                for key, val in single_data.items():
                    if isinstance(val, list):
                        single_products.extend(val)
    
    if multi_data:
        if isinstance(multi_data, list):
            multi_products = multi_data
        elif isinstance(multi_data, dict):
            multi_products = multi_data.get('products', multi_data.get('results', []))
            multi_metrics = multi_data.get('metrics', {})
            if not multi_products:
                for key, val in multi_data.items():
                    if isinstance(val, list):
                        multi_products.extend(val)
    
    # ============================================
    # SINGLE AGENT ANALYSIS
    # ============================================
    print("\n" + "🔵"*30)
    print("SINGLE AGENT DETAILED ANALYSIS")
    print("🔵"*30)
    
    single_df = analyze_products(single_products, "SINGLE AGENT")
    single_latency = analyze_latency(single_metrics, "SINGLE AGENT")
    single_bandwidth = analyze_bandwidth(single_metrics, single_products, "SINGLE AGENT")
    single_accuracy = analyze_accuracy(single_products, "SINGLE AGENT")
    single_websites = analyze_websites(single_products, single_metrics, "SINGLE AGENT")
    single_queries = analyze_queries(single_metrics, single_products, "SINGLE AGENT")
    
    # ============================================
    # MULTI-AGENT ANALYSIS
    # ============================================
    print("\n" + "🟢"*30)
    print("MULTI-AGENT DETAILED ANALYSIS")
    print("🟢"*30)
    
    multi_df = analyze_products(multi_products, "MULTI-AGENT")
    multi_latency = analyze_latency(multi_metrics, "MULTI-AGENT")
    multi_bandwidth = analyze_bandwidth(multi_metrics, multi_products, "MULTI-AGENT")
    multi_accuracy = analyze_accuracy(multi_products, "MULTI-AGENT")
    multi_websites = analyze_websites(multi_products, multi_metrics, "MULTI-AGENT")
    multi_queries = analyze_queries(multi_metrics, multi_products, "MULTI-AGENT")
    
    # Generate UMAP visualizations
    single_umap = generate_umap(single_products, "umap_single_agent_pkl.png", "Single Agent UMAP")
    multi_umap = generate_umap(multi_products, "umap_multi_agent_pkl.png", "Multi-Agent UMAP")
    
    # Generate comparison table with all metrics
    if single_df is not None and multi_df is not None:
        generate_comparison_table(
            single_df, multi_df, single_umap, multi_umap, single_metrics, multi_metrics
        )
        
        # Generate detailed markdown report
        generate_detailed_report(
            single_products, multi_products,
            single_metrics, multi_metrics,
            single_latency, multi_latency,
            single_bandwidth, multi_bandwidth,
            single_accuracy, multi_accuracy,
            single_websites, multi_websites,
            single_queries, multi_queries,
            single_umap, multi_umap
        )
    
    print("\n" + "="*60)
    print("✅ ANALYSIS COMPLETE!")
    print("="*60)
    print("\n📁 Generated Files:")
    print("   • umap_single_agent_pkl.png")
    print("   • umap_multi_agent_pkl.png")
    print("   • PKL_COMPARISON_REPORT.md")
    print("   • PKL_DETAILED_ANALYSIS.md")


def generate_detailed_report(
    single_products, multi_products,
    single_metrics, multi_metrics,
    single_latency, multi_latency,
    single_bandwidth, multi_bandwidth,
    single_accuracy, multi_accuracy,
    single_websites, multi_websites,
    single_queries, multi_queries,
    single_umap, multi_umap
):
    """Generate comprehensive detailed markdown report"""
    
    report = []
    report.append("# PKL Detailed Analysis Report\n")
    report.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    # ============================================
    # 1. EXECUTIVE SUMMARY
    # ============================================
    report.append("## 1. Executive Summary\n\n")
    report.append("| Metric | Single Agent | Multi-Agent | Winner |\n")
    report.append("|--------|--------------|-------------|--------|\n")
    
    s_total = len(single_products)
    m_total = len(multi_products)
    winner_products = "Multi-Agent ✅" if m_total > s_total else "Single Agent ✅"
    report.append(f"| Total Products | {s_total} | {m_total} | {winner_products} |\n")
    
    s_acc = single_accuracy.get('overall_accuracy', 0)
    m_acc = multi_accuracy.get('overall_accuracy', 0)
    winner_acc = "Multi-Agent ✅" if m_acc > s_acc else "Single Agent ✅" if s_acc > m_acc else "Tie"
    report.append(f"| Data Accuracy | {s_acc:.1f}% | {m_acc:.1f}% | {winner_acc} |\n")
    
    s_lat = single_latency.get('avg_latency', 0)
    m_lat = multi_latency.get('avg_latency', 0)
    winner_lat = "Single Agent ✅" if s_lat < m_lat and s_lat > 0 else "Multi-Agent ✅" if m_lat > 0 else "N/A"
    report.append(f"| Avg Latency (s) | {s_lat:.2f} | {m_lat:.2f} | {winner_lat} |\n")
    
    s_bw = single_bandwidth.get('products_per_second', 0)
    m_bw = multi_bandwidth.get('products_per_second', 0)
    winner_bw = "Multi-Agent ✅" if m_bw > s_bw else "Single Agent ✅" if s_bw > 0 else "N/A"
    report.append(f"| Throughput (prod/s) | {s_bw:.4f} | {m_bw:.4f} | {winner_bw} |\n\n")
    
    # ============================================
    # 2. LATENCY ANALYSIS
    # ============================================
    report.append("## 2. Latency Analysis\n\n")
    report.append("### 2.1 Latency Comparison\n\n")
    report.append("| Metric | Single Agent | Multi-Agent |\n")
    report.append("|--------|--------------|-------------|\n")
    
    for metric in ['total_queries', 'avg_latency', 'min_latency', 'max_latency', 'median_latency', 'std_latency', 'p95_latency', 'p99_latency']:
        s_val = single_latency.get(metric, 0)
        m_val = multi_latency.get(metric, 0)
        if 'latency' in metric:
            report.append(f"| {metric.replace('_', ' ').title()} | {s_val:.2f}s | {m_val:.2f}s |\n")
        else:
            report.append(f"| {metric.replace('_', ' ').title()} | {s_val} | {m_val} |\n")
    
    # ============================================
    # 3. BANDWIDTH ANALYSIS
    # ============================================
    report.append("\n## 3. Bandwidth Analysis\n\n")
    report.append("| Metric | Single Agent | Multi-Agent |\n")
    report.append("|--------|--------------|-------------|\n")
    
    s_kb = single_bandwidth.get('total_data_kb', 0)
    m_kb = multi_bandwidth.get('total_data_kb', 0)
    report.append(f"| Total Data Scraped | {s_kb:.2f} KB | {m_kb:.2f} KB |\n")
    
    s_ps = single_bandwidth.get('products_per_second', 0)
    m_ps = multi_bandwidth.get('products_per_second', 0)
    report.append(f"| Products/Second | {s_ps:.4f} | {m_ps:.4f} |\n")
    
    s_kbs = single_bandwidth.get('kb_per_second', 0)
    m_kbs = multi_bandwidth.get('kb_per_second', 0)
    report.append(f"| KB/Second | {s_kbs:.2f} | {m_kbs:.2f} |\n")
    
    s_avg = single_bandwidth.get('avg_product_size_bytes', 0)
    m_avg = multi_bandwidth.get('avg_product_size_bytes', 0)
    report.append(f"| Avg Product Size | {s_avg:.0f} bytes | {m_avg:.0f} bytes |\n")
    
    # ============================================
    # 4. ACCURACY ANALYSIS
    # ============================================
    report.append("\n## 4. Accuracy & Data Quality\n\n")
    report.append("### 4.1 Field Completeness\n\n")
    report.append("| Field | Single Agent | Multi-Agent |\n")
    report.append("|-------|--------------|-------------|\n")
    
    for field in ['has_name', 'has_price', 'has_rating', 'has_reviews', 'has_image', 'has_link', 'has_description', 'has_technical_details']:
        s_pct = single_accuracy.get(f'{field}_pct', 0)
        m_pct = multi_accuracy.get(f'{field}_pct', 0)
        field_name = field.replace('has_', '').replace('_', ' ').title()
        report.append(f"| {field_name} | {s_pct:.1f}% | {m_pct:.1f}% |\n")
    
    report.append(f"\n**Overall Accuracy Score:**\n")
    report.append(f"- Single Agent: {single_accuracy.get('overall_accuracy', 0):.1f}%\n")
    report.append(f"- Multi-Agent: {multi_accuracy.get('overall_accuracy', 0):.1f}%\n\n")
    
    # ============================================
    # 5. WEBSITE COMPARISON
    # ============================================
    report.append("## 5. Website Performance\n\n")
    
    # Multi-agent websites
    report.append("### 5.1 Multi-Agent Website Ranking\n\n")
    report.append("| Rank | Website | Products | Price Data | Rating Data | Score |\n")
    report.append("|------|---------|----------|------------|-------------|-------|\n")
    
    ranked = sorted(multi_websites.items(), key=lambda x: x[1]['score'], reverse=True)
    for i, (source, stats) in enumerate(ranked, 1):
        report.append(f"| #{i} | {source} | {stats['product_count']} | {stats['price_completeness']:.1f}% | {stats['rating_completeness']:.1f}% | {stats['score']:.1f} |\n")
    
    if ranked:
        report.append(f"\n**🏆 Best Website: {ranked[0][0]}**\n\n")
    
    # ============================================
    # 6. QUERY ANALYSIS
    # ============================================
    report.append("## 6. Query Analysis\n\n")
    
    # Top queries
    queries = multi_metrics.get('queries', [])
    if queries:
        by_products = sorted(queries, key=lambda x: x.get('products', 0), reverse=True)
        
        report.append("### 6.1 Top 10 Queries (Most Products)\n\n")
        report.append("| Rank | Query | Category | Products | Time (s) |\n")
        report.append("|------|-------|----------|----------|----------|\n")
        
        for i, q in enumerate(by_products[:10], 1):
            report.append(f"| {i} | {q.get('query', 'N/A')} | {q.get('category', 'N/A')} | {q.get('products', 0)} | {q.get('time', 0):.1f} |\n")
        
        report.append("\n### 6.2 Bottom 10 Queries (Least Products)\n\n")
        report.append("| Rank | Query | Category | Products | Time (s) |\n")
        report.append("|------|-------|----------|----------|----------|\n")
        
        for i, q in enumerate(by_products[-10:], 1):
            report.append(f"| {i} | {q.get('query', 'N/A')} | {q.get('category', 'N/A')} | {q.get('products', 0)} | {q.get('time', 0):.1f} |\n")
        
        # Category breakdown
        report.append("\n### 6.3 Category Performance\n\n")
        report.append("| Category | Total Products | Queries | Avg Time (s) |\n")
        report.append("|----------|----------------|---------|-------------|\n")
        
        categories = multi_queries.get('categories', {})
        for cat, stats in sorted(categories.items(), key=lambda x: x[1]['products'], reverse=True):
            avg_time = stats['time'] / stats['queries'] if stats['queries'] > 0 else 0
            report.append(f"| {cat} | {stats['products']} | {stats['queries']} | {avg_time:.1f} |\n")
    
    # ============================================
    # 7. UMAP CLUSTERING
    # ============================================
    report.append("\n## 7. UMAP Clustering Quality\n\n")
    report.append("| Metric | Single Agent | Multi-Agent | Interpretation |\n")
    report.append("|--------|--------------|-------------|----------------|\n")
    
    if single_umap and multi_umap:
        s_sil = single_umap.get('silhouette_score', 0)
        m_sil = multi_umap.get('silhouette_score', 0)
        report.append(f"| Silhouette Score | {s_sil:.4f} | {m_sil:.4f} | Higher = Better |\n")
        
        s_db = single_umap.get('davies_bouldin_index', 0)
        m_db = multi_umap.get('davies_bouldin_index', 0)
        report.append(f"| Davies-Bouldin | {s_db:.4f} | {m_db:.4f} | Lower = Better |\n")
        
        s_pur = single_umap.get('cluster_purity', 0)
        m_pur = multi_umap.get('cluster_purity', 0)
        report.append(f"| Cluster Purity | {s_pur:.1f}% | {m_pur:.1f}% | Higher = Better |\n")
    
    report.append("\n**Visualizations:**\n")
    report.append("- Single Agent: `umap_single_agent_pkl.png`\n")
    report.append("- Multi-Agent: `umap_multi_agent_pkl.png`\n\n")
    
    # ============================================
    # 8. CONCLUSION
    # ============================================
    report.append("## 8. Conclusion\n\n")
    
    # Determine overall winner
    scores = {'Single Agent': 0, 'Multi-Agent': 0}
    
    if m_total > s_total:
        scores['Multi-Agent'] += 1
    else:
        scores['Single Agent'] += 1
    
    if m_acc > s_acc:
        scores['Multi-Agent'] += 1
    elif s_acc > m_acc:
        scores['Single Agent'] += 1
    
    if len(multi_websites) > len(single_websites):
        scores['Multi-Agent'] += 1
    
    winner = max(scores, key=scores.get)
    
    report.append(f"**Overall Winner: {winner}**\n\n")
    report.append("### Key Findings:\n\n")
    report.append(f"1. **Product Count:** Multi-Agent scraped {m_total - s_total} more products ({m_total} vs {s_total})\n")
    report.append(f"2. **Data Quality:** {'Multi-Agent' if m_acc > s_acc else 'Single Agent'} has better data accuracy ({max(m_acc, s_acc):.1f}%)\n")
    report.append(f"3. **Website Coverage:** Multi-Agent covers {len(multi_websites)} websites vs Single Agent {len(single_websites)}\n")
    
    if queries:
        best_query = by_products[0] if by_products else {}
        worst_query = by_products[-1] if by_products else {}
        report.append(f"4. **Best Query:** \"{best_query.get('query', 'N/A')}\" with {best_query.get('products', 0)} products\n")
        report.append(f"5. **Worst Query:** \"{worst_query.get('query', 'N/A')}\" with {worst_query.get('products', 0)} products\n")
    
    # Save report
    with open('PKL_DETAILED_ANALYSIS.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print("\n✓ Saved: PKL_DETAILED_ANALYSIS.md")

if __name__ == "__main__":
    main()
