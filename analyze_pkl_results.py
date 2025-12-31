"""
Chapter 5 Results Generator: Comprehensive Analysis of E-Commerce Scraper Performance

This script analyzes the results from comparison_test.py and generates detailed
Chapter 5 style results with performance metrics, visualizations, and comparisons.

Usage:
    python analyze_pkl_results.py

Output:
    - CHAPTER_5_RESULTS.md (Complete Chapter 5 report)
    - Performance visualizations
"""

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from typing import Dict, List, Any, Optional

# Set style for professional plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_test_results() -> tuple:
    """Load results from both test methods"""
    single_data, multi_data = None, None
    
    if os.path.exists('scraped_single_agent.pkl'):
        with open('scraped_single_agent.pkl', 'rb') as f:
            single_data = pickle.load(f)
        print("✓ Loaded single agent results")
    
    if os.path.exists('scraped_multi_agent.pkl'):
        with open('scraped_multi_agent.pkl', 'rb') as f:
            multi_data = pickle.load(f)
        print("✓ Loaded multi-agent results")
    
    return single_data, multi_data

def generate_performance_plots(single_data: Dict, multi_data: Dict):
    """Generate performance comparison plots"""
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('E-Commerce Scraper Performance Analysis', fontsize=16, fontweight='bold')
    
    # 1. Total Products Comparison
    methods = []
    products = []
    if single_data:
        methods.append('Single Agent')
        products.append(single_data['metrics']['total_products'])
    if multi_data:
        methods.append('Multi-Agent')
        products.append(multi_data['metrics']['total_products'])
    
    axes[0,0].bar(methods, products, color=['#3498db', '#e74c3c'])
    axes[0,0].set_title('Total Products Scraped')
    axes[0,0].set_ylabel('Number of Products')
    for i, v in enumerate(products):
        axes[0,0].text(i, v + 1, str(v), ha='center', fontweight='bold')
    
    # 2. Time Efficiency
    times = []
    if single_data:
        times.append(single_data['metrics']['total_time_minutes'])
    if multi_data:
        times.append(multi_data['metrics']['total_time_minutes'])
    
    axes[0,1].bar(methods, times, color=['#3498db', '#e74c3c'])
    axes[0,1].set_title('Total Execution Time')
    axes[0,1].set_ylabel('Time (minutes)')
    for i, v in enumerate(times):
        axes[0,1].text(i, v + 0.5, f'{v:.1f}m', ha='center', fontweight='bold')
    
    # 3. Throughput Comparison
    throughput = []
    if single_data:
        throughput.append(single_data['metrics']['throughput_products_per_second'])
    if multi_data:
        throughput.append(multi_data['metrics']['throughput_products_per_second'])
    
    axes[0,2].bar(methods, throughput, color=['#3498db', '#e74c3c'])
    axes[0,2].set_title('Throughput (Products/Second)')
    axes[0,2].set_ylabel('Products per Second')
    for i, v in enumerate(throughput):
        axes[0,2].text(i, v + 0.001, f'{v:.3f}', ha='center', fontweight='bold')
    
    # 4. Source Distribution (Multi-Agent only)
    if multi_data:
        sources = multi_data['metrics']['products_per_source']
        source_names = list(sources.keys())
        source_counts = list(sources.values())
        
        axes[1,0].pie(source_counts, labels=source_names, autopct='%1.1f%%', startangle=90)
        axes[1,0].set_title('Source Distribution (Multi-Agent)')
    else:
        axes[1,0].text(0.5, 0.5, 'Multi-Agent data\nnot available', 
                      ha='center', va='center', transform=axes[1,0].transAxes)
        axes[1,0].set_title('Source Distribution')
    
    # 5. Latency Distribution
    if multi_data and 'queries' in multi_data['metrics']:
        query_times = [q['time'] for q in multi_data['metrics']['queries'] if q['time'] > 0]
        if query_times:
            axes[1,1].hist(query_times, bins=15, alpha=0.7, color='#e74c3c', edgecolor='black')
            axes[1,1].axvline(np.mean(query_times), color='red', linestyle='--', 
                             label=f'Mean: {np.mean(query_times):.1f}s')
            axes[1,1].set_title('Query Latency Distribution')
            axes[1,1].set_xlabel('Time (seconds)')
            axes[1,1].set_ylabel('Frequency')
            axes[1,1].legend()
    
    # 6. Success Rate Comparison
    success_rates = []
    if single_data:
        success_rates.append(single_data['metrics']['success_rate'])
    if multi_data:
        success_rates.append(multi_data['metrics']['success_rate'])
    
    axes[1,2].bar(methods, success_rates, color=['#3498db', '#e74c3c'])
    axes[1,2].set_title('Success Rate')
    axes[1,2].set_ylabel('Success Rate (%)')
    axes[1,2].set_ylim(0, 100)
    for i, v in enumerate(success_rates):
        axes[1,2].text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('performance_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Generated performance_analysis.png")

def generate_chapter_5_report(single_data: Dict, multi_data: Dict):
    """Generate comprehensive Chapter 5 report"""
    
    report = []
    report.append("# Chapter 5: Results and Analysis")
    report.append("")
    report.append("This chapter presents a comprehensive evaluation of the e-commerce price comparison system, comparing single-agent sequential scraping with multi-agent parallel scraping across four major Indian platforms.")
    report.append("")
    
    # 5.1 System Performance Evaluation
    report.append("## 5.1 System Performance Evaluation")
    report.append("")
    
    # Dataset Overview
    report.append("### 5.1.1 Dataset Overview")
    report.append("")
    
    total_single = single_data['metrics']['total_products'] if single_data else 0
    total_multi = multi_data['metrics']['total_products'] if multi_data else 0
    
    report.append("| Metric | Single Agent | Multi-Agent |")
    report.append("|--------|--------------|-------------|")
    report.append(f"| **Total Products Scraped** | {total_single} | {total_multi} |")
    report.append(f"| **Total Categories Tested** | 10 | 10 |")
    report.append(f"| **Queries per Category** | 4 | 4 |")
    report.append(f"| **Total Search Queries** | 40 | 40 |")
    
    if single_data:
        report.append(f"| **Single Agent Success Rate** | {single_data['metrics']['success_rate']:.1f}% | - |")
    if multi_data:
        report.append(f"| **Multi-Agent Success Rate** | - | {multi_data['metrics']['success_rate']:.1f}% |")
    
    report.append("")
    
    # Product Distribution
    report.append("### 5.1.2 Product Distribution by Source")
    report.append("")
    
    if multi_data:
        sources = multi_data['metrics']['products_per_source']
        report.append("| E-Commerce Platform | Products Scraped | Percentage |")
        report.append("|---------------------|------------------|------------|")
        total = sum(sources.values())
        for source, count in sources.items():
            percentage = (count / total * 100) if total > 0 else 0
            report.append(f"| **{source}** | {count} | {percentage:.1f}% |")
        report.append("")
    
    # Category Distribution
    if multi_data and 'products_per_category' in multi_data['metrics']:
        report.append("### 5.1.3 Product Distribution by Category")
        report.append("")
        categories = multi_data['metrics']['products_per_category']
        report.append("| Product Category | Products Found |")
        report.append("|------------------|----------------|")
        for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            report.append(f"| **{category}** | {count} |")
        report.append("")
    
    # 5.2 Time Efficiency Analysis
    report.append("## 5.2 Time Efficiency Analysis")
    report.append("")
    
    report.append("### 5.2.1 Overall Performance Metrics")
    report.append("")
    
    report.append("| Performance Metric | Single Agent | Multi-Agent | Improvement |")
    report.append("|--------------------|--------------|-------------|-------------|")
    
    if single_data and multi_data:
        single_time = single_data['metrics']['total_time_minutes']
        multi_time = multi_data['metrics']['total_time_minutes']
        time_improvement = ((single_time - multi_time) / single_time * 100) if single_time > 0 else 0
        
        single_throughput = single_data['metrics']['throughput_products_per_second']
        multi_throughput = multi_data['metrics']['throughput_products_per_second']
        throughput_improvement = ((multi_throughput - single_throughput) / single_throughput * 100) if single_throughput > 0 else 0
        
        report.append(f"| **Total Execution Time** | {single_time:.1f} min | {multi_time:.1f} min | {time_improvement:+.1f}% |")
        report.append(f"| **Average Time per Query** | {single_data['metrics']['avg_time_per_query']:.1f}s | {multi_data['metrics']['avg_time_per_query']:.1f}s | - |")
        report.append(f"| **Throughput (Products/sec)** | {single_throughput:.3f} | {multi_throughput:.3f} | {throughput_improvement:+.1f}% |")
        report.append(f"| **Products per Query** | {single_data['metrics']['avg_products_per_query']:.1f} | {multi_data['metrics']['avg_products_per_query']:.1f} | - |")
    
    report.append("")
    
    # Latency Analysis
    report.append("### 5.2.2 Latency Distribution Analysis")
    report.append("")
    
    if multi_data and 'queries' in multi_data['metrics']:
        query_times = [q['time'] for q in multi_data['metrics']['queries'] if q['time'] > 0]
        if query_times:
            report.append("| Latency Metric | Value |")
            report.append("|----------------|-------|")
            report.append(f"| **Mean Latency** | {np.mean(query_times):.2f}s |")
            report.append(f"| **Median Latency** | {np.median(query_times):.2f}s |")
            report.append(f"| **95th Percentile** | {np.percentile(query_times, 95):.2f}s |")
            report.append(f"| **99th Percentile** | {np.percentile(query_times, 99):.2f}s |")
            report.append(f"| **Standard Deviation** | {np.std(query_times):.2f}s |")
            report.append(f"| **Min Latency** | {np.min(query_times):.2f}s |")
            report.append(f"| **Max Latency** | {np.max(query_times):.2f}s |")
            report.append("")
    
    # 5.3 Bandwidth and Data Transfer Analysis
    report.append("## 5.3 Bandwidth and Data Transfer Analysis")
    report.append("")
    
    # Estimate data transfer based on products
    if multi_data:
        total_products = multi_data['metrics']['total_products']
        estimated_data_per_product = 15  # KB (HTML + images + metadata)
        total_data_kb = total_products * estimated_data_per_product
        total_data_mb = total_data_kb / 1024
        
        execution_time_hours = multi_data['metrics']['total_time'] / 3600
        bandwidth_mbps = (total_data_mb * 8) / (execution_time_hours * 3600) if execution_time_hours > 0 else 0
        
        report.append("| Data Transfer Metric | Value |")
        report.append("|---------------------|-------|")
        report.append(f"| **Estimated Data per Product** | {estimated_data_per_product} KB |")
        report.append(f"| **Total Data Transferred** | {total_data_mb:.1f} MB |")
        report.append(f"| **Average Bandwidth Usage** | {bandwidth_mbps:.2f} Mbps |")
        report.append(f"| **Data Efficiency** | {total_data_kb/max(1, total_products):.1f} KB/product |")
        report.append("")
    
    # 5.4 Resource Utilization & Energy Efficiency
    report.append("## 5.4 Resource Utilization & Energy Efficiency")
    report.append("")
    
    if single_data and 'power' in single_data['metrics'] and multi_data and 'power' in multi_data['metrics']:
        single_power = single_data['metrics']['power']
        multi_power = multi_data['metrics']['power']
        
        report.append("| Resource Metric | Single Agent | Multi-Agent |")
        report.append("|-----------------|--------------|-------------|")
        report.append(f"| **Average CPU Usage** | {single_power.get('avg_cpu_percent', 0):.1f}% | {multi_power.get('avg_cpu_percent', 0):.1f}% |")
        report.append(f"| **Average CPU Power** | {single_power.get('avg_cpu_power_watts', 0):.2f}W | {multi_power.get('avg_cpu_power_watts', 0):.2f}W |")
        report.append(f"| **Memory Usage** | {single_power.get('avg_memory_percent', 0):.1f}% | {multi_power.get('avg_memory_percent', 0):.1f}% |")
        report.append(f"| **Total Energy Consumption** | {single_power.get('total_energy_kwh', 0):.6f} kWh | {multi_power.get('total_energy_kwh', 0):.6f} kWh |")
        report.append(f"| **CO₂ Emissions (India)** | {single_power.get('co2_emissions_grams', 0):.2f}g | {multi_power.get('co2_emissions_grams', 0):.2f}g |")
        report.append("")
        
        # Energy efficiency per product
        single_energy_per_product = single_power.get('total_energy_kwh', 0) / max(1, total_single) * 1000000  # µWh
        multi_energy_per_product = multi_power.get('total_energy_kwh', 0) / max(1, total_multi) * 1000000  # µWh
        
        report.append("### 5.4.1 Energy Efficiency Analysis")
        report.append("")
        report.append("| Efficiency Metric | Single Agent | Multi-Agent |")
        report.append("|-------------------|--------------|-------------|")
        report.append(f"| **Energy per Product** | {single_energy_per_product:.2f} µWh | {multi_energy_per_product:.2f} µWh |")
        report.append(f"| **CO₂ per Product** | {single_power.get('co2_emissions_grams', 0)/max(1, total_single):.4f}g | {multi_power.get('co2_emissions_grams', 0)/max(1, total_multi):.4f}g |")
        report.append("")
    
    # 5.5 Data Quality & Accuracy Assessment
    report.append("## 5.5 Data Quality & Accuracy Assessment")
    report.append("")
    
    # Calculate data completeness
    if multi_data and 'products' in multi_data:
        products = multi_data['products']
        total_products = len(products)
        
        # Check completeness of key fields
        name_complete = sum(1 for p in products if p.get('name') and len(str(p['name']).strip()) > 5)
        price_complete = sum(1 for p in products if p.get('price_numeric') and p['price_numeric'] > 0)
        rating_complete = sum(1 for p in products if p.get('rating') and p['rating'] > 0)
        image_complete = sum(1 for p in products if p.get('image_url') and 'http' in str(p['image_url']))
        sentiment_complete = sum(1 for p in products if p.get('sentiment_label'))
        
        report.append("### 5.5.1 Data Completeness Analysis")
        report.append("")
        report.append("| Data Field | Complete Records | Completeness Rate |")
        report.append("|------------|------------------|-------------------|")
        report.append(f"| **Product Name** | {name_complete}/{total_products} | {name_complete/max(1,total_products)*100:.1f}% |")
        report.append(f"| **Price Information** | {price_complete}/{total_products} | {price_complete/max(1,total_products)*100:.1f}% |")
        report.append(f"| **Rating Data** | {rating_complete}/{total_products} | {rating_complete/max(1,total_products)*100:.1f}% |")
        report.append(f"| **Product Images** | {image_complete}/{total_products} | {image_complete/max(1,total_products)*100:.1f}% |")
        report.append(f"| **Sentiment Analysis** | {sentiment_complete}/{total_products} | {sentiment_complete/max(1,total_products)*100:.1f}% |")
        report.append("")
        
        # Price range analysis
        valid_prices = [p['price_numeric'] for p in products if p.get('price_numeric') and p['price_numeric'] > 0]
        if valid_prices:
            report.append("### 5.5.2 Price Distribution Analysis")
            report.append("")
            report.append("| Price Metric | Value |")
            report.append("|--------------|-------|")
            report.append(f"| **Minimum Price** | ₹{min(valid_prices):,.0f} |")
            report.append(f"| **Maximum Price** | ₹{max(valid_prices):,.0f} |")
            report.append(f"| **Average Price** | ₹{np.mean(valid_prices):,.0f} |")
            report.append(f"| **Median Price** | ₹{np.median(valid_prices):,.0f} |")
            report.append(f"| **Price Standard Deviation** | ₹{np.std(valid_prices):,.0f} |")
            report.append("")
    
    # 5.6 Query Performance Analysis
    report.append("## 5.6 Query Performance Analysis")
    report.append("")
    
    if multi_data and 'queries' in multi_data['metrics']:
        queries = multi_data['metrics']['queries']
        
        # Category performance
        category_performance = {}
        for query in queries:
            category = query['category']
            if category not in category_performance:
                category_performance[category] = {'total_time': 0, 'total_products': 0, 'count': 0}
            category_performance[category]['total_time'] += query['time']
            category_performance[category]['total_products'] += query['products']
            category_performance[category]['count'] += 1
        
        report.append("### 5.6.1 Performance by Product Category")
        report.append("")
        report.append("| Category | Avg Time/Query | Avg Products/Query | Total Products |")
        report.append("|----------|----------------|-------------------|----------------|")
        
        for category, perf in sorted(category_performance.items(), key=lambda x: x[1]['total_time']):
            avg_time = perf['total_time'] / max(1, perf['count'])
            avg_products = perf['total_products'] / max(1, perf['count'])
            report.append(f"| **{category}** | {avg_time:.1f}s | {avg_products:.1f} | {perf['total_products']} |")
        
        report.append("")
    
    # 5.7 UMAP Clustering Analysis
    report.append("## 5.7 UMAP Clustering Analysis")
    report.append("")
    
    if multi_data and 'umap' in multi_data['metrics'] and multi_data['metrics']['umap']:
        umap_data = multi_data['metrics']['umap']
        
        report.append("### 5.7.1 Clustering Quality Metrics")
        report.append("")
        report.append("| Clustering Metric | Value | Interpretation |")
        report.append("|-------------------|-------|----------------|")
        report.append(f"| **Silhouette Score** | {umap_data.get('silhouette_score', 0):.3f} | Cluster separation quality (-1 to 1, higher is better) |")
        report.append(f"| **Davies-Bouldin Index** | {umap_data.get('davies_bouldin_index', 0):.2f} | Cluster compactness (lower is better) |")
        report.append(f"| **Cluster Purity** | {umap_data.get('cluster_purity', 0):.1f}% | Category coherence within clusters |")
        report.append(f"| **Number of Clusters** | {umap_data.get('n_clusters', 0)} | Detected product groups |")
        report.append("")
        
        # Interpretation
        silhouette = umap_data.get('silhouette_score', 0)
        if silhouette > 0.5:
            interpretation = "Excellent clustering quality"
        elif silhouette > 0.25:
            interpretation = "Good clustering quality"
        elif silhouette > 0:
            interpretation = "Moderate clustering quality"
        else:
            interpretation = "Poor clustering quality"
        
        report.append(f"**Clustering Quality Assessment:** {interpretation}")
        report.append("")
    
    # 5.8 Sentiment Analysis Validation
    report.append("## 5.8 Sentiment Analysis Validation")
    report.append("")
    
    if multi_data and 'products' in multi_data:
        products = multi_data['products']
        
        # Sentiment distribution
        sentiment_counts = {'Positive': 0, 'Negative': 0, 'Neutral': 0, 'Unknown': 0}
        sentiment_confidences = []
        
        for product in products:
            sentiment = product.get('sentiment_label', 'Unknown')
            if sentiment in sentiment_counts:
                sentiment_counts[sentiment] += 1
            else:
                sentiment_counts['Unknown'] += 1
            
            confidence = product.get('sentiment_confidence', 0)
            if confidence > 0:
                sentiment_confidences.append(confidence)
        
        total_with_sentiment = sum(sentiment_counts.values()) - sentiment_counts['Unknown']
        
        report.append("### 5.8.1 Sentiment Distribution")
        report.append("")
        report.append("| Sentiment | Count | Percentage |")
        report.append("|-----------|-------|------------|")
        
        for sentiment, count in sentiment_counts.items():
            if sentiment != 'Unknown':
                percentage = (count / max(1, total_with_sentiment)) * 100
                report.append(f"| **{sentiment}** | {count} | {percentage:.1f}% |")
        
        report.append("")
        
        if sentiment_confidences:
            report.append("### 5.8.2 Sentiment Analysis Confidence")
            report.append("")
            report.append("| Confidence Metric | Value |")
            report.append("|-------------------|-------|")
            report.append(f"| **Average Confidence** | {np.mean(sentiment_confidences):.1f}% |")
            report.append(f"| **Median Confidence** | {np.median(sentiment_confidences):.1f}% |")
            report.append(f"| **Min Confidence** | {np.min(sentiment_confidences):.1f}% |")
            report.append(f"| **Max Confidence** | {np.max(sentiment_confidences):.1f}% |")
            report.append("")
            
            # High confidence analysis
            high_confidence = sum(1 for c in sentiment_confidences if c >= 80)
            report.append(f"**High Confidence Predictions (≥80%):** {high_confidence}/{len(sentiment_confidences)} ({high_confidence/len(sentiment_confidences)*100:.1f}%)")
            report.append("")
    
    # Conclusions
    report.append("## 5.9 Key Findings and Conclusions")
    report.append("")
    
    if single_data and multi_data:
        # Performance comparison
        if multi_data['metrics']['total_time'] < single_data['metrics']['total_time']:
            time_improvement = ((single_data['metrics']['total_time'] - multi_data['metrics']['total_time']) / single_data['metrics']['total_time']) * 100
            report.append(f"1. **Performance Improvement**: Multi-agent approach achieved {time_improvement:.1f}% faster execution time")
        
        # Throughput comparison
        if multi_data['metrics']['throughput_products_per_second'] > single_data['metrics']['throughput_products_per_second']:
            throughput_improvement = ((multi_data['metrics']['throughput_products_per_second'] - single_data['metrics']['throughput_products_per_second']) / single_data['metrics']['throughput_products_per_second']) * 100
            report.append(f"2. **Throughput Enhancement**: {throughput_improvement:.1f}% improvement in products per second")
        
        # Coverage
        if multi_data['metrics']['total_products'] > single_data['metrics']['total_products']:
            coverage_improvement = multi_data['metrics']['total_products'] - single_data['metrics']['total_products']
            report.append(f"3. **Coverage Expansion**: Multi-agent scraped {coverage_improvement} additional products")
    
    if multi_data:
        # Source diversity
        active_sources = sum(1 for count in multi_data['metrics']['products_per_source'].values() if count > 0)
        report.append(f"4. **Source Diversity**: Successfully integrated {active_sources}/4 e-commerce platforms")
        
        # Success rate
        success_rate = multi_data['metrics']['success_rate']
        report.append(f"5. **Reliability**: Achieved {success_rate:.1f}% success rate across all queries")
    
    report.append("")
    report.append("---")
    report.append(f"*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    
    # Write report to file
    with open('CHAPTER_5_RESULTS.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print("✓ Generated CHAPTER_5_RESULTS.md")

def main():
    """Main analysis function"""
    print("\n" + "=" * 70)
    print("   CHAPTER 5 RESULTS GENERATOR")
    print("   Comprehensive Analysis of E-Commerce Scraper Performance")
    print("=" * 70)
    
    # Load test results
    single_data, multi_data = load_test_results()
    
    if not single_data and not multi_data:
        print("\n❌ No test results found!")
        print("   Run 'python comparison_test.py' first to generate test data")
        return
    
    # Generate visualizations
    if single_data or multi_data:
        print("\n📊 Generating performance plots...")
        generate_performance_plots(single_data, multi_data)
    
    # Generate comprehensive report
    print("\n📝 Generating Chapter 5 report...")
    generate_chapter_5_report(single_data, multi_data)
    
    print("\n✅ Analysis Complete!")
    print("\n📁 Generated Files:")
    for filename in ['CHAPTER_5_RESULTS.md', 'performance_analysis.png']:
        if os.path.exists(filename):
            print(f"   ✓ {filename}")
    
    print(f"\n📖 Open 'CHAPTER_5_RESULTS.md' for complete Chapter 5 results")

if __name__ == "__main__":
    main()