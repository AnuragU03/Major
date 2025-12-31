# COMPARISON RESULTS: Single Agent vs Multi-Agent Scraping

*Generated on 2025-12-31 17:20:03*


## Executive Summary

- **Single Agent (Try.py)**: Scraped 142 products in 127.2 minutes

- **Multi-Agent (Parallel)**: Scraped 143 products in 44.4 minutes

- **Speedup**: 2.86x faster with multi-agent approach


## Table 1: Performance Comparison

| Metric | Single Agent | Multi-Agent | Improvement |
|--------|--------------|-------------|-------------|
| Total Products | 142 | 143 | +1 |
| Total Time | 127.2 min | 44.4 min | 2.86x faster |
| Avg Time/Query | 190.8s | 66.6s | 124.2s saved |
| Error Count | 0 | 0 | +0 |


## Table 2: Products by Source

| Source | Single Agent | Multi-Agent |
|--------|--------------|-------------|
| Amazon | 50 | 38 |
| Flipkart | 77 | 36 |
| Croma | 0 | 40 |
| Reliance Digital | 15 | 29 |


## Table 3: Category-wise Results

| Category | Single Agent | Multi-Agent |
|----------|--------------|-------------|
| Smartphones | 17 | 13 |
| Laptops | 16 | 15 |
| Smartwatches | 10 | 15 |
| Tablets | 23 | 14 |
| Wireless Earbuds | 15 | 14 |
| Headphones | 6 | 14 |
| Smart TVs | 16 | 14 |
| Cameras | 9 | 15 |
| Gaming | 10 | 15 |
| Smart Speakers | 20 | 14 |


## Analysis

### Speed Improvement

The multi-agent approach achieved a **2.86x speedup** compared to the single-agent approach. 
This is because the multi-agent system scrapes all 4 platforms (Amazon, Flipkart, Croma, Reliance Digital) 
simultaneously using parallel threads, while the single-agent approach scrapes them sequentially.


### Product Coverage

- Single Agent: 142 total products

- Multi-Agent: 143 total products

- Coverage difference: 1 products


### Error Handling

- Single Agent Errors: 0

- Multi-Agent Errors: 0

- The single-agent approach had fewer errors.


## Table 4: Power Consumption Comparison

| Metric | Single Agent | Multi-Agent | Difference |
|--------|--------------|-------------|------------|
| Avg CPU Usage (%) | 5.5% | 0.0% | -5.5% |
| Avg CPU Power (W) | 2.80W | 2.80W | +0.00W |
| Avg Memory (%) | 63.9% | 52.5% | -11.4% |
| Total Energy (kWh) | 0.027180 | 0.009474 | -0.017706 |
| CO₂ Emissions (kg) | 0.022287 | 0.007769 | -0.014519 |


**Energy Efficiency:**

- Single Agent: 5224 products/kWh

- Multi-Agent: 15094 products/kWh

- Multi-Agent is **2.89x more energy efficient**


## Table 5: UMAP Clustering Quality Comparison

| Metric | Single Agent | Multi-Agent | Interpretation |
|--------|--------------|-------------|----------------|
| Silhouette Score | 0.0000 | 0.0000 | Similar clustering |
| Davies-Bouldin Index | 0.0000 | 0.0000 | Similar separation |
| Cluster Purity (%) | 0.0% | 0.0% | Category coherence |
| Number of Clusters | 0 | 0 | Auto-detected |


**UMAP Visualizations:**

- Single Agent: `umap_single_agent.png`

- Multi-Agent: `umap_multi_agent.png`


## Conclusion

The multi-agent parallel scraping approach demonstrates a **2.86x performance improvement** 
over the traditional single-agent sequential approach. This validates the effectiveness of 
the multi-agent architecture for large-scale e-commerce data collection.


**Power Efficiency:** The multi-agent approach used **65.1% less energy** 
due to shorter execution time despite higher parallel CPU usage.


**Clustering Quality:** Both approaches achieved similar UMAP clustering quality, 
indicating that the multi-agent architecture does not compromise data quality for speed.
