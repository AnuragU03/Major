# PKL Detailed Analysis Report

Generated: 2025-12-31 17:22:16


## 1. Executive Summary


| Metric | Single Agent | Multi-Agent | Winner |

|--------|--------------|-------------|--------|

| Total Products | 155 | 143 | Single Agent ✅ |

| Data Accuracy | 87.1% | 90.2% | Multi-Agent ✅ |

| Avg Latency (s) | 0.00 | 66.58 | Multi-Agent ✅ |

| Throughput (prod/s) | 0.0000 | 0.0537 | Multi-Agent ✅ |


## 2. Latency Analysis


### 2.1 Latency Comparison


| Metric | Single Agent | Multi-Agent |

|--------|--------------|-------------|

| Total Queries | 0 | 40 |

| Avg Latency | 0.00s | 66.58s |

| Min Latency | 0.00s | 49.42s |

| Max Latency | 0.00s | 108.31s |

| Median Latency | 0.00s | 65.65s |

| Std Latency | 0.00s | 9.61s |

| P95 Latency | 0.00s | 81.38s |

| P99 Latency | 0.00s | 101.18s |


## 3. Bandwidth Analysis


| Metric | Single Agent | Multi-Agent |

|--------|--------------|-------------|

| Total Data Scraped | 1034.94 KB | 674.30 KB |

| Products/Second | 0.0000 | 0.0537 |

| KB/Second | 0.00 | 0.25 |

| Avg Product Size | 6837 bytes | 4829 bytes |


## 4. Accuracy & Data Quality


### 4.1 Field Completeness


| Field | Single Agent | Multi-Agent |

|-------|--------------|-------------|

| Name | 100.0% | 100.0% |

| Price | 100.0% | 79.7% |

| Rating | 48.4% | 81.1% |

| Reviews | 0.0% | 2.1% |

| Image | 100.0% | 100.0% |

| Link | 100.0% | 100.0% |

| Description | 56.1% | 74.8% |

| Technical Details | 41.3% | 68.5% |


**Overall Accuracy Score:**

- Single Agent: 87.1%

- Multi-Agent: 90.2%


## 5. Website Performance


### 5.1 Multi-Agent Website Ranking


| Rank | Website | Products | Price Data | Rating Data | Score |

|------|---------|----------|------------|-------------|-------|

| #1 | Amazon.in | 38 | 100.0% | 100.0% | 75.2 |

| #2 | Flipkart | 36 | 100.0% | 94.4% | 72.7 |

| #3 | Croma | 40 | 100.0% | 47.5% | 60.2 |

| #4 | Reliance Digital | 29 | 0.0% | 86.2% | 37.5 |


**🏆 Best Website: Amazon.in**


## 6. Query Analysis


### 6.1 Top 10 Queries (Most Products)


| Rank | Query | Category | Products | Time (s) |

|------|-------|----------|----------|----------|

| 1 | Samsung Galaxy S24 | Smartphones | 4 | 67.3 |

| 2 | Dell Inspiron | Laptops | 4 | 64.9 |

| 3 | HP Pavilion | Laptops | 4 | 66.2 |

| 4 | Lenovo ThinkPad | Laptops | 4 | 66.8 |

| 5 | Apple Watch Series 9 | Smartwatches | 4 | 67.4 |

| 6 | Samsung Galaxy Watch | Smartwatches | 4 | 63.2 |

| 7 | Noise ColorFit | Smartwatches | 4 | 65.3 |

| 8 | iPad | Tablets | 4 | 64.2 |

| 9 | Redmi Pad | Tablets | 4 | 69.5 |

| 10 | Samsung Galaxy Buds | Wireless Earbuds | 4 | 66.0 |


### 6.2 Bottom 10 Queries (Least Products)


| Rank | Query | Category | Products | Time (s) |

|------|-------|----------|----------|----------|

| 1 | Lenovo Tab | Tablets | 3 | 90.0 |

| 2 | AirPods Pro | Wireless Earbuds | 3 | 60.7 |

| 3 | Noise Buds | Wireless Earbuds | 3 | 64.4 |

| 4 | JBL Tune | Headphones | 3 | 65.7 |

| 5 | Sennheiser | Headphones | 3 | 50.3 |

| 6 | Samsung Smart TV 55 | Smart TVs | 3 | 62.4 |

| 7 | LG OLED TV | Smart TVs | 3 | 65.2 |

| 8 | GoPro Hero | Cameras | 3 | 68.5 |

| 9 | Gaming Laptop | Gaming | 3 | 65.9 |

| 10 | Apple HomePod | Smart Speakers | 2 | 49.4 |


### 6.3 Category Performance


| Category | Total Products | Queries | Avg Time (s) |

|----------|----------------|---------|-------------|

| Laptops | 15 | 4 | 65.6 |

| Smartwatches | 15 | 4 | 65.0 |

| Cameras | 15 | 4 | 67.3 |

| Gaming | 15 | 4 | 66.4 |

| Tablets | 14 | 4 | 72.7 |

| Wireless Earbuds | 14 | 4 | 63.2 |

| Headphones | 14 | 4 | 61.5 |

| Smart TVs | 14 | 4 | 78.4 |

| Smart Speakers | 14 | 4 | 65.3 |

| Smartphones | 13 | 4 | 60.4 |


## 7. UMAP Clustering Quality


| Metric | Single Agent | Multi-Agent | Interpretation |

|--------|--------------|-------------|----------------|

| Silhouette Score | -0.2900 | -0.1870 | Higher = Better |

| Davies-Bouldin | 7.2250 | 9.0040 | Lower = Better |

| Cluster Purity | 59.4% | 37.8% | Higher = Better |


**Visualizations:**

- Single Agent: `umap_single_agent_pkl.png`

- Multi-Agent: `umap_multi_agent_pkl.png`


## 8. Conclusion


**Overall Winner: Multi-Agent**


### Key Findings:


1. **Product Count:** Multi-Agent scraped -12 more products (143 vs 155)

2. **Data Quality:** Multi-Agent has better data accuracy (90.2%)

3. **Website Coverage:** Multi-Agent covers 4 websites vs Single Agent 3

4. **Best Query:** "Samsung Galaxy S24" with 4 products

5. **Worst Query:** "Apple HomePod" with 2 products
