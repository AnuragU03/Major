# PKL Results Comparison Report

## 1. Product Comparison

| Metric | Single Agent | Multi-Agent |
|--------|--------------|-------------|
| Total Products | 155 | 143 |
| Amazon.in Products | 73 | 38 |
| Flipkart Products | 68 | 36 |
| Croma Products | 0 | 40 |
| Reliance Digital Products | 14 | 29 |
| Silhouette Score | -0.2900 | -0.1870 |
| Davies-Bouldin Index | 7.2250 | 9.0040 |
| Cluster Purity (%) | 59.40 | 37.80 |
| N Categories | 8 | 10 |

## 2. Time Performance

| Metric | Single Agent | Multi-Agent |
|--------|--------------|-------------|
| Total Time (seconds) | 0.00 | 2664.28 |
| Avg Time/Query (sec) | 0.00 | 66.61 |
| Products/Second | 0.0000 | 0.0537 |
| Speedup Factor | 1.00x | 0.00x |

## 3. Power Consumption

| Metric | Single Agent | Multi-Agent |
|--------|--------------|-------------|
| Avg CPU Usage (%) | 0.0 | 0.0 |
| Avg CPU Power (W) | 0.00 | 2.80 |
| Avg Memory (%) | 0.0 | 52.5 |
| Total Energy (kWh) | 0.000000 | 0.009474 |
| CO₂ Emissions (kg) | 0.000000 | 0.007769 |

## 4. Accuracy & Reliability

| Metric | Single Agent | Multi-Agent |
|--------|--------------|-------------|
| Total Errors | 0 | 0 |
| Success Rate (%) | 100.0 | 100.0 |
| Avg Products/Query | 0.00 | 3.58 |
| Source Coverage | 3/4 | 4/4 |

## 5. UMAP Visualizations

- Single Agent: `umap_single_agent_pkl.png`
- Multi-Agent: `umap_multi_agent_pkl.png`

## 6. Conclusion

**Time comparison not available** (single agent timing data missing).

- **Products Scraped:** Multi-Agent got -12 more products
- **Source Coverage:** Multi-Agent covers 4/4 sources vs Single Agent 3/4
- **Success Rate:** Single=100.0% vs Multi=100.0%
