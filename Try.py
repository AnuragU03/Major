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
from tkinter import ttk, scrolledtext, messagebox
from PIL import Image, ImageTk
import webbrowser
import requests
from io import BytesIO
from threading import Thread
from webdriver_manager.chrome import ChromeDriverManager
import networkx as nx
from matplotlib.patches import FancyBboxPatch
import os
from typing import Dict, List, Any, Optional

# ===========================
# Graph Database System
# ===========================

class ProductGraphDatabase:
    """Graph database for products with nodes and relationships"""
    
    def __init__(self):
        self.graph = nx.DiGraph()  # Directed graph
        self.node_types = {
            'Product': [],
            'Brand': [],
            'Category': [],
            'Source': [],
            'PriceRange': [],
            'Rating': []
        }
        
    def add_product_node(self, product):
        """Add product as a node with all relationships"""
        product_id = product.get('id', f"prod_{random.randint(1000, 9999)}")
        
        # Add product node
        self.graph.add_node(product_id,
                           type='Product',
                           name=product.get('name', 'Unknown'),
                           price=product.get('price_numeric', 0),
                           rating=product.get('rating', 'N/A'),
                           source=product.get('source', 'Unknown'),
                           link=product.get('product_link', ''),
                           image=product.get('image_url', ''),
                           timestamp=product.get('timestamp', datetime.now().isoformat()))
        
        self.node_types['Product'].append(product_id)
        
        # Extract and add brand node
        brand = self._extract_brand(product.get('name', ''))
        if brand:
            brand_id = f"brand_{brand.lower().replace(' ', '_')}"
            if not self.graph.has_node(brand_id):
                self.graph.add_node(brand_id, type='Brand', name=brand)
                self.node_types['Brand'].append(brand_id)
            self.graph.add_edge(product_id, brand_id, relationship='HAS_BRAND')
        
        # Add category node
        category = product.get('category', 'Electronics')
        category_id = f"cat_{category.lower().replace(' ', '_')}"
        if not self.graph.has_node(category_id):
            self.graph.add_node(category_id, type='Category', name=category)
            self.node_types['Category'].append(category_id)
        self.graph.add_edge(product_id, category_id, relationship='IN_CATEGORY')
        
        # Add source node
        source = product.get('source', 'Unknown')
        source_id = f"src_{source.lower()}"
        if not self.graph.has_node(source_id):
            self.graph.add_node(source_id, type='Source', name=source)
            self.node_types['Source'].append(source_id)
        self.graph.add_edge(product_id, source_id, relationship='SOLD_BY')
        
        # Add price range node
        price = product.get('price_numeric', 0)
        price_range = self._get_price_range(price)
        price_id = f"price_{price_range.lower().replace(' ', '_')}"
        if not self.graph.has_node(price_id):
            self.graph.add_node(price_id, type='PriceRange', name=price_range)
            self.node_types['PriceRange'].append(price_id)
        self.graph.add_edge(product_id, price_id, relationship='PRICED_IN')
        
        # Add rating node
        rating_str = str(product.get('rating', ''))
        rating_match = re.search(r'(\d+\.?\d*)', rating_str)
        if rating_match:
            rating_val = float(rating_match.group(1))
            rating_category = self._get_rating_category(rating_val)
            rating_id = f"rating_{rating_category.lower().replace(' ', '_')}"
            if not self.graph.has_node(rating_id):
                self.graph.add_node(rating_id, type='Rating', name=rating_category)
                self.node_types['Rating'].append(rating_id)
            self.graph.add_edge(product_id, rating_id, relationship='HAS_RATING')
        
        # Add specification relationships
        if product.get('technical_details'):
            for key, value in product.get('technical_details', {}).items():
                spec_id = f"spec_{key.lower().replace(' ', '_')[:30]}"
                if not self.graph.has_node(spec_id):
                    self.graph.add_node(spec_id, type='Specification', name=key, value=str(value))
                self.graph.add_edge(product_id, spec_id, relationship='HAS_SPEC')
    
    def _extract_brand(self, product_name):
        """Extract brand from product name"""
        common_brands = ['Apple', 'Samsung', 'OnePlus', 'Xiaomi', 'Redmi', 'Realme', 
                        'Oppo', 'Vivo', 'Google', 'Pixel', 'iPhone', 'Galaxy', 
                        'Mi', 'Poco', 'Asus', 'Sony', 'Nokia', 'Motorola', 'LG']
        
        for brand in common_brands:
            if brand.lower() in product_name.lower():
                return brand
        
        # Try to extract first word as brand
        words = product_name.split()
        if words:
            return words[0]
        return None
    
    def _get_price_range(self, price):
        """Categorize price into ranges"""
        if price == 0:
            return "Unknown"
        elif price < 10000:
            return "Budget (< ₹10K)"
        elif price < 25000:
            return "Mid-Range (₹10K-25K)"
        elif price < 50000:
            return "Premium (₹25K-50K)"
        else:
            return "Flagship (> ₹50K)"
    
    def _get_rating_category(self, rating):
        """Categorize rating"""
        if rating >= 4.5:
            return "Excellent (4.5+)"
        elif rating >= 4.0:
            return "Very Good (4.0-4.5)"
        elif rating >= 3.5:
            return "Good (3.5-4.0)"
        elif rating >= 3.0:
            return "Average (3.0-3.5)"
        else:
            return "Below Average (< 3.0)"
    
    def get_graph_stats(self):
        """Get graph statistics"""
        return {
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'products': len(self.node_types['Product']),
            'brands': len(self.node_types['Brand']),
            'categories': len(self.node_types['Category']),
            'sources': len(self.node_types['Source']),
            'price_ranges': len(self.node_types['PriceRange']),
            'ratings': len(self.node_types['Rating'])
        }
    
    def export_to_neo4j_format(self, filename='graph_export_neo4j.json'):
        """Export in Neo4j-compatible format"""
        nodes = []
        edges = []
        
        for node_id, node_data in self.graph.nodes(data=True):
            nodes.append({
                'id': node_id,
                'labels': [node_data.get('type', 'Unknown')],
                'properties': {k: v for k, v in node_data.items() if k != 'type'}
            })
        
        for source, target, edge_data in self.graph.edges(data=True):
            edges.append({
                'source': source,
                'target': target,
                'type': edge_data.get('relationship', 'RELATED_TO'),
                'properties': {k: v for k, v in edge_data.items() if k != 'relationship'}
            })
        
        export_data = {
            'nodes': nodes,
            'edges': edges,
            'metadata': {
                'exported_at': datetime.now().isoformat(),
                'total_nodes': len(nodes),
                'total_edges': len(edges)
            }
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Graph exported to {filename} (Neo4j format)")
        print(f"   • {len(nodes)} nodes")
        print(f"   • {len(edges)} relationships")
        return filename
    
    def export_to_gephi_format(self, nodes_file='graph_nodes.csv', edges_file='graph_edges.csv'):
        """Export in Gephi-compatible CSV format"""
        # Export nodes
        nodes_data = []
        for node_id, node_data in self.graph.nodes(data=True):
            nodes_data.append({
                'Id': node_id,
                'Label': node_data.get('name', node_id),
                'Type': node_data.get('type', 'Unknown'),
                **{k: v for k, v in node_data.items() if k not in ['name', 'type']}
            })
        
        nodes_df = pd.DataFrame(nodes_data)
        nodes_df.to_csv(nodes_file, index=False, encoding='utf-8-sig')
        
        # Export edges
        edges_data = []
        for source, target, edge_data in self.graph.edges(data=True):
            edges_data.append({
                'Source': source,
                'Target': target,
                'Type': edge_data.get('relationship', 'RELATED_TO'),
                'Weight': 1
            })
        
        edges_df = pd.DataFrame(edges_data)
        edges_df.to_csv(edges_file, index=False, encoding='utf-8-sig')
        
        print(f"\n✓ Graph exported to Gephi format")
        print(f"   • Nodes: {nodes_file}")
        print(f"   • Edges: {edges_file}")
        return nodes_file, edges_file
    
    def visualize_graph(self, save_file='product_graph.png'):
        """Visualize the graph with matplotlib"""
        if self.graph.number_of_nodes() == 0:
            print("❌ Graph is empty. No data to visualize.")
            return
        
        plt.figure(figsize=(20, 16))
        
        # Define colors for different node types
        color_map = {
            'Product': '#FF6B6B',      # Red
            'Brand': '#4ECDC4',         # Turquoise
            'Category': '#95E1D3',      # Mint
            'Source': '#FFD93D',        # Yellow
            'PriceRange': '#A8E6CF',    # Light green
            'Rating': '#FFB6B9',        # Pink
            'Specification': '#C7CEEA'  # Lavender
        }
        
        # Get node colors based on type
        node_colors = [color_map.get(self.graph.nodes[node].get('type', 'Unknown'), '#CCCCCC') 
                      for node in self.graph.nodes()]
        
        # Calculate node sizes based on degree (connections)
        node_sizes = [300 + self.graph.degree(node) * 100 for node in self.graph.nodes()]
        
        # Use spring layout for better visualization
        pos = nx.spring_layout(self.graph, k=2, iterations=50, seed=42)
        
        # Draw edges
        nx.draw_networkx_edges(self.graph, pos, alpha=0.2, arrows=True, 
                              arrowsize=10, edge_color='gray', width=1.5)
        
        # Draw nodes
        nx.draw_networkx_nodes(self.graph, pos, node_color=node_colors, 
                              node_size=node_sizes, alpha=0.9, linewidths=2, 
                              edgecolors='black')
        
        # Draw labels (only for non-product nodes to reduce clutter)
        labels = {}
        for node in self.graph.nodes():
            node_type = self.graph.nodes[node].get('type', '')
            if node_type != 'Product' and node_type != 'Specification':
                name = self.graph.nodes[node].get('name', node)
                labels[node] = name[:20]  # Truncate long names
        
        nx.draw_networkx_labels(self.graph, pos, labels, font_size=8, 
                               font_weight='bold', font_color='black')
        
        # Add legend
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                     markerfacecolor=color, markersize=10, 
                                     label=node_type) 
                          for node_type, color in color_map.items()]
        
        plt.legend(handles=legend_elements, loc='upper left', fontsize=10, 
                  framealpha=0.9, title='Node Types')
        
        plt.title('Product Knowledge Graph', fontsize=20, fontweight='bold', pad=20)
        plt.axis('off')
        plt.tight_layout()
        
        # Save
        plt.savefig(save_file, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"\n✓ Graph visualization saved to {save_file}")
        
        # Show
        plt.show()
        
    def get_product_recommendations(self, product_id, top_k=5):
        """Get similar products based on graph connections"""
        if not self.graph.has_node(product_id):
            return []
        
        # Find products with similar connections
        product_neighbors = set(self.graph.neighbors(product_id))
        
        recommendations = []
        for other_product in self.node_types['Product']:
            if other_product != product_id:
                other_neighbors = set(self.graph.neighbors(other_product))
                common_neighbors = product_neighbors.intersection(other_neighbors)
                similarity_score = len(common_neighbors) / max(len(product_neighbors), 1)
                
                if similarity_score > 0:
                    recommendations.append({
                        'product_id': other_product,
                        'similarity': similarity_score,
                        'common_attributes': len(common_neighbors)
                    })
        
        # Sort by similarity
        recommendations.sort(key=lambda x: x['similarity'], reverse=True)
        return recommendations[:top_k]


# ===========================
# Generic RAG Pipeline (adapted from template)
# ===========================

class RAGPipeline:
    """
    Generic Retrieval-Augmented Generation pipeline.
    Storage format: JSON list of dicts at storage_path.
    Uses optional Search API (SerpAPI) and Gemini for LLM extraction.
    """
    def __init__(self, storage_path: str, llm_api_key: str, search_api_key: Optional[str] = None):
        self.storage_path = storage_path
        self.llm_api_key = llm_api_key
        self.search_api_key = search_api_key
        self.knowledge_base = self._load_knowledge_base()

    def _load_knowledge_base(self) -> List[Dict[str, Any]]:
        try:
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return []
        except Exception:
            return []

    def _save_knowledge_base(self):
        try:
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(self.knowledge_base, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[RAGPipeline] Save error: {e}")

    def search(self, query: str) -> List[Dict[str, Any]]:
        # Tier 1: Exact
        exact = self._exact_search(query)
        if exact:
            return self._enrich_and_return(exact, query)
        # Tier 2: Fuzzy
        fuzzy = self._fuzzy_search(query)
        if fuzzy:
            return self._enrich_and_return(fuzzy, query)
        # Tier 3: Fetch and add
        return self._fetch_and_add(query)

    def _exact_search(self, query: str) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        q = query.lower()
        for item in self.knowledge_base:
            if q in str(item.get('name', '')).lower() or q in str(item.get('category', '')).lower():
                results.append(item)
        return results

    def _fuzzy_search(self, query: str) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        q = query.lower()
        for item in self.knowledge_base:
            searchable = ' '.join([
                str(item.get('description', '')),
                str(item.get('tags', '')),
                str(item.get('content', ''))
            ]).lower()
            if q in searchable:
                results.append(item)
        return results

    def _enrich_and_return(self, results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        for item in results:
            try:
                fresh = self._web_enrichment(query, item.get('name', query))
                if isinstance(fresh, dict):
                    item.update(fresh)
            except Exception as e:
                print(f"[RAGPipeline] Enrichment error: {e}")
        self._save_knowledge_base()
        return results

    def _fetch_and_add(self, query: str) -> List[Dict[str, Any]]:
        base = self._fetch_base_data(query)
        if not base:
            return []
        enrichment = self._web_enrichment(query, base.get('name', query))
        new_entry = {**base, **(enrichment or {})}
        self.knowledge_base.append(new_entry)
        self._save_knowledge_base()
        return [new_entry]

    def _fetch_base_data(self, query: str) -> Optional[Dict[str, Any]]:
        # Minimal base data; projects can override to pull from primary sources
        return {
            'name': query,
            'id': query.lower().replace(' ', '_'),
            'timestamp': datetime.now().isoformat(),
            'category': 'generic'
        }

    def _web_enrichment(self, query: str, entity_name: str) -> Dict[str, Any]:
        snippets = self._web_search(f"{entity_name} details information")
        return self._llm_extract(query, snippets)

    def _web_search(self, query: str) -> List[str]:
        if not self.search_api_key:
            return ["Sample data: No search API configured"]
        try:
            url = 'https://serpapi.com/search.json'
            params = {'engine': 'google', 'q': query, 'api_key': self.search_api_key, 'num': 5}
            resp = requests.get(url, params=params, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                return [r.get('snippet', '') for r in data.get('organic_results', []) if r.get('snippet')]
            else:
                print(f"[Search] HTTP {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            print(f"[Search] Error: {e}")
        return []

    def _llm_extract(self, query: str, snippets: List[str]) -> Dict[str, Any]:
        schema = {'key_facts': [], 'attributes': [], 'related_items': []}
        prompt = (
            f"You are a data extraction expert.\n"
            f"From these web snippets about \"{query}\", extract structured information.\n\n"
            "Return ONLY valid JSON with these fields:\n"
            "- key_facts: List of important facts\n"
            "- attributes: List of key attributes/characteristics\n"
            "- related_items: List of related or similar items\n\n"
            f"Snippets:\n{chr(10).join(snippets[:5])}\n\n"
            "Return ONLY the JSON object, no explanation."
        )
        if not self.llm_api_key:
            return schema
        try:
            url = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent'
            headers = {'Content-Type': 'application/json', 'X-goog-api-key': self.llm_api_key}
            data = {'contents': [{'parts': [{'text': prompt}]}]}
            resp = requests.post(url, headers=headers, json=data, timeout=30)
            if resp.status_code == 200:
                result = resp.json()
                text = result['candidates'][0]['content']['parts'][0]['text']
                import re as _re
                m = _re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
                json_text = m.group(1) if m else text
                try:
                    parsed = json.loads(json_text)
                    if isinstance(parsed, dict):
                        return parsed
                except Exception:
                    pass
        except Exception as e:
            print(f"[LLM] Error: {e}")
        return schema


def build_generic_rag() -> RAGPipeline:
    """Factory to build a pipeline using environment variables."""
    llm_key = os.getenv('GEMINI_API_KEY', '')
    serp_key = os.getenv('SERPAPI_KEY', '')
    storage = 'knowledge_base.json'
    return RAGPipeline(storage_path=storage, llm_api_key=llm_key, search_api_key=serp_key)


# ===========================
# RAG Storage System (Enhanced)
# ===========================

class ProductRAGStorage:
    """RAG-based storage system for product data with semantic search"""
    
    def __init__(self, storage_file='product_rag_db.pkl'):
        self.storage_file = storage_file
        self.products = []
        self.vectorizer = TfidfVectorizer(max_features=500)
        self.vectors = None
        self.graph_db = ProductGraphDatabase()  # Add graph database
        self.load_storage()
    
    def add_product(self, product_data):
        """Add product to RAG storage"""
        product_data['timestamp'] = datetime.now().isoformat()
        product_data['id'] = f"{product_data['source']}_{len(self.products)}"
        self.products.append(product_data)
        self.graph_db.add_product_node(product_data)  # Add to graph
        self._update_vectors()
        self.save_storage()
    
    def add_products_batch(self, products_list):
        """Add multiple products at once"""
        for product in products_list:
            product['timestamp'] = datetime.now().isoformat()
            product['id'] = f"{product.get('source', 'unknown')}_{len(self.products)}"
            self.products.append(product)
            self.graph_db.add_product_node(product)  # Add to graph
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
        
        # Auto-export to CSV after every save
        self.auto_export_to_csv()
    
    def auto_export_to_csv(self):
        """Automatically export to CSV with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export to timestamped file
        timestamped_file = f'products_export_{timestamp}.csv'
        
        # Also maintain a "latest" file
        latest_file = 'products_export_latest.csv'
        
        if not self.products:
            return
        
        # Flatten nested dictionaries for CSV
        flattened = []
        for p in self.products:
            flat = p.copy()
            
            # Flatten technical_details
            if 'technical_details' in flat and isinstance(flat['technical_details'], dict):
                for k, v in flat['technical_details'].items():
                    # Clean key name for CSV column
                    clean_key = k.replace(' - ', '_').replace(' ', '_').replace('/', '_')
                    flat[f'spec_{clean_key}'] = v
                del flat['technical_details']
            
            # Flatten additional_info
            if 'additional_info' in flat and isinstance(flat['additional_info'], dict):
                for k, v in flat['additional_info'].items():
                    clean_key = k.replace(' - ', '_').replace(' ', '_').replace('/', '_')
                    flat[f'info_{clean_key}'] = v
                del flat['additional_info']
            
            # Convert features list to string
            if 'features' in flat and isinstance(flat['features'], list):
                flat['features'] = ' | '.join(flat['features'])
            
            flattened.append(flat)
        
        df = pd.DataFrame(flattened)
        
        # Save both files
        df.to_csv(timestamped_file, index=False, encoding='utf-8-sig')
        df.to_csv(latest_file, index=False, encoding='utf-8-sig')
        
        print(f"   📁 Auto-exported to: {timestamped_file}")
        print(f"   📁 Updated: {latest_file}")
    
    def load_storage(self):
        """Load storage from disk"""
        try:
            with open(self.storage_file, 'rb') as f:
                data = pickle.load(f)
                self.products = data.get('products', [])
                self.vectorizer = data.get('vectorizer', TfidfVectorizer(max_features=500))
                self.vectors = data.get('vectors')
            print(f"Loaded {len(self.products)} products from storage")
            
            # Rebuild graph database from loaded products
            if self.products:
                print(f"🕸️  Rebuilding knowledge graph from {len(self.products)} products...")
                for product in self.products:
                    self.graph_db.add_product_node(product)
                print(f"✓ Graph rebuilt: {self.graph_db.get_graph_stats()['total_nodes']} nodes, {self.graph_db.get_graph_stats()['total_edges']} edges")
        except FileNotFoundError:
            print("No existing storage found, starting fresh")
        except Exception as e:
            print(f"Error loading storage: {e}")
    
    def export_to_csv(self, filename='products_export.csv'):
        """Export products to CSV with enhanced GraphRAG-like structure"""
        if not self.products:
            print("No products to export")
            return
        
        # Create multiple CSV files for GraphRAG structure
        base_name = filename.replace('.csv', '')
        
        # 1. Main products file
        flattened = []
        for p in self.products:
            flat = p.copy()
            
            # Flatten technical_details
            if 'technical_details' in flat and isinstance(flat['technical_details'], dict):
                for k, v in flat['technical_details'].items():
                    clean_key = k.replace(' - ', '_').replace(' ', '_').replace('/', '_')
                    flat[f'spec_{clean_key}'] = v
                del flat['technical_details']
            
            # Flatten additional_info
            if 'additional_info' in flat and isinstance(flat['additional_info'], dict):
                for k, v in flat['additional_info'].items():
                    clean_key = k.replace(' - ', '_').replace(' ', '_').replace('/', '_')
                    flat[f'info_{clean_key}'] = v
                del flat['additional_info']
            
            # Convert features list to string
            if 'features' in flat and isinstance(flat['features'], list):
                flat['features'] = ' | '.join(flat['features'])
            
            flattened.append(flat)
        
        # Export main products
        df_products = pd.DataFrame(flattened)
        df_products.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"✓ Exported {len(self.products)} products to {filename}")
        
        # 2. Export specifications as separate graph (for GraphRAG)
        specs_data = []
        for p in self.products:
            product_id = p.get('id', '')
            product_name = p.get('name', '')
            
            if 'technical_details' in p and isinstance(p['technical_details'], dict):
                for spec_name, spec_value in p['technical_details'].items():
                    specs_data.append({
                        'product_id': product_id,
                        'product_name': product_name,
                        'specification_name': spec_name,
                        'specification_value': spec_value,
                        'source': p.get('source', '')
                    })
        
        if specs_data:
            df_specs = pd.DataFrame(specs_data)
            specs_file = f"{base_name}_specifications.csv"
            df_specs.to_csv(specs_file, index=False, encoding='utf-8-sig')
            print(f"✓ Exported {len(specs_data)} specifications to {specs_file}")
        
        # 3. Export relationships (product-category-source graph)
        relationships = []
        for p in self.products:
            relationships.append({
                'product_id': p.get('id', ''),
                'product_name': p.get('name', ''),
                'category': p.get('category', ''),
                'source': p.get('source', ''),
                'price': p.get('price_numeric', 0),
                'rating': p.get('rating', ''),
                'timestamp': p.get('timestamp', '')
            })
        
        df_relations = pd.DataFrame(relationships)
        relations_file = f"{base_name}_relationships.csv"
        df_relations.to_csv(relations_file, index=False, encoding='utf-8-sig')
        print(f"✓ Exported {len(relationships)} relationships to {relations_file}")
    
    def clear_storage(self):
        """Clear all stored products"""
        self.products = []
        self.vectors = None
        self.save_storage()
    
    def get_product_graph(self):
        """Get product relationships as a graph structure for GraphRAG"""
        graph = {
            'nodes': [],
            'edges': []
        }
        
        # Create nodes for products
        for p in self.products:
            graph['nodes'].append({
                'id': p.get('id'),
                'type': 'product',
                'name': p.get('name'),
                'category': p.get('category'),
                'source': p.get('source'),
                'price': p.get('price_numeric', 0)
            })
        
        # Create nodes for categories and sources
        categories = set(p.get('category') for p in self.products)
        sources = set(p.get('source') for p in self.products)
        
        for cat in categories:
            if cat:
                graph['nodes'].append({
                    'id': f'cat_{cat}',
                    'type': 'category',
                    'name': cat
                })
        
        for src in sources:
            if src:
                graph['nodes'].append({
                    'id': f'src_{src}',
                    'type': 'source',
                    'name': src
                })
        
        # Create edges (relationships)
        for p in self.products:
            product_id = p.get('id')
            
            # Product -> Category edge
            if p.get('category'):
                graph['edges'].append({
                    'from': product_id,
                    'to': f"cat_{p.get('category')}",
                    'type': 'belongs_to'
                })
            
            # Product -> Source edge
            if p.get('source'):
                graph['edges'].append({
                    'from': product_id,
                    'to': f"src_{p.get('source')}",
                    'type': 'sold_by'
                })
        
        return graph
    
    def export_graph_to_json(self, filename='product_graph.json'):
        """Export the product graph as JSON for GraphRAG visualization"""
        graph = self.get_product_graph()
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(graph, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Exported product graph to {filename}")
        print(f"  - {len([n for n in graph['nodes'] if n['type'] == 'product'])} product nodes")
        print(f"  - {len([n for n in graph['nodes'] if n['type'] == 'category'])} category nodes")
        print(f"  - {len([n for n in graph['nodes'] if n['type'] == 'source'])} source nodes")
        print(f"  - {len(graph['edges'])} edges")

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
        
        print(f"      Searching for Amazon product details...")
        
        # Search for sections containing keywords: "detail", "product information", "specification"
        detail_keywords = ['detail', 'product information', 'specification', 'technical', 'product details']
        
        # Method 1: Try specific IDs first
        # Technical Details by ID
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
            print(f"      Technical Details table not found by ID, trying alternative methods...")
        
        # Method 2: Search by text for sections containing "detail" or "information"
        if not details['technical_details']:
            try:
                # Find all tables and sections
                all_tables = driver.find_elements(By.TAG_NAME, "table")
                print(f"      Found {len(all_tables)} tables on page, searching for details...")
                
                for table in all_tables:
                    try:
                        # Check if table or its parent contains detail keywords
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
        
        # Additional Information
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
        
        # Product Description - search for "description" keyword
        desc_found = False
        try:
            desc_element = driver.find_element(By.ID, "feature-bullets")
            features = desc_element.find_elements(By.TAG_NAME, "li")
            print(f"      Found {len(features)} product features")
            for feat in features:
                text = feat.text.strip()
                if text:
                    details['features'].append(text)
            details['description'] = ' | '.join(details['features'])
            desc_found = True
        except:
            pass
        
        # Alternative description locations - search by keyword
        if not desc_found:
            try:
                desc = driver.find_element(By.ID, "productDescription")
                details['description'] = desc.text.strip()
                print(f"      Found product description")
                desc_found = True
            except:
                pass
        
        # Method 3: Search all divs for "description" or "detail" text
        if not desc_found:
            try:
                all_divs = driver.find_elements(By.TAG_NAME, "div")
                for div in all_divs:
                    try:
                        # Check class and id for description keywords
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
        
        print(f"      Amazon extraction complete: {len(details['technical_details'])} specs, "
              f"{len(details['additional_info'])} additional info, {len(details['features'])} features")
        
        return details
        
    except Exception as e:
        print(f"Error scraping Amazon details: {e}")
        import traceback
        traceback.print_exc()
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
        
        print(f"      Searching for Flipkart product details...")
        
        # Keywords to search for in Flipkart pages
        spec_keywords = ['specification', 'specifications', 'specs']
        desc_keywords = ['description', 'product description', 'about']
        detail_keywords = ['product detail', 'product details', 'details', 'product information']
        
        # Method 1: Try to find elements by searching text content
        try:
            # Find all headings and check for keywords
            all_headings = driver.find_elements(By.CSS_SELECTOR, "div, span, h1, h2, h3")
            spec_sections = []
            
            for heading in all_headings:
                try:
                    text = heading.text.strip().lower()
                    
                    # Check if heading contains specification keywords
                    if any(keyword in text for keyword in spec_keywords + detail_keywords):
                        # Found a specification section
                        parent = heading
                        for _ in range(3):  # Go up 3 levels to find the container
                            try:
                                parent = parent.find_element(By.XPATH, "..")
                                # Check if this parent has table rows
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
                    # Try to get section title
                    section_title = ""
                    try:
                        title_elem = section.find_element(By.CSS_SELECTOR, "div._4BJ2V\\+, div._1AtVbE, div._3dtsli, span, h3")
                        section_title = title_elem.text.strip()
                    except:
                        section_title = "Specifications"
                    
                    # Get rows
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
        
        # Method 2: Try multiple CSS selectors for specifications sections
        if not details['technical_details']:
            spec_section_selectors = [
                "div._9V0sS6",  # New Flipkart layout
                "div.GNDEQ-",
                "div._1s76Cw",
                "div._3dtsli",
                "div.aMraIH",
                "div[class*='spec']",  # Any class containing 'spec'
                "div[class*='detail']"  # Any class containing 'detail'
            ]
            
            spec_sections = []
            for selector in spec_section_selectors:
                try:
                    sections = driver.find_elements(By.CSS_SELECTOR, selector)
                    if sections:
                        spec_sections = sections
                        print(f"      Found {len(sections)} spec sections using selector: {selector}")
                        break
                except:
                    continue
            
            # Extract specifications from found sections
            if spec_sections:
                for section_idx, section in enumerate(spec_sections):
                    try:
                        # Try to find section title
                        section_title = ""
                        title_selectors = [
                            "div._4BJ2V\\+",  # New layout
                            "div._1AtVbE",
                            "div._3dtsli", 
                            "div._2RngUh",
                            "span"
                        ]
                        
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
                        
                        print(f"      Processing section: {section_title}")
                        
                        # Try to find specification rows
                        row_selectors = [
                            "tr.WJdYP6",  # New layout
                            "tr._2-N8s",
                            "li.W5FkOm",
                            "div.row",
                            "tr"  # Generic tr
                        ]
                        
                        rows = []
                        for row_sel in row_selectors:
                            try:
                                found_rows = section.find_elements(By.CSS_SELECTOR, row_sel)
                                if found_rows:
                                    rows = found_rows
                                    print(f"        Found {len(rows)} rows using selector: {row_sel}")
                                    break
                            except:
                                continue
                        
                        # Extract key-value pairs from rows
                        for row in rows:
                            try:
                                # Try multiple selectors for table cells
                                cell_selectors = [
                                    "td.URwL2w",  # New layout - key
                                    "td._7eSDEY",  # New layout - value
                                    "td._2H-kL",
                                    "td._2vIOIi",
                                    "td",  # Generic td
                                    "li._21Ahn-"
                                ]
                                
                                # Try to get key and value
                                cells = []
                                for cell_sel in cell_selectors:
                                    try:
                                        found_cells = row.find_elements(By.CSS_SELECTOR, cell_sel)
                                        if found_cells:
                                            cells.extend(found_cells)
                                            break  # Stop after first successful selector
                                    except:
                                        continue
                                
                                # If we have at least 2 cells, treat as key-value
                                if len(cells) >= 2:
                                    key = cells[0].text.strip()
                                    value = cells[1].text.strip()
                                    if key and value and len(key) < 100:
                                        details['technical_details'][f"{section_title} - {key}"] = value
                                        print(f"          {key}: {value[:50]}")
                                elif len(cells) == 1:
                                    # Single cell - might be a feature/highlight
                                    text = cells[0].text.strip()
                                    if text and len(text) > 3:
                                        details['features'].append(text)
                            except Exception as row_error:
                                continue
                    
                    except Exception as section_error:
                        print(f"      Error in section: {section_error}")
                        continue
        
        # Description - search by keywords
        desc_selectors = [
            "div._1mXcCf",
            "div._2418kt",
            "div.yN\\+eNk",
            "div._2RngUh p",
            "div.product-description",
            "div[class*='description']",  # Any class containing 'description'
            "div[class*='desc']"  # Any class containing 'desc'
        ]
        
        for desc_sel in desc_selectors:
            try:
                desc_element = driver.find_element(By.CSS_SELECTOR, desc_sel)
                details['description'] = desc_element.text.strip()
                if details['description']:
                    print(f"      Found description using: {desc_sel}")
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
        
        # Highlights/Features - try multiple selectors
        highlight_selectors = [
            "li._21Ahn-",
            "li.WJdYP6",
            "ul._1D2qrc li",
            "div._2418kt ul li",
            "li[class*='highlight']",  # Any class containing 'highlight'
            "li[class*='feature']"  # Any class containing 'feature'
        ]
        
        for hl_sel in highlight_selectors:
            try:
                highlights = driver.find_elements(By.CSS_SELECTOR, hl_sel)
                if highlights:
                    for hl in highlights:
                        text = hl.text.strip()
                        if text and len(text) > 3 and text not in details['features']:
                            details['features'].append(text)
                    if details['features']:
                        print(f"      Found {len(details['features'])} features using: {hl_sel}")
                        break
            except:
                continue
        
        # Combine features into description if description is empty
        if not details['description'] and details['features']:
            details['description'] = ' | '.join(details['features'])
        
        # Print summary
        print(f"      Flipkart extraction complete: {len(details['technical_details'])} specs, {len(details['features'])} features")
        
        return details
        
    except Exception as e:
        print(f"Error scraping Flipkart details: {e}")
        import traceback
        traceback.print_exc()
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
    
    # Try up to 2 times if page doesn't load
    for attempt in range(2):
        try:
            if attempt > 0:
                print(f"  Retry attempt {attempt + 1}/2...")
            
            url = f"https://www.flipkart.com/search?q={product_name.replace(' ', '+')}"
            driver.get(url)
            time.sleep(random.uniform(7, 10))  # Increased wait time
            
            try:
                close_btn = WebDriverWait(driver, 5).until(  # Increased from 3 to 5
                    EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), '✕')]"))
                )
                close_btn.click()
                time.sleep(2)  # Increased from 1 to 2
            except:
                pass
            
            # Try to wait for products to load
            WebDriverWait(driver, 30).until(  # Increased from 15 to 30 seconds
                EC.presence_of_element_located((By.CSS_SELECTOR, "div[data-id], div.CGtC98, div.tUxRFH, div._1AtVbE"))
            )
            time.sleep(5)  # Increased from 3 to 5
            
            # If we get here, page loaded successfully
            break
            
        except TimeoutException:
            if attempt == 0:
                print(f"  Flipkart: First attempt timed out, retrying...")
                continue
            else:
                print(f"  Flipkart: Results took too long to load after 2 attempts (waited 30 seconds each)")
                print(f"  Trying to continue with whatever loaded...")
        except Exception as e:
            print(f"  Flipkart error on attempt {attempt + 1}: {str(e)}")
            if attempt == 0:
                continue
            else:
                return []
    
    products = []
    
    # Find product containers
    item_selectors = ["div[data-id]", "div.CGtC98", "div.tUxRFH", "div._1AtVbE"]
    items = []
    
    for selector in item_selectors:
        items = driver.find_elements(By.CSS_SELECTOR, selector)
        if items:
            print(f"Flipkart: Found {len(items)} items with selector '{selector}', scraping details for top {max_products}...")
            break
    
    if not items:
        print("Flipkart: Could not find product items")
        return []
    
    processed_count = 0
    
    for idx, item in enumerate(items):
        if processed_count >= max_products:
            break
            
        try:
            print(f"  Scraping Flipkart product {processed_count + 1}/{max_products}...")
            
            # Get product link first
            product_link = ""
            try:
                link_elem = item.find_element(By.CSS_SELECTOR, "a[href*='/p/'], a.wjcEIp, a.CGtC98, a.rPDeLR")
                href = link_elem.get_attribute("href")
                if href and ('/p/' in href or '/product/' in href):
                    # Clean the link
                    if '?' in href:
                        href = href.split('?')[0]
                    product_link = href if href.startswith('http') else f"https://www.flipkart.com{href}"
                    print(f"    Found link: {product_link[:60]}...")
            except:
                print(f"    No valid product link found, skipping...")
                continue
            
            if not product_link or 'search' in product_link:
                continue
            
            # Get product name - try multiple selectors
            name = ""
            name_selectors = [
                "div.KzDlHZ",
                "div.wjcEIp", 
                "a.wjcEIp",
                "div.syl9yP",
                "a.VJA3rP",
                "div._2WkVRV",
                "a.IRpwTa"
            ]
            
            for sel in name_selectors:
                try:
                    name_elem = item.find_element(By.CSS_SELECTOR, sel)
                    name = name_elem.text.strip()
                    # Clean up any numbering from the name
                    name = re.sub(r'^\d+\.\s*', '', name)
                    if name and len(name) > 5 and not name.startswith('₹'):
                        print(f"    Found name: {name[:60]}")
                        break
                except:
                    continue
            
            if not name or len(name) < 5:
                print(f"    Failed to extract valid name, skipping...")
                continue
            
            # Get price - try multiple selectors
            price_text = "0"
            price_selectors = [
                "div.Nx9bqj",
                "div._30jeq3", 
                "div._1_WHN1",
                "div.hl05eU",
                "div._25b18c"
            ]
            
            for sel in price_selectors:
                try:
                    price_elem = item.find_element(By.CSS_SELECTOR, sel)
                    price_text = price_elem.text.strip()
                    if price_text and '₹' in price_text:
                        print(f"    Found price: {price_text}")
                        break
                except:
                    continue
            
            # Get image
            image_url = ""
            try:
                img_elem = item.find_element(By.CSS_SELECTOR, "img")
                image_url = img_elem.get_attribute("src")
            except:
                pass
            
            # Get rating
            rating = ""
            try:
                rating_elem = item.find_element(By.CSS_SELECTOR, "div.XQDdHH, div._3LWZlK, div.Y1HWO0")
                rating = rating_elem.text.strip()
            except:
                pass
            
            # Get reviews
            reviews = ""
            try:
                reviews_elem = item.find_element(By.CSS_SELECTOR, "span._2_R_DZ, span.Wphh3N")
                reviews = reviews_elem.text.strip()
            except:
                pass
            
            # Open product page in new tab
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
            processed_count += 1
            print(f"    ✓ Successfully scraped product {processed_count}")
            
        except Exception as e:
            print(f"  ✗ Error on Flipkart product {processed_count + 1}: {e}")
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
    """Keep only phone-related products - improved to be more lenient"""
    if not products:
        return products

    # Extract meaningful search terms (remove generic words)
    generic_terms = {"mobile", "phone", "phones", "smartphone", "smartphones", "cell", "the", "a", "an", "best"}
    tokens = [t for t in re.split(r'\W+', search_term.lower()) if t and t not in generic_terms]

    # Keywords that indicate it's a phone product
    phone_include = {"phone", "mobile", "smartphone", "iphone", "galaxy", "pixel", "oneplus", "redmi", "realme", "oppo", "vivo"}

    # Only exclude clear accessories/non-phone items
    exclude_keywords = [
        "case", "cover", "bumper", "back cover", "protective case",
        "charger", "cable", "adapter", "power adapter",
        "tempered glass", "screen guard", "screen protector", "glass protector",
        "pouch", "strap", "stand", "holder",
        "battery pack", "lens protector", "camera protector",
        "headphone", "earphone", "earbuds", "airpods",
        "powerbank", "power bank", "charging cable",
        "ring holder", "popsocket", "pop socket"
    ]

    filtered = []
    for p in products:
        title = p.get("name", "").lower()
        if not title:
            continue

        # Check if it's clearly an accessory (must match full phrase for more accuracy)
        is_accessory = False
        for ex in exclude_keywords:
            # Only exclude if the keyword is a clear match
            if ex in title:
                # But don't exclude if it also has phone keywords (e.g., "phone with case")
                if not any(phone_kw in title for phone_kw in phone_include):
                    is_accessory = True
                    break
        
        if is_accessory:
            continue

        # If search has specific tokens (like "iphone", "16"), product must contain them
        if tokens:
            # Check if product name contains the search tokens
            if any(tok in title for tok in tokens):
                filtered.append(p)
                continue
            # Or if it contains general phone keywords
            elif any(f in title for f in phone_include):
                filtered.append(p)
                continue
        else:
            # No specific tokens - keep anything with phone keywords
            if any(f in title for f in phone_include):
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
        
        # Add indicator for detailed specs
        if row.get('technical_details') and len(row.get('technical_details', {})) > 0:
            spec_count = len(row['technical_details'])
            info_parts.append(f"📋 {spec_count} Specs Available")
        
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
        
        # Technical Details Button - Show if any detailed data exists
        has_details = (row.get('technical_details') or row.get('additional_info') or 
                      row.get('features') or (row.get('description') and len(row.get('description', '')) > 200))
        
        if has_details:
            def show_details(product_row=row):
                details_window = tk.Toplevel(root)
                details_window.title(f"Details - {product_row['name'][:50]}")
                details_window.geometry("900x700")
                
                # Add scrollbar to details window
                details_frame = ttk.Frame(details_window)
                details_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
                
                text_widget = scrolledtext.ScrolledText(details_frame, wrap=tk.WORD, font=("Courier", 10))
                text_widget.pack(fill=tk.BOTH, expand=True)
                
                # Header
                details_text = "="*90 + "\n"
                details_text += f"PRODUCT DETAILS - {product_row['source']}\n"
                details_text += "="*90 + "\n\n"
                
                details_text += f"Product Name: {product_row['name']}\n"
                details_text += f"Price: ₹{product_row.get('price_numeric', 0):.0f}\n"
                details_text += f"Source: {product_row['source']}\n"
                details_text += f"Category: {product_row.get('category', 'N/A')}\n"
                
                if product_row.get('rating'):
                    details_text += f"Rating: {product_row['rating']}\n"
                if product_row.get('reviews'):
                    details_text += f"Reviews: {product_row['reviews']}\n"
                if product_row.get('availability'):
                    details_text += f"Availability: {product_row['availability']}\n"
                
                details_text += "\n" + "="*90 + "\n\n"
                
                # Technical Details (Specifications)
                if product_row.get('technical_details') and isinstance(product_row['technical_details'], dict):
                    if len(product_row['technical_details']) > 0:
                        details_text += "📋 SPECIFICATIONS:\n" + "-"*90 + "\n"
                        for key, value in product_row['technical_details'].items():
                            # Handle long values
                            if len(str(value)) > 60:
                                details_text += f"{key}:\n  {value}\n"
                            else:
                                details_text += f"{key:45} : {value}\n"
                        details_text += "\n"
                
                # Additional Information
                if product_row.get('additional_info') and isinstance(product_row['additional_info'], dict):
                    if len(product_row['additional_info']) > 0:
                        details_text += "ℹ️  ADDITIONAL INFORMATION:\n" + "-"*90 + "\n"
                        for key, value in product_row['additional_info'].items():
                            if len(str(value)) > 60:
                                details_text += f"{key}:\n  {value}\n"
                            else:
                                details_text += f"{key:45} : {value}\n"
                        details_text += "\n"
                
                # Features/Highlights
                if product_row.get('features') and isinstance(product_row['features'], list):
                    if len(product_row['features']) > 0:
                        details_text += "✨ FEATURES & HIGHLIGHTS:\n" + "-"*90 + "\n"
                        for idx, feat in enumerate(product_row['features'], 1):
                            details_text += f"{idx}. {feat}\n"
                        details_text += "\n"
                
                # Full Description
                if product_row.get('description'):
                    details_text += "📝 DESCRIPTION:\n" + "-"*90 + "\n"
                    details_text += product_row['description'] + "\n\n"
                
                # Product Link
                if product_row.get('product_link'):
                    details_text += "🔗 PRODUCT LINK:\n" + "-"*90 + "\n"
                    details_text += product_row['product_link'] + "\n\n"
                
                details_text += "="*90 + "\n"
                
                text_widget.insert(tk.END, details_text)
                text_widget.config(state=tk.DISABLED)
                
                # Add a copy button
                def copy_to_clipboard():
                    details_window.clipboard_clear()
                    details_window.clipboard_append(details_text)
                    copy_btn.config(text="✓ Copied!", bg="#27ae60")
                    details_window.after(2000, lambda: copy_btn.config(text="📋 Copy Details", bg="#95a5a6"))
                
                copy_btn = tk.Button(details_window, text="📋 Copy Details", 
                                    command=copy_to_clipboard, bg="#95a5a6", fg="white",
                                    font=("Arial", 10, "bold"), padx=15, pady=5)
                copy_btn.pack(pady=5)
            
            details_btn = tk.Button(product_frame, text="📄 View Full Details", 
                                   command=show_details, bg="#3498db", fg="white",
                                   font=("Arial", 9, "bold"), cursor="hand2", padx=10, pady=3)
            details_btn.grid(row=4, column=1, sticky=tk.W, padx=10, pady=5)
        
        # Product link
        if row.get('product_link'):
            def open_link(url=row['product_link']):
                webbrowser.open(url)
            
            link_label = tk.Label(product_frame, text="🔗 View on Website", 
                                font=("Arial", 9, "underline"), fg="blue",
                                bg=product_frame['bg'], cursor="hand2")
            link_label.grid(row=5, column=1, sticky=tk.W, padx=10, pady=3)
            link_label.bind("<Button-1>", lambda e, url=row['product_link']: webbrowser.open(url))
    
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
        messagebox.showinfo("Export", "Data exported to products_detailed_export.csv")
    
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
    
    # Only apply phone filter if user is specifically searching for phones
    phone_search_terms = ['phone', 'mobile', 'smartphone', 'iphone', 'galaxy', 'pixel', 'oneplus', 'redmi']
    is_phone_search = any(term in product_name.lower() for term in phone_search_terms)
    
    if is_phone_search:
        before_count = len(all_products)
        all_products = filter_only_phones(all_products, product_name)
        after_count = len(all_products)
        if before_count != after_count:
            print(f"\n📱 Phone Filter: Removed {before_count - after_count} accessories, kept {after_count} phone products")
    else:
        print(f"\n✓ No phone filter applied - keeping all {len(all_products)} products")

    if not all_products:
        print("\n❌ No products found after filtering.")
        return None

    print(f"\n{'='*70}")
    print("STEP 3: Storing Products in RAG Database...")
    print("-"*70)
    rag_storage.add_products_batch(all_products)
    print(f"✓ Successfully stored {len(all_products)} products with full details")
    print(f"✓ Auto-saved to database: {rag_storage.storage_file}")
    
    print(f"\n{'='*70}")
    print("STEP 4: Generating Comprehensive Analysis Report...")
    print("-"*70)
    create_detailed_report(rag_storage)
    
    df = pd.DataFrame(all_products)
    
    # Filter out products with 0 price before sorting
    df_valid_price = df[df['price_numeric'] > 0]
    if not df_valid_price.empty:
        df = df_valid_price
    
    df = df.sort_values(by="price_numeric")
    
    print(f"{'='*70}")
    print("AGENT EXECUTION COMPLETED SUCCESSFULLY!".center(70))
    print(f"{'='*70}")
    print(f"\n📊 Total Products Found: {len(df)}")
    if not df.empty and 'price_numeric' in df.columns:
        valid_prices = df[df['price_numeric'] > 0]
        if not valid_prices.empty:
            print(f"💰 Lowest Price: ₹{valid_prices['price_numeric'].min():.0f}")
            print(f"💰 Highest Price: ₹{valid_prices['price_numeric'].max():.0f}")
    detailed_count = sum(1 for p in all_products if p.get('technical_details') or p.get('additional_info'))
    print(f"📝 Products with Full Details: {detailed_count}/{len(df)}")
    
    # Show source breakdown
    if 'source' in df.columns:
        print(f"\n📦 By Source:")
        for source in df['source'].unique():
            count = len(df[df['source'] == source])
            print(f"   {source}: {count} products")
    
    print(f"\n{'='*70}\n")
    
    return df


# ===========================
# Smart Search Function
# ===========================

def smart_search_products(product_name, rag_storage, max_products=5, force_new_scrape=False):
    """
    Smart search that checks database first, only scrapes if needed.
    
    Args:
        product_name: Product to search for
        rag_storage: ProductRAGStorage instance
        max_products: Max products per source for new scrapes
        force_new_scrape: If True, always scrape new data
    
    Returns:
        DataFrame with results and boolean indicating if new scrape was done
    """
    print(f"\n{'='*70}")
    print(f"🔍 SMART PRODUCT SEARCH".center(70))
    print(f"{'='*70}")
    print(f"Query: {product_name}")
    
    # Step 1: Check database first (unless forced)
    if not force_new_scrape:
        print(f"\n📊 Step 1: Checking GraphRAG Database...")
        print("-"*70)
        
        db_results = rag_storage.semantic_search(product_name, top_k=20)
        
        if db_results:
            print(f"✓ Found {len(db_results)} matching products in database!")
            
            # Analyze freshness
            result_df = pd.DataFrame(db_results)
            
            # Count products per source
            source_counts = result_df['source'].value_counts().to_dict()
            print(f"\n📦 Database contains:")
            for source, count in source_counts.items():
                print(f"   • {source}: {count} products")
            
            # Check if we have enough diverse results
            has_both_sources = len(source_counts) >= 2
            has_enough_products = len(db_results) >= (max_products * 0.6)  # At least 60% of requested
            
            if has_both_sources and has_enough_products:
                print(f"\n✅ Database has sufficient data!")
                print("💡 Using cached results (faster, saves resources)")
                
                choice = input("\n🤔 Use cached data or scrape fresh? (cached/fresh): ").strip().lower()
                
                if choice in ['cached', 'c', '']:
                    print("\n✓ Using cached database results")
                    result_df = result_df.sort_values(by="price_numeric")
                    return result_df, False
                else:
                    print("\n🌐 Starting fresh web scrape...")
            else:
                print(f"\n⚠️  Database results incomplete:")
                if not has_both_sources:
                    print("   • Missing data from some sources")
                if not has_enough_products:
                    print(f"   • Only {len(db_results)} products (requested {max_products} per source)")
                print("\n🌐 Performing fresh web scrape for better results...")
        else:
            print(f"❌ No matching products found in database")
            print("🌐 Starting fresh web scrape...")
    else:
        print(f"\n🌐 Force scrape enabled - bypassing database check")
    
    # Step 2: Perform fresh scrape
    print(f"\n{'='*70}")
    print(f"🌐 FRESH WEB SCRAPE".center(70))
    print(f"{'='*70}\n")
    
    result_df = smart_product_agent(product_name, rag_storage, max_products)
    
    return result_df, True


# ===========================
# Main Execution
# ===========================

if __name__ == "__main__":
    # Initialize RAG storage
    rag_storage = ProductRAGStorage('product_rag_database.pkl')
    
    print("\n" + "="*70)
    print("🛍️  SMART PRODUCT COMPARISON SYSTEM (GraphRAG)".center(70))
    print("="*70)
    print("\n📋 Main Menu:")
    print("1. 🛒 Smart Product Search (Auto-checks Database First)")
    print("2. 🌐 Force New Web Scrape (Bypass Database)")
    print("3. 🔎 Search in Database Only (Semantic Search)")
    print("4. 📊 View Database Statistics")
    print("5. 📈 Show Analysis Graphs")
    print("6. 🕸️  Visualize Knowledge Graph")
    print("7. 📁 Export Graph (Neo4j/Gephi Format)")
    print("8. 💾 Export Database to CSV (GraphRAG Structure)")
    print("9. 🌐 Export Product Graph (JSON)")
    print("10. 🗑️  Clear Database")
    print("11. 🚪 Exit")
    print("12. 🔎 Generic RAG Pipeline (SerpAPI + Gemini)")
    
    print("6. 💾 Export Database to CSV (GraphRAG Structure)")
    print("7. 🌐 Export Product Graph (JSON)")
    print("8. 🗑️  Clear Database")
    print("9. 🚪 Exit")
    
    while True:
        choice = input("\nEnter choice (1-11): ").strip()
        
        if choice == "1":
            # Smart search - checks database first
            product_name = input("\n🔍 Enter product name to search: ").strip()
            if product_name:
                max_products = input("📦 How many products per source? (default: 5): ").strip()
                max_products = int(max_products) if max_products.isdigit() else 5
                
                result_df, did_scrape = smart_search_products(product_name, rag_storage, max_products, force_new_scrape=False)
                
                if result_df is not None and not result_df.empty:
                    if did_scrape:
                        print("\n✓ Fresh data scraped and stored in GraphRAG database")
                    else:
                        print("\n✓ Results loaded from GraphRAG database")
                    
                    print("\n🖥️  Opening GUI with detailed product information...")
                    display_results_gui_with_details(result_df, rag_storage)
                else:
                    print("\n❌ No products found.")
        
        elif choice == "2":
            # Force new scrape
            product_name = input("\n🌐 Enter product name to search: ").strip()
            if product_name:
                max_products = input("📦 How many products per source? (default: 5): ").strip()
                max_products = int(max_products) if max_products.isdigit() else 5
                
                result_df, _ = smart_search_products(product_name, rag_storage, max_products, force_new_scrape=True)
                
                if result_df is not None and not result_df.empty:
                    print("\n✓ Fresh data scraped and stored in GraphRAG database")
                    print("\n🖥️  Opening GUI with detailed product information...")
                    display_results_gui_with_details(result_df, rag_storage)
                else:
                    print("\n❌ No products found.")
        
        elif choice == "3":
            # Database search only
            query = input("\n🔎 Enter search query: ").strip()
            if query:
                results = rag_storage.semantic_search(query, top_k=20)
                if results:
                    result_df = pd.DataFrame(results)
                    result_df = result_df.sort_values(by="price_numeric")
                    print(f"\n✓ Found {len(result_df)} matching products in database")
                    display_results_gui_with_details(result_df, rag_storage)
                else:
                    print("\n❌ No matching products found in database.")
        
        elif choice == "4":
            # Statistics
            create_detailed_report(rag_storage)
            # Also show graph stats
            print("\n" + "="*70)
            print("📊 KNOWLEDGE GRAPH STATISTICS")
            print("="*70)
            stats = rag_storage.graph_db.get_graph_stats()
            for key, value in stats.items():
                print(f"{key.replace('_', ' ').title():30} : {value}")
        
        elif choice == "5":
            # Graphs
            create_comprehensive_graphs(rag_storage)
        
        elif choice == "6":
            # Visualize Knowledge Graph
            print("\n🕸️  Generating Knowledge Graph Visualization...")
            rag_storage.graph_db.visualize_graph('product_knowledge_graph.png')
        
        elif choice == "7":
            # Export graph in Neo4j/Gephi format
            print("\n📁 Export Options:")
            print("1. Neo4j JSON format")
            print("2. Gephi CSV format")
            print("3. Both formats")
            export_choice = input("Choose export format (1-3): ").strip()
            
            if export_choice == "1":
                rag_storage.graph_db.export_to_neo4j_format('graph_neo4j.json')
            elif export_choice == "2":
                rag_storage.graph_db.export_to_gephi_format('graph_nodes.csv', 'graph_edges.csv')
            elif export_choice == "3":
                rag_storage.graph_db.export_to_neo4j_format('graph_neo4j.json')
                rag_storage.graph_db.export_to_gephi_format('graph_nodes.csv', 'graph_edges.csv')
            else:
                print("❌ Invalid choice")
        
        elif choice == "8":
            # Export CSV
            filename = input("💾 Enter filename (default: products_detailed_export.csv): ").strip()
            if not filename:
                filename = "products_detailed_export.csv"
            rag_storage.export_to_csv(filename)
        
        elif choice == "9":
            # Export JSON graph
            filename = input("🌐 Enter filename (default: product_graph.json): ").strip()
            if not filename:
                filename = "product_graph.json"
            rag_storage.export_graph_to_json(filename)
        
        elif choice == "10":
            # Clear database
            confirm = input("⚠️  Are you sure you want to clear all data? (yes/no): ").strip().lower()
            if confirm == "yes":
                rag_storage.clear_storage()
                print("✓ Database cleared successfully.")
        
        elif choice == "11":
            # Exit
            print("\n👋 Exiting... Goodbye!")
            break
        elif choice == "12":
            print("\nGeneric RAG Pipeline")
            pipeline = build_generic_rag()
            q = input("Enter any query/topic to run RAG (or blank to cancel): ").strip()
            if not q:
                continue
            results = pipeline.search(q)
            if not results:
                print("No results.")
                continue
            print(f"\n✓ {len(results)} result(s)\n")
            # Pretty print top 1-3
            for idx, item in enumerate(results[:3], start=1):
                print(f"Result {idx}:")
                print(f"  Name: {item.get('name')}")
                if item.get('key_facts'):
                    print(f"  Key facts: {item.get('key_facts')}")
                if item.get('attributes'):
                    print(f"  Attributes: {item.get('attributes')}")
                if item.get('related_items'):
                    print(f"  Related: {item.get('related_items')}")
                print("---")
        
        else:
            print("❌ Invalid choice. Please enter 1-11.")

