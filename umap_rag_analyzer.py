"""
UMAP RAG Analyzer - Product Clustering with RAG Storage
Creates UMAP visualizations and integrates with RAG pipeline

Author: Major Project - E-Commerce Price Comparison
"""

import pickle
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics.pairwise import cosine_similarity

# Try to import UMAP
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("⚠️ UMAP not installed. Run: pip install umap-learn")


class RAGStorage:
    """
    Retrieval-Augmented Generation (RAG) Storage for Products
    Uses TF-IDF vectorization for semantic search
    """
    
    def __init__(self, storage_path: str = 'rag_storage.pkl'):
        self.storage_path = storage_path
        self.products: List[Dict] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.product_vectors = None
        self.metadata = {
            'created_at': None,
            'last_updated': None,
            'total_products': 0,
            'categories': [],
            'sources': []
        }
        
        # Load existing storage if available
        if os.path.exists(storage_path):
            self.load()
    
    def add_products(self, products: List[Dict]):
        """Add products to RAG storage"""
        for product in products:
            # Check for duplicates using name similarity
            is_duplicate = False
            for existing in self.products:
                if self._is_similar(product.get('name', ''), existing.get('name', '')):
                    is_duplicate = True
                    # Update existing with newer data
                    existing.update(product)
                    break
            
            if not is_duplicate:
                product['added_at'] = datetime.now().isoformat()
                self.products.append(product)
        
        # Re-vectorize
        self._vectorize_products()
        self._update_metadata()
        print(f"📦 RAG Storage: {len(self.products)} products")
    
    def _is_similar(self, name1: str, name2: str, threshold: float = 0.85) -> bool:
        """Check if two product names are similar"""
        if not name1 or not name2:
            return False
        
        # Simple word overlap check
        words1 = set(name1.lower().split())
        words2 = set(name2.lower().split())
        
        if not words1 or not words2:
            return False
        
        overlap = len(words1 & words2) / max(len(words1), len(words2))
        return overlap >= threshold
    
    def _vectorize_products(self):
        """Create TF-IDF vectors for all products"""
        if not self.products:
            return
        
        # Combine product text fields
        texts = []
        for p in self.products:
            text_parts = [
                p.get('name', ''),
                p.get('category', ''),
                p.get('description', ''),
                ' '.join(p.get('features', [])) if isinstance(p.get('features'), list) else str(p.get('features', ''))
            ]
            texts.append(' '.join(filter(None, text_parts)))
        
        self.vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.product_vectors = self.vectorizer.fit_transform(texts)
    
    def _update_metadata(self):
        """Update storage metadata"""
        self.metadata['last_updated'] = datetime.now().isoformat()
        self.metadata['total_products'] = len(self.products)
        
        if self.products:
            self.metadata['categories'] = list(set(p.get('category', 'Unknown') for p in self.products))
            self.metadata['sources'] = list(set(p.get('source', 'Unknown') for p in self.products))
        
        if not self.metadata['created_at']:
            self.metadata['created_at'] = self.metadata['last_updated']
    
    def search(self, query: str, top_k: int = 10, category: str = None) -> List[Dict]:
        """
        Search for products using semantic similarity.
        
        Args:
            query: Search query
            top_k: Number of results to return
            category: Optional category filter
            
        Returns:
            List of matching products with similarity scores
        """
        if not self.products or self.vectorizer is None:
            return []
        
        # Vectorize query
        query_vector = self.vectorizer.transform([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.product_vectors)[0]
        
        # Get top matches
        top_indices = similarities.argsort()[::-1]
        
        results = []
        for idx in top_indices:
            if len(results) >= top_k:
                break
            
            product = self.products[idx].copy()
            product['similarity_score'] = float(similarities[idx])
            
            # Apply category filter if specified
            if category and product.get('category', '').lower() != category.lower():
                continue
            
            # Only include if similarity is above threshold
            if similarities[idx] > 0.1:
                results.append(product)
        
        return results
    
    def get_cache_hit(self, query: str, threshold: float = 0.8) -> Optional[List[Dict]]:
        """
        Check if query can be served from cache.
        
        Args:
            query: Search query
            threshold: Similarity threshold for cache hit
            
        Returns:
            List of cached products if hit, None otherwise
        """
        results = self.search(query, top_k=20)
        
        if results and results[0]['similarity_score'] >= threshold:
            print(f"✅ Cache HIT: Query '{query[:30]}...' (Score: {results[0]['similarity_score']:.2f})")
            return results
        
        print(f"❌ Cache MISS: Query '{query[:30]}...'")
        return None
    
    def get_statistics(self) -> Dict:
        """Get storage statistics"""
        if not self.products:
            return {'total': 0}
        
        df = pd.DataFrame(self.products)
        
        stats = {
            'total_products': len(self.products),
            'by_category': df['category'].value_counts().to_dict() if 'category' in df else {},
            'by_source': df['source'].value_counts().to_dict() if 'source' in df else {},
            'price_stats': {
                'min': float(df['price'].min()) if 'price' in df else 0,
                'max': float(df['price'].max()) if 'price' in df else 0,
                'mean': float(df['price'].mean()) if 'price' in df else 0,
                'median': float(df['price'].median()) if 'price' in df else 0
            } if 'price' in df else {},
            'metadata': self.metadata
        }
        
        return stats
    
    def save(self):
        """Save storage to disk"""
        data = {
            'products': self.products,
            'metadata': self.metadata
        }
        
        with open(self.storage_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"💾 RAG Storage saved: {self.storage_path}")
    
    def load(self):
        """Load storage from disk"""
        try:
            with open(self.storage_path, 'rb') as f:
                data = pickle.load(f)
            
            self.products = data.get('products', [])
            self.metadata = data.get('metadata', self.metadata)
            
            # Re-vectorize
            self._vectorize_products()
            
            print(f"📂 RAG Storage loaded: {len(self.products)} products")
        except Exception as e:
            print(f"⚠️ Could not load RAG storage: {e}")


class UMAPAnalyzer:
    """
    UMAP Dimensionality Reduction and Visualization
    Creates publication-quality visualizations for report
    """
    
    def __init__(self, products: List[Dict]):
        """
        Initialize UMAP analyzer.
        
        Args:
            products: List of product dictionaries
        """
        self.products = products
        self.df = pd.DataFrame(products)
        self.features = None
        self.embedding = None
        self.metrics = {}
    
    def prepare_features(self):
        """Prepare feature matrix for UMAP"""
        print("📊 Preparing features for UMAP...")
        
        # 1. Text features (TF-IDF on product names)
        text_data = self.df['name'].fillna('').tolist()
        vectorizer = TfidfVectorizer(max_features=50, stop_words='english')
        text_features = vectorizer.fit_transform(text_data).toarray()
        print(f"   Text features: {text_features.shape[1]} dimensions")
        
        # 2. Numerical features (price)
        if 'price' in self.df.columns:
            price_data = self.df['price'].fillna(0).values.reshape(-1, 1)
            scaler = StandardScaler()
            numerical_features = scaler.fit_transform(price_data)
            print(f"   Price feature: 1 dimension")
        else:
            numerical_features = np.zeros((len(self.df), 1))
        
        # 3. Rating features (if available)
        if 'rating' in self.df.columns:
            ratings = pd.to_numeric(self.df['rating'], errors='coerce').fillna(0)
            rating_features = StandardScaler().fit_transform(ratings.values.reshape(-1, 1))
            print(f"   Rating feature: 1 dimension")
        else:
            rating_features = np.zeros((len(self.df), 1))
        
        # 4. Sentiment features (if available)
        if 'sentiment_score' in self.df.columns:
            sentiment_features = self.df['sentiment_score'].fillna(0.5).values.reshape(-1, 1)
            print(f"   Sentiment feature: 1 dimension")
        else:
            sentiment_features = np.zeros((len(self.df), 1))
        
        # Combine all features
        self.features = np.hstack([
            numerical_features,
            rating_features,
            sentiment_features,
            text_features
        ])
        
        print(f"   Total features: {self.features.shape[1]} dimensions")
        return self.features
    
    def run_umap(self, n_neighbors: int = 15, min_dist: float = 0.1, 
                 n_components: int = 2, random_state: int = 42) -> np.ndarray:
        """
        Run UMAP dimensionality reduction.
        
        Args:
            n_neighbors: Number of neighbors for UMAP
            min_dist: Minimum distance for UMAP
            n_components: Output dimensions
            random_state: Random seed
            
        Returns:
            UMAP embedding
        """
        if not UMAP_AVAILABLE:
            raise ImportError("UMAP not available. Install with: pip install umap-learn")
        
        if self.features is None:
            self.prepare_features()
        
        print(f"🗺️ Running UMAP (neighbors={n_neighbors}, min_dist={min_dist})...")
        start_time = time.time()
        
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=n_components,
            random_state=random_state,
            verbose=False
        )
        
        self.embedding = reducer.fit_transform(self.features)
        
        elapsed = time.time() - start_time
        print(f"   ✅ UMAP complete in {elapsed:.1f}s")
        
        return self.embedding
    
    def calculate_clustering_metrics(self) -> Dict:
        """Calculate clustering quality metrics"""
        if self.embedding is None:
            self.run_umap()
        
        print("📏 Calculating clustering metrics...")
        
        # Encode categories for metrics
        if 'category' in self.df.columns:
            le = LabelEncoder()
            labels = le.fit_transform(self.df['category'].fillna('Unknown'))
        else:
            labels = np.zeros(len(self.df))
        
        # Calculate metrics
        try:
            silhouette = silhouette_score(self.embedding, labels)
            davies_bouldin = davies_bouldin_score(self.embedding, labels)
            
            # Calculate cluster purity (how pure each cluster is)
            from sklearn.cluster import KMeans
            n_clusters = len(self.df['category'].unique()) if 'category' in self.df.columns else 5
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(self.embedding)
            
            # Purity calculation
            purity = 0
            for cluster_id in range(n_clusters):
                cluster_mask = cluster_labels == cluster_id
                if cluster_mask.sum() > 0:
                    cluster_categories = labels[cluster_mask]
                    most_common = np.bincount(cluster_categories).max()
                    purity += most_common
            purity = purity / len(labels)
            
            self.metrics = {
                'silhouette_score': round(silhouette, 3),
                'davies_bouldin_index': round(davies_bouldin, 3),
                'cluster_purity': round(purity * 100, 1),
                'n_products': len(self.df),
                'n_categories': n_clusters,
                'interpretation': self._interpret_metrics(silhouette, davies_bouldin, purity)
            }
            
            print(f"   Silhouette Score: {self.metrics['silhouette_score']}")
            print(f"   Davies-Bouldin Index: {self.metrics['davies_bouldin_index']}")
            print(f"   Cluster Purity: {self.metrics['cluster_purity']}%")
            
        except Exception as e:
            print(f"   ⚠️ Error calculating metrics: {e}")
            self.metrics = {'error': str(e)}
        
        return self.metrics
    
    def _interpret_metrics(self, silhouette: float, davies_bouldin: float, purity: float) -> Dict:
        """Interpret clustering metrics"""
        interpretation = {}
        
        # Silhouette interpretation
        if silhouette > 0.7:
            interpretation['silhouette'] = "Excellent clustering - strong structure"
        elif silhouette > 0.5:
            interpretation['silhouette'] = "Good clustering - reasonable structure"
        elif silhouette > 0.3:
            interpretation['silhouette'] = "Fair clustering - weak structure"
        else:
            interpretation['silhouette'] = "Poor clustering - no substantial structure"
        
        # Davies-Bouldin interpretation (lower is better)
        if davies_bouldin < 0.5:
            interpretation['davies_bouldin'] = "Excellent cluster separation"
        elif davies_bouldin < 1.0:
            interpretation['davies_bouldin'] = "Good cluster separation"
        elif davies_bouldin < 2.0:
            interpretation['davies_bouldin'] = "Moderate cluster separation"
        else:
            interpretation['davies_bouldin'] = "Poor cluster separation"
        
        # Purity interpretation
        if purity > 0.9:
            interpretation['purity'] = "Excellent category accuracy"
        elif purity > 0.8:
            interpretation['purity'] = "Good category accuracy"
        elif purity > 0.7:
            interpretation['purity'] = "Fair category accuracy"
        else:
            interpretation['purity'] = "Poor category accuracy"
        
        return interpretation
    
    def create_visualization(self, save_path: str = 'umap_visualization.png', 
                            figsize: Tuple[int, int] = (20, 6)) -> str:
        """
        Create 3-panel UMAP visualization for report.
        
        Args:
            save_path: Path to save the figure
            figsize: Figure size
            
        Returns:
            Path to saved figure
        """
        if self.embedding is None:
            self.run_umap()
        
        print(f"🎨 Creating UMAP visualization...")
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Panel 1: Color by Category
        if 'category' in self.df.columns:
            categories = pd.Categorical(self.df['category'])
            unique_categories = self.df['category'].unique()
            n_categories = len(unique_categories)
            
            # Use a colormap
            colors = plt.cm.tab10(np.linspace(0, 1, min(10, n_categories)))
            
            for i, cat in enumerate(unique_categories):
                mask = self.df['category'] == cat
                axes[0].scatter(
                    self.embedding[mask, 0],
                    self.embedding[mask, 1],
                    c=[colors[i % 10]],
                    label=cat,
                    s=60,
                    alpha=0.7,
                    edgecolors='white',
                    linewidths=0.5
                )
            
            axes[0].set_title('Products by Category', fontsize=14, fontweight='bold')
            axes[0].legend(loc='best', fontsize=8, ncol=2)
        else:
            axes[0].scatter(self.embedding[:, 0], self.embedding[:, 1], s=60, alpha=0.7)
            axes[0].set_title('Products (No Category)', fontsize=14, fontweight='bold')
        
        axes[0].set_xlabel('UMAP Dimension 1', fontsize=12)
        axes[0].set_ylabel('UMAP Dimension 2', fontsize=12)
        
        # Panel 2: Color by Source
        if 'source' in self.df.columns:
            source_colors = {'Amazon': '#FF9900', 'Flipkart': '#2874F0', 
                           'Croma': '#00A652', 'Reliance Digital': '#0066B3'}
            
            for source in self.df['source'].unique():
                mask = self.df['source'] == source
                color = source_colors.get(source, '#888888')
                axes[1].scatter(
                    self.embedding[mask, 0],
                    self.embedding[mask, 1],
                    c=color,
                    label=source,
                    s=60,
                    alpha=0.7,
                    edgecolors='white',
                    linewidths=0.5
                )
            
            axes[1].legend(fontsize=12)
        else:
            axes[1].scatter(self.embedding[:, 0], self.embedding[:, 1], s=60, alpha=0.7, c='blue')
        
        axes[1].set_title('Products by Source', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('UMAP Dimension 1', fontsize=12)
        axes[1].set_ylabel('UMAP Dimension 2', fontsize=12)
        
        # Panel 3: Color by Price Range
        if 'price' in self.df.columns:
            try:
                price_labels = pd.qcut(self.df['price'].fillna(0), q=4, 
                                       labels=['Budget', 'Mid-Range', 'Premium', 'Luxury'])
                
                price_colors = {'Budget': '#2ECC71', 'Mid-Range': '#3498DB', 
                               'Premium': '#9B59B6', 'Luxury': '#E74C3C'}
                
                for label in ['Budget', 'Mid-Range', 'Premium', 'Luxury']:
                    mask = price_labels == label
                    axes[2].scatter(
                        self.embedding[mask, 0],
                        self.embedding[mask, 1],
                        c=price_colors[label],
                        label=label,
                        s=60,
                        alpha=0.7,
                        edgecolors='white',
                        linewidths=0.5
                    )
                
                axes[2].legend(fontsize=12)
            except Exception as e:
                # Fallback to continuous colormap
                scatter = axes[2].scatter(
                    self.embedding[:, 0],
                    self.embedding[:, 1],
                    c=self.df['price'].fillna(0),
                    cmap='viridis',
                    s=60,
                    alpha=0.7,
                    edgecolors='white',
                    linewidths=0.5
                )
                plt.colorbar(scatter, ax=axes[2], label='Price (₹)')
        else:
            axes[2].scatter(self.embedding[:, 0], self.embedding[:, 1], s=60, alpha=0.7, c='green')
        
        axes[2].set_title('Products by Price Range', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('UMAP Dimension 1', fontsize=12)
        axes[2].set_ylabel('UMAP Dimension 2', fontsize=12)
        
        # Add overall title with metrics
        if self.metrics and 'silhouette_score' in self.metrics:
            fig.suptitle(
                f"UMAP Product Clustering (n={len(self.df)}) | "
                f"Silhouette: {self.metrics['silhouette_score']:.2f} | "
                f"Purity: {self.metrics['cluster_purity']:.0f}%",
                fontsize=14, fontweight='bold', y=1.02
            )
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"   ✅ Saved: {save_path}")
        
        return save_path
    
    def create_category_breakdown(self, save_path: str = 'category_umap.png') -> str:
        """Create individual UMAP plots for each category"""
        if self.embedding is None:
            self.run_umap()
        
        if 'category' not in self.df.columns:
            print("⚠️ No category column found")
            return None
        
        categories = self.df['category'].unique()
        n_categories = len(categories)
        
        # Calculate grid size
        n_cols = min(4, n_categories)
        n_rows = (n_categories + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
        axes = axes.flatten() if n_categories > 1 else [axes]
        
        for i, category in enumerate(categories):
            mask = self.df['category'] == category
            
            # All points in gray
            axes[i].scatter(
                self.embedding[:, 0],
                self.embedding[:, 1],
                c='lightgray',
                s=30,
                alpha=0.3
            )
            
            # Category points highlighted
            axes[i].scatter(
                self.embedding[mask, 0],
                self.embedding[mask, 1],
                c=plt.cm.tab10(i % 10),
                s=60,
                alpha=0.8,
                edgecolors='black',
                linewidths=0.5
            )
            
            axes[i].set_title(f'{category} (n={mask.sum()})', fontsize=12, fontweight='bold')
            axes[i].set_xlabel('UMAP 1')
            axes[i].set_ylabel('UMAP 2')
        
        # Hide empty subplots
        for j in range(i+1, len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"   ✅ Saved: {save_path}")
        
        return save_path
    
    def create_sentiment_overlay(self, save_path: str = 'sentiment_umap.png') -> str:
        """Create UMAP with sentiment overlay"""
        if self.embedding is None:
            self.run_umap()
        
        if 'sentiment_score' not in self.df.columns:
            print("⚠️ No sentiment_score column found")
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Left: Sentiment score continuous
        scatter = axes[0].scatter(
            self.embedding[:, 0],
            self.embedding[:, 1],
            c=self.df['sentiment_score'].fillna(0.5),
            cmap='RdYlGn',
            s=60,
            alpha=0.7,
            edgecolors='white',
            linewidths=0.5
        )
        plt.colorbar(scatter, ax=axes[0], label='Sentiment Score')
        axes[0].set_title('Sentiment Score Distribution', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('UMAP Dimension 1', fontsize=12)
        axes[0].set_ylabel('UMAP Dimension 2', fontsize=12)
        
        # Right: Sentiment categories
        if 'sentiment' in self.df.columns:
            sentiment_colors = {'positive': '#2ECC71', 'neutral': '#F1C40F', 
                              'negative': '#E74C3C', 'unknown': '#95A5A6'}
            
            for sentiment in ['positive', 'neutral', 'negative', 'unknown']:
                mask = self.df['sentiment'] == sentiment
                if mask.sum() > 0:
                    axes[1].scatter(
                        self.embedding[mask, 0],
                        self.embedding[mask, 1],
                        c=sentiment_colors.get(sentiment, '#888888'),
                        label=f'{sentiment.title()} ({mask.sum()})',
                        s=60,
                        alpha=0.7,
                        edgecolors='white',
                        linewidths=0.5
                    )
            
            axes[1].legend(fontsize=11)
        else:
            axes[1].scatter(self.embedding[:, 0], self.embedding[:, 1], s=60, alpha=0.7)
        
        axes[1].set_title('Sentiment Categories', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('UMAP Dimension 1', fontsize=12)
        axes[1].set_ylabel('UMAP Dimension 2', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"   ✅ Saved: {save_path}")
        
        return save_path
    
    def get_metrics_table(self) -> str:
        """Generate metrics table for report"""
        if not self.metrics:
            self.calculate_clustering_metrics()
        
        table = """
## Table 5: UMAP Clustering Quality Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Silhouette Score | {silhouette_score} | {silhouette_interp} |
| Davies-Bouldin Index | {davies_bouldin_index} | {db_interp} |
| Cluster Purity | {cluster_purity}% | {purity_interp} |
| Products Analyzed | {n_products} | - |
| Categories | {n_categories} | - |

**Key Finding:** {key_finding}
""".format(
            silhouette_score=self.metrics.get('silhouette_score', 'N/A'),
            davies_bouldin_index=self.metrics.get('davies_bouldin_index', 'N/A'),
            cluster_purity=self.metrics.get('cluster_purity', 'N/A'),
            n_products=self.metrics.get('n_products', 0),
            n_categories=self.metrics.get('n_categories', 0),
            silhouette_interp=self.metrics.get('interpretation', {}).get('silhouette', 'N/A'),
            db_interp=self.metrics.get('interpretation', {}).get('davies_bouldin', 'N/A'),
            purity_interp=self.metrics.get('interpretation', {}).get('purity', 'N/A'),
            key_finding=self._generate_key_finding()
        )
        
        return table
    
    def _generate_key_finding(self) -> str:
        """Generate key finding statement for report"""
        silhouette = self.metrics.get('silhouette_score', 0)
        purity = self.metrics.get('cluster_purity', 0)
        
        if silhouette > 0.6 and purity > 85:
            return f"UMAP visualization demonstrates excellent product clustering with silhouette score of {silhouette:.2f} and {purity:.0f}% cluster purity, indicating well-separated product categories suitable for RAG-based semantic search."
        elif silhouette > 0.4:
            return f"UMAP analysis shows reasonable clustering structure (silhouette: {silhouette:.2f}, purity: {purity:.0f}%), validating the semantic similarity captured by TF-IDF vectorization."
        else:
            return f"Clustering results suggest overlapping product features across categories, which may benefit from enhanced feature engineering."


def analyze_scraped_data(pickle_file: str):
    """
    Analyze scraped product data and create visualizations.
    Main entry point for analysis.
    """
    print("\n" + "🔬"*30)
    print("  UMAP RAG ANALYSIS")
    print("🔬"*30)
    
    # Load data
    print(f"\n📂 Loading {pickle_file}...")
    
    try:
        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)
        
        products = data.get('products', data if isinstance(data, list) else [])
        print(f"   ✅ Loaded {len(products)} products")
    except Exception as e:
        print(f"   ❌ Error loading file: {e}")
        return None
    
    # Initialize RAG storage
    print("\n📦 Initializing RAG Storage...")
    rag = RAGStorage('rag_products.pkl')
    rag.add_products(products)
    rag.save()
    
    # Print statistics
    stats = rag.get_statistics()
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total Products: {stats['total_products']}")
    print(f"   Categories: {len(stats['by_category'])}")
    for cat, count in sorted(stats['by_category'].items()):
        print(f"      {cat}: {count}")
    print(f"   Sources: {stats['by_source']}")
    if stats.get('price_stats'):
        print(f"   Price Range: ₹{stats['price_stats']['min']:,.0f} - ₹{stats['price_stats']['max']:,.0f}")
    
    # Run UMAP analysis
    if UMAP_AVAILABLE:
        print("\n🗺️ Running UMAP Analysis...")
        analyzer = UMAPAnalyzer(products)
        analyzer.prepare_features()
        analyzer.run_umap()
        analyzer.calculate_clustering_metrics()
        
        # Create visualizations
        print("\n🎨 Creating Visualizations...")
        analyzer.create_visualization('umap_visualization.png')
        analyzer.create_category_breakdown('umap_categories.png')
        
        if 'sentiment_score' in pd.DataFrame(products).columns:
            analyzer.create_sentiment_overlay('umap_sentiment.png')
        
        # Print metrics table
        print(analyzer.get_metrics_table())
        
        return {
            'rag_storage': rag,
            'analyzer': analyzer,
            'metrics': analyzer.metrics
        }
    else:
        print("\n⚠️ UMAP not available. Install with: pip install umap-learn")
        return {'rag_storage': rag}


# Test
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        pickle_file = sys.argv[1]
    else:
        # Look for pickle files
        pkl_files = [f for f in os.listdir('.') if f.endswith('.pkl')]
        if pkl_files:
            print("Available pickle files:")
            for i, f in enumerate(pkl_files, 1):
                print(f"  {i}. {f}")
            choice = input("Select file (Enter for first): ").strip()
            pickle_file = pkl_files[int(choice)-1] if choice else pkl_files[0]
        else:
            print("No pickle files found. Creating test data...")
            
            # Create test data
            test_products = [
                {'name': f'Samsung Galaxy S{24+i%3}', 'price': 50000+i*1000, 'category': 'Smartphones', 'source': 'Amazon', 'rating': '4.5', 'sentiment_score': 0.8}
                for i in range(20)
            ] + [
                {'name': f'iPhone {15+i%2} Pro', 'price': 80000+i*2000, 'category': 'Smartphones', 'source': 'Flipkart', 'rating': '4.7', 'sentiment_score': 0.85}
                for i in range(15)
            ] + [
                {'name': f'Dell Inspiron {i}', 'price': 45000+i*1000, 'category': 'Laptops', 'source': 'Amazon', 'rating': '4.2', 'sentiment_score': 0.7}
                for i in range(15)
            ] + [
                {'name': f'Sony WH-1000XM{4+i%2}', 'price': 25000+i*500, 'category': 'Headphones', 'source': 'Flipkart', 'rating': '4.6', 'sentiment_score': 0.9}
                for i in range(10)
            ]
            
            with open('test_products.pkl', 'wb') as f:
                pickle.dump({'products': test_products}, f)
            
            pickle_file = 'test_products.pkl'
    
    analyze_scraped_data(pickle_file)
