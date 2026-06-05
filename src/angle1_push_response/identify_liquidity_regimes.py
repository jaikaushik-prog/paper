"""
Intraday Liquidity Regimes - Phase 2: Regime Identification
============================================================
Identifies distinct intraday liquidity regimes using unsupervised clustering.

Methods:
- K-means (baseline, fixed k)
- HDBSCAN (density-based, auto k)

Outputs:
- Regime assignments per stock-year
- Cluster quality metrics (silhouette)
- Cluster centroid interpretations
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Try importing HDBSCAN
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    print("Warning: hdbscan not installed. Install with: pip install hdbscan")

# Configuration
RESULTS_DIR = r"c:\Users\DELL\Desktop\project_nifty_liquid\results"
PLOTS_DIR = r"c:\Users\DELL\Desktop\project_nifty_liquid\plots"

# Feature columns for clustering
FEATURE_COLS = [
    'open_intensity', 'close_intensity', 'midday_flatness',
    'u_shape_strength', 'volume_entropy', 'peak_concentration', 'skewness'
]

# Regime names for interpretation
REGIME_NAMES = {
    0: 'U-Shape Strong',
    1: 'Flat Liquidity', 
    2: 'Open-Dominated',
    3: 'Close-Dominated',
    4: 'Fragmented'
}


def load_features():
    """Load the feature matrix."""
    path = os.path.join(RESULTS_DIR, "intraday_features.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Feature matrix not found at {path}. Run construct_intraday_profiles.py first.")
    
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} stock-year observations")
    return df


def prepare_features(df):
    """Prepare and standardize features for clustering."""
    # Extract feature matrix
    X = df[FEATURE_COLS].values
    
    # Handle any NaN/Inf
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, scaler


def kmeans_clustering(X, k_range=(3, 7)):
    """
    Run K-means for different k values and evaluate.
    Returns best model based on silhouette score.
    """
    results = []
    best_score = -1
    best_model = None
    best_k = None
    
    print("\nK-Means Clustering:")
    print("-" * 50)
    
    for k in range(k_range[0], k_range[1] + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        labels = kmeans.fit_predict(X)
        
        # Silhouette score
        sil_score = silhouette_score(X, labels)
        
        # Inertia (within-cluster sum of squares)
        inertia = kmeans.inertia_
        
        results.append({
            'k': k,
            'silhouette': sil_score,
            'inertia': inertia
        })
        
        print(f"  k={k}: Silhouette={sil_score:.4f}, Inertia={inertia:.1f}")
        
        if sil_score > best_score:
            best_score = sil_score
            best_model = kmeans
            best_k = k
    
    print(f"\n  → Best k={best_k} with silhouette={best_score:.4f}")
    
    return best_model, pd.DataFrame(results)


def hdbscan_clustering(X, min_cluster_size=50):
    """
    Run HDBSCAN clustering.
    """
    if not HDBSCAN_AVAILABLE:
        print("HDBSCAN not available. Skipping.")
        return None, None
    
    print("\nHDBSCAN Clustering:")
    print("-" * 50)
    
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=10,
        metric='euclidean',
        cluster_selection_method='eom'
    )
    
    labels = clusterer.fit_predict(X)
    
    # Count clusters (excluding noise = -1)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    
    # Silhouette (excluding noise)
    if n_clusters > 1:
        mask = labels >= 0
        sil_score = silhouette_score(X[mask], labels[mask])
    else:
        sil_score = 0
    
    print(f"  Found {n_clusters} clusters + {n_noise} noise points")
    print(f"  Silhouette (excl. noise): {sil_score:.4f}")
    
    return clusterer, labels


def interpret_clusters(df, labels, X_scaled, method_name='kmeans'):
    """
    Interpret cluster centroids and assign regime names.
    """
    df = df.copy()
    df['cluster'] = labels
    
    print(f"\n{method_name.upper()} Cluster Interpretation:")
    print("-" * 60)
    
    interpretations = []
    
    for cluster_id in sorted(df['cluster'].unique()):
        if cluster_id == -1:  # HDBSCAN noise
            continue
            
        cluster_mask = df['cluster'] == cluster_id
        cluster_data = df[cluster_mask][FEATURE_COLS]
        n_samples = cluster_mask.sum()
        
        # Mean features
        mean_features = cluster_data.mean()
        
        # Interpret based on feature patterns
        open_int = mean_features['open_intensity']
        close_int = mean_features['close_intensity']
        u_shape = mean_features['u_shape_strength']
        entropy = mean_features['volume_entropy']
        
        # Heuristic naming
        if u_shape > 0.05:
            regime_name = "U-Shape Strong"
        elif open_int > close_int * 1.3:
            regime_name = "Open-Dominated"
        elif close_int > open_int * 1.3:
            regime_name = "Close-Dominated"
        elif entropy > 4.0:
            regime_name = "Flat Liquidity"
        else:
            regime_name = "Mixed Pattern"
        
        interpretations.append({
            'cluster': cluster_id,
            'regime_name': regime_name,
            'n_samples': n_samples,
            'pct_samples': 100 * n_samples / len(df),
            'open_intensity': open_int,
            'close_intensity': close_int,
            'u_shape_strength': u_shape,
            'entropy': entropy
        })
        
        print(f"  Cluster {cluster_id}: {regime_name}")
        print(f"    N={n_samples} ({100*n_samples/len(df):.1f}%)")
        print(f"    Open={open_int:.3f}, Close={close_int:.3f}, U-shape={u_shape:.4f}")
    
    return pd.DataFrame(interpretations)


def run_clustering_pipeline(df, validate=False):
    """
    Run full clustering pipeline.
    """
    # Prepare features
    X_scaled, scaler = prepare_features(df)
    
    print(f"\nFeature matrix shape: {X_scaled.shape}")
    
    # K-Means
    kmeans_model, kmeans_results = kmeans_clustering(X_scaled, k_range=(3, 6))
    kmeans_labels = kmeans_model.predict(X_scaled)
    
    # Interpret K-means clusters
    kmeans_interp = interpret_clusters(df, kmeans_labels, X_scaled, 'K-Means')
    
    # HDBSCAN
    if HDBSCAN_AVAILABLE:
        hdbscan_model, hdbscan_labels = hdbscan_clustering(X_scaled, min_cluster_size=50)
        if hdbscan_labels is not None:
            hdbscan_interp = interpret_clusters(df, hdbscan_labels, X_scaled, 'HDBSCAN')
    else:
        hdbscan_labels = None
        hdbscan_interp = None
    
    # Create output DataFrame
    output_df = df.copy()
    output_df['regime_kmeans'] = kmeans_labels
    if hdbscan_labels is not None:
        output_df['regime_hdbscan'] = hdbscan_labels
    
    # Add regime names
    regime_map = dict(zip(kmeans_interp['cluster'], kmeans_interp['regime_name']))
    output_df['regime_name'] = output_df['regime_kmeans'].map(regime_map)
    
    if validate:
        # Validation: bootstrap stability
        print("\nBootstrap Stability Test:")
        print("-" * 50)
        n_bootstrap = 10
        stability_scores = []
        
        for i in range(n_bootstrap):
            # Sample with replacement
            idx = np.random.choice(len(X_scaled), size=len(X_scaled), replace=True)
            X_boot = X_scaled[idx]
            
            km_boot = KMeans(n_clusters=kmeans_model.n_clusters, random_state=i, n_init=10)
            labels_boot = km_boot.fit_predict(X_boot)
            
            sil = silhouette_score(X_boot, labels_boot)
            stability_scores.append(sil)
        
        print(f"  Mean silhouette: {np.mean(stability_scores):.4f} ± {np.std(stability_scores):.4f}")
    
    return output_df, kmeans_interp, kmeans_results


def save_results(output_df, interp_df, eval_df):
    """Save clustering results."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Regime assignments
    output_path = os.path.join(RESULTS_DIR, "regime_assignments.csv")
    output_df.to_csv(output_path, index=False)
    print(f"\nSaved regime assignments to {output_path}")
    
    # Cluster interpretations
    interp_path = os.path.join(RESULTS_DIR, "regime_interpretations.csv")
    interp_df.to_csv(interp_path, index=False)
    print(f"Saved interpretations to {interp_path}")
    
    # Evaluation metrics
    eval_path = os.path.join(RESULTS_DIR, "clustering_evaluation.csv")
    eval_df.to_csv(eval_path, index=False)
    print(f"Saved evaluation to {eval_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Identify liquidity regimes')
    parser.add_argument('--validate', action='store_true', help='Run bootstrap validation')
    args = parser.parse_args()
    
    # Load features
    df = load_features()
    
    # Run clustering
    output_df, interp_df, eval_df = run_clustering_pipeline(df, validate=args.validate)
    
    # Save results
    save_results(output_df, interp_df, eval_df)
    
    # Summary
    print("\n" + "=" * 60)
    print("REGIME DISTRIBUTION SUMMARY")
    print("=" * 60)
    print(output_df['regime_name'].value_counts())
