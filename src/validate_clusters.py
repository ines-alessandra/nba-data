#!/usr/bin/env python3
"""
NBA Team-Season Clustering Validation Script
Author: Data Scientist

This script:
1. Loads the team_season_features.csv dataset.
2. Removes categorical and identifier metadata columns.
3. Imputes any missing values with the median of each feature.
4. Normalizes features using StandardScaler.
5. Performs PCA, retaining 95% of explained variance.
6. Runs K-Means for k = 2..10, recording Inertia (WCSS), Silhouette Score, and Davies-Bouldin Index.
7. Saves a multi-panel plot to viz/clustering_validation.png.
8. Outputs the exact metric values in a formatted table.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

from team_clustering_common import extract_feature_matrix, season_relative_zscore

def run_clustering_validation(csv_path: str, output_plot_path: str):
    # Set seed for reproducibility
    np.random.seed(42)
    
    # 1. Load the matrix
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")
        
    df = pd.read_csv(csv_path)
    print(f"Loaded dataset from {csv_path} with shape: {df.shape}")

    # 2. Separate categorical/non-numeric columns and save for later
    # Keep team_id, team_abbreviation, team_name, season_end_year, season_year as metadata
    df_features = extract_feature_matrix(df)
    print(f"Features dimension before cleaning: {df_features.shape}")

    # 3. Impute missing values with the median of each feature
    imputer = SimpleImputer(strategy='median')
    features_imputed = pd.DataFrame(imputer.fit_transform(df_features), columns=df_features.columns, index=df_features.index)

    # 4. Standardize each feature PER SEASON (not pooled across 2013-2025) --
    # see team_clustering_common.py docstring for why this matters over this range.
    features_scaled = season_relative_zscore(features_imputed, df['season_end_year'])

    # 5. Apply PCA retaining 95% of explained variance
    pca = PCA(n_components=0.95, random_state=42)
    features_pca = pca.fit_transform(features_scaled.values)

    print("\n" + "="*50)
    print("                 PCA DIMENSIONALITY REDUCTION      ")
    print("="*50)
    print(f"Original number of features: {df_features.shape[1]}")
    print(f"Number of principal components needed for 95% variance: {pca.n_components_}")
    
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    for idx, var in enumerate(explained_variance):
        print(f"  PC{idx+1:02d}: Variance Explained = {var:.4f} (Cumulative = {cumulative_variance[idx]:.4f})")

    # 6. Execute K-Means for k from 2 to 10 and compute metrics
    k_values = range(2, 11)
    results_list = []

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features_pca)
        
        inertia = kmeans.inertia_
        sil = silhouette_score(features_pca, labels, random_state=42)
        db = davies_bouldin_score(features_pca, labels)
        
        results_list.append({
            'k': k,
            'Inertia (WCSS)': inertia,
            'Silhouette Score': sil,
            'Davies-Bouldin Index': db
        })

    results_df = pd.DataFrame(results_list)
    print("\n" + "="*50)
    print("               K-MEANS CLUSTERING METRICS         ")
    print("="*50)
    print(results_df.to_string(index=False, formatters={
        'Inertia (WCSS)': '{:,.2f}'.format,
        'Silhouette Score': '{:.6f}'.format,
        'Davies-Bouldin Index': '{:.6f}'.format
    }))
    print("="*50)

    # 7. Plot Elbow, Silhouette, and Davies-Bouldin in a single figure
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Color palette
    colors = ['#1f77b4', '#2ca02c', '#d62728']

    # Subplot 1: Elbow method (WCSS)
    sns.lineplot(x='k', y='Inertia (WCSS)', data=results_df, marker='o', ax=axes[0], color=colors[0], linewidth=2.5, markersize=8)
    axes[0].set_title('Elbow Curve (WCSS)', fontsize=14, fontweight='bold', pad=12)
    axes[0].set_xlabel('Number of Clusters (k)', fontsize=12)
    axes[0].set_ylabel('Within-Cluster Sum of Squares', fontsize=12)
    axes[0].set_xticks(k_values)
    
    # Subplot 2: Silhouette Score
    sns.lineplot(x='k', y='Silhouette Score', data=results_df, marker='o', ax=axes[1], color=colors[1], linewidth=2.5, markersize=8)
    axes[1].set_title('Silhouette Score (Higher is Better)', fontsize=14, fontweight='bold', pad=12)
    axes[1].set_xlabel('Number of Clusters (k)', fontsize=12)
    axes[1].set_ylabel('Silhouette Coefficient', fontsize=12)
    axes[1].set_xticks(k_values)
    
    # Subplot 3: Davies-Bouldin Index
    sns.lineplot(x='k', y='Davies-Bouldin Index', data=results_df, marker='o', ax=axes[2], color=colors[2], linewidth=2.5, markersize=8)
    axes[2].set_title('Davies-Bouldin Index (Lower is Better)', fontsize=14, fontweight='bold', pad=12)
    axes[2].set_xlabel('Number of Clusters (k)', fontsize=12)
    axes[2].set_ylabel('Davies-Bouldin Score', fontsize=12)
    axes[2].set_xticks(k_values)

    plt.suptitle('NBA Team-Season Clustering Validation Metrics (k=2..10)', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save validation plot
    plot_dir = Path(output_plot_path).parent
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved validation plots to: {output_plot_path}")
    plt.close()

if __name__ == '__main__':
    project_dir = Path(__file__).resolve().parent.parent
    csv_file = project_dir / "data" / "team_season_features.csv"
    output_img = project_dir / "viz" / "clustering_validation.png"
    
    run_clustering_validation(str(csv_file), str(output_img))
