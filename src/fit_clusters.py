#!/usr/bin/env python3
"""
NBA Team-Season Final Clustering and Validation
Author: Data Scientist

This script:
1. Loads the team_season_features.csv dataset.
2. Preprocesses features (imputation, scaling, PCA).
3. Fits the final K-Means clustering model with k=4.
4. Fits Agglomerative Clustering (linkage='ward') and calculates the Adjusted Rand Index (ARI) to verify consistency.
5. Runs Univariate ANOVA F-tests on all original features across clusters and identifies the top 5 differentiating features.
6. Profiles clusters by computing the mean of each original feature and plotting a Z-score normalized heatmap.
7. Saves the final dataset with cluster assignments to data/team_season_clusters.csv.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
from scipy import stats

from team_clustering_common import extract_feature_matrix, fit_pooled_clusters

def run_final_clustering(csv_path: str, output_csv_path: str, output_heatmap_path: str):
    # Set seed for reproducibility
    np.random.seed(42)
    
    # 1. Load the matrix
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Separate metadata and features
    df_features = extract_feature_matrix(df)
    feature_names = df_features.columns.tolist()

    # Preprocess: impute (pooled median) + standardize PER SEASON (not pooled
    # across 2013-2025) + PCA (95% variance) + KMeans(k=4). See
    # team_clustering_common.py docstring for why season-relative scaling
    # matters here.
    kmeans_labels, features_pca, pca, _, features_imputed = fit_pooled_clusters(
        df_features, df['season_end_year'], k=4, random_state=42
    )

    # Add cluster label to original DataFrame
    df_clustered = df.copy()
    df_clustered['cluster'] = kmeans_labels
    
    # 3. Execute Agglomerative Clustering (k=4, linkage='ward')
    agg_clustering = AgglomerativeClustering(n_clusters=4, linkage='ward')
    agg_labels = agg_clustering.fit_predict(features_pca)
    
    # Compute Adjusted Rand Index (ARI)
    ari = adjusted_rand_score(kmeans_labels, agg_labels)
    print("\n" + "="*50)
    print("           CLUSTERING CONSISTENCY ANALYSIS        ")
    print("="*50)
    print(f"K-Means vs. Agglomerative Clustering (Ward's linkage)")
    print(f"Adjusted Rand Index (ARI): {ari:.6f}")
    if ari > 0.8:
        print("Consistency: Excellent (Almost identical clusters)")
    elif ari > 0.6:
        print("Consistency: Very Good (High overlap)")
    elif ari > 0.4:
        print("Consistency: Moderate (Some structural differences)")
    else:
        print("Consistency: Low (Distinct cluster boundaries)")
    print("="*50)

    # 4. Univariate Validation (ANOVA F-test)
    # We test each original feature (unscaled) across the 4 K-Means clusters
    anova_results = []
    
    # Group indices
    groups = [features_imputed[kmeans_labels == i] for i in range(4)]
    
    for idx, feature in enumerate(feature_names):
        # Gather data for this feature across the 4 clusters
        g0 = groups[0][:, idx]
        g1 = groups[1][:, idx]
        g2 = groups[2][:, idx]
        g3 = groups[3][:, idx]
        
        # Run one-way ANOVA
        f_stat, p_val = stats.f_oneway(g0, g1, g2, g3)
        anova_results.append({
            'Feature': feature,
            'F-Statistic': f_stat,
            'p-value': p_val
        })
        
    anova_df = pd.DataFrame(anova_results)
    # Sort by F-Statistic descending
    anova_df = anova_df.sort_values(by='F-Statistic', ascending=False)
    
    print("\n" + "="*50)
    print("             ANOVA F-TEST RESULTS (TOP 10 FEATURES) ")
    print("="*50)
    print(anova_df.head(10).to_string(index=False, formatters={
        'F-Statistic': '{:,.4f}'.format,
        'p-value': '{:.2e}'.format
    }))
    print("="*50)
    
    print("\nTop 5 differentiating features (highest F-statistic):")
    top_5 = anova_df.head(5)['Feature'].tolist()
    for rank, feat in enumerate(top_5, 1):
        print(f"  {rank}. {feat} (F = {anova_df.iloc[rank-1]['F-Statistic']:.2f})")
    print("="*50)

    # Save ANOVA results for full access
    anova_df.to_csv(Path(output_csv_path).parent / "cluster_anova_results.csv", index=False)

    # 5. Cluster Profiling & Heatmap
    # Calculate the mean of each original feature for each cluster
    # Using df_features to get actual unscaled values
    df_features_with_cluster = df_features.copy()
    df_features_with_cluster['cluster'] = kmeans_labels
    
    cluster_means = df_features_with_cluster.groupby('cluster').mean()
    
    # To compare features on a single heatmap, we compute Z-score per row (feature):
    # Z = (Mean_in_cluster - Mean_of_means) / Std_of_means
    # Wait, let's write a standard row-wise scaler:
    row_means = cluster_means.mean(axis=0)
    row_stds = cluster_means.std(axis=0)
    # Avoid division by zero
    row_stds = row_stds.replace(0, 1.0)
    cluster_means_z = (cluster_means - row_means) / row_stds
    
    # Transpose so features are rows, clusters are columns
    cluster_means_z_t = cluster_means_z.T
    cluster_means_z_t.columns = [f'Cluster {i}' for i in range(4)]
    
    # For visualization, let's sort the rows by the maximum absolute Z-score or group them.
    # Sorting by F-statistic allows us to put the most differentiating features at the top!
    sorted_features = anova_df['Feature'].tolist()
    cluster_means_z_t = cluster_means_z_t.reindex(sorted_features)
    
    # Generate Heatmap
    plt.figure(figsize=(10, 18))
    sns.heatmap(
        cluster_means_z_t, 
        cmap='coolwarm', 
        center=0, 
        annot=True, 
        fmt=".2f", 
        linewidths=0.5, 
        cbar_kws={'label': 'Z-Score (relativo entre clusters)'}
    )
    plt.title('NBA Team-Season Cluster Profiles\n(Features ordenadas por poder de diferenciação - ANOVA F)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Clusters', fontsize=12)
    plt.ylabel('Features Originais', fontsize=12)
    plt.tight_layout()
    
    # Save heatmap
    Path(output_heatmap_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_heatmap_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved cluster profiles heatmap to: {output_heatmap_path}")
    plt.close()
    
    # 6. Save final DataFrame
    # Reorganize columns: metadata first, then cluster, then all features
    first_cols = ['team_id', 'team_abbreviation', 'team_name', 'season_end_year', 'season_year', 'cluster']
    remaining_cols = [col for col in df_clustered.columns if col not in first_cols]
    
    df_final = df_clustered[first_cols + remaining_cols]
    df_final.to_csv(output_csv_path, index=False)
    print(f"Saved final clustered dataset to: {output_csv_path}")
    
    # Print cluster sizes
    print("\nCluster Sizes:")
    print(df_final['cluster'].value_counts().sort_index().to_string())

if __name__ == '__main__':
    project_dir = Path(__file__).resolve().parent.parent
    csv_file = project_dir / "data" / "team_season_features.csv"
    output_csv = project_dir / "data" / "team_season_clusters.csv"
    output_img = project_dir / "viz" / "cluster_heatmap.png"
    
    run_final_clustering(str(csv_file), str(output_csv), str(output_img))
