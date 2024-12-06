# Performs clustering analysis using KMeans and DBSCAN algorithms.
# Also performs association rule mining using the Apriori algorithm.

import pandas as pd
import data_preprocessing as dp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
import time
from sklearn.cluster import DBSCAN
from mlxtend.frequent_patterns import apriori, association_rules

# Applies the elbow method to identify the optimal number of clusters for KMeans++
# Plots both the silhouette scores and within-cluster sum of squares (WCSS) for different values of K
def kmeans_clustering_analysis(df, max_k=20):

    train_set, test_set = train_test_split(df, test_size=0.2, shuffle=True, random_state=5)

    silhouette_scores = []
    wcss = []

    for k in range(2, max_k + 1):
        print(f"Running KMeans with K={k}...")
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=5)
        kmeans.fit(train_set)
        cluster_labels = kmeans.predict(train_set)

        # Calculate silhouette score
        silhouette_avg = silhouette_score(train_set, cluster_labels)
        silhouette_scores.append(silhouette_avg)

        # Calculate within-cluster sum of squares (WCSS)
        wcss.append(kmeans.inertia_)

    # Plot silhouette scores
    plt.figure(figsize=(10, 6))
    plt.plot(range(2, max_k + 1), silhouette_scores, marker='o', linestyle='--')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Analysis for Optimal K')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot within-cluster sum of squares (WCSS)
    plt.figure(figsize=(10, 6))
    plt.plot(range(2, max_k + 1), wcss, marker='o', linestyle='--')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
    plt.title('Elbow Method for Optimal K')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Applies the DBSCAN algorithm to perform clustering analysis
# Grid search is used to find the best parameters for epsilon and min_samples
# Plots the clusters and prints the number of clusters and noise points
def dbscan_clustering_analysis(df):

    # Split into 80% training and 20% testing
    train_set, test_set = train_test_split(df, test_size=0.2, shuffle=True, random_state=5)


    param_grid = {
        'eps': [0.3, 0.5, 0.7, 1.0, 1.5],
        'min_samples': [3, 5, 10, 15]
    }

    best_score = -1
    best_params = {'eps': None, 'min_samples': None}

    for eps in param_grid['eps']:
        for min_samples in param_grid['min_samples']:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            cluster_labels = dbscan.fit_predict(train_set)

            if len(set(cluster_labels)) > 1:  # Ensure that there is more than one cluster
                score = silhouette_score(train_set, cluster_labels)
                if score > best_score:
                    best_score = score
                    best_params['eps'] = eps
                    best_params['min_samples'] = min_samples

    print(f"Best parameters: {best_params}")
    print(f"Best silhouette score: {best_score:.2f}")

    # Perform DBSCAN clustering with the best parameters
    best_dbscan = DBSCAN(eps=best_params['eps'], min_samples=best_params['min_samples'])
    cluster_labels = best_dbscan.fit_predict(train_set)

    # Plot the clusters
    plt.figure(figsize=(10, 6))
    unique_labels = set(cluster_labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = [0, 0, 0, 1]  # Black for noise

        class_member_mask = (cluster_labels == k)
        xy = train_set[class_member_mask].to_numpy()
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=6)

    plt.title('DBSCAN Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Cluster labels
    labels = best_dbscan.labels_

    # Number of clusters (excluding noise)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"Number of clusters: {n_clusters}")
    print(f"Number of noise points: {list(labels).count(-1)}")


# Applies the Apriori algorithm to perform association rule mining
# Two separate analyses are performed: team performance and demographic performance
# The rules are filtered to only include those related to medals (Gold, Silver, Bronze)
def my_apriori(df):

    # Team performance
    df_team = df[['Team', 'Medal', 'Sport', 'Event', 'Decade']]
    team_transactions = pd.get_dummies(df_team)
    print(f"Starting Aproiri for team performance")
    frequent_itemsets_team = apriori(team_transactions, min_support=0.005, use_colnames=True)
    rules_team = association_rules(frequent_itemsets_team, metric="lift", min_threshold=1)
    filtered_rules_medals = rules_team[rules_team['antecedents'].apply(
        lambda x: any(medal in str(x) for medal in ['Medal_Gold', 'Medal_Silver', 'Medal_Bronze'])) |
                                  rules_team['consequents'].apply(lambda x: any(
                                      medal in str(x) for medal in ['Medal_Gold', 'Medal_Silver', 'Medal_Bronze']))]
    print("Team Performance Rules for Medals:")
    print(filtered_rules_medals.head().to_string())
    print("Number of rules: ", len(filtered_rules_medals))

    # Save team performance rules to a CSV file
    filtered_rules_medals.to_csv('team_performance_rules.csv', index=False)
    print("Team performance rules saved to 'team_performance_rules.csv'")



    # demographic performance
    df_demo = df[['Age_Group', 'Height_Group', 'Weight_Group', 'Medal', 'Sport', 'Event', 'Decade']]
    demo_transactions = pd.get_dummies(df_demo)
    print(f"Starting Aproiri for demographic performance")
    frequent_itemsets_demo = apriori(demo_transactions, min_support=0.005, use_colnames=True)
    rules_demo = association_rules(frequent_itemsets_demo, metric="lift", min_threshold=1)
    filtered_rules_medals = rules_demo[rules_demo['antecedents'].apply(
        lambda x: any(medal in str(x) for medal in ['Medal_Gold', 'Medal_Silver', 'Medal_Bronze'])) |
                                       rules_demo['consequents'].apply(lambda x: any(
                                           medal in str(x) for medal in
                                           ['Medal_Gold', 'Medal_Silver', 'Medal_Bronze']))]
    print("Team Performance Rules for Medals:")
    print(filtered_rules_medals.head().to_string())
    print("Number of rules: ", len(filtered_rules_medals))

    # Save demographic performance rules to a CSV file
    filtered_rules_medals.to_csv('demographic_performance_rules.csv', index=False)
    print("Demographic performance rules saved to 'demographic_performance_rules.csv'")


# Example usage
df = dp.encoding_no_target()
df, scaler = dp.standardize(df)
kmeans_clustering_analysis(df)
dbscan_clustering_analysis(df)

df = dp.encoding_association()
my_apriori(df)
