# This file handles EDA and feature selection for Regression and Classification

import data_preprocessing as dp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
import statsmodels.api as sm
import seaborn as sns

# Gets encoded data with target 'Weight' and standardizes it
# Performs PCA, RF Feature Importance, and Backward Stepwise Regression
#   to identify important features to select for regression
# Also outputs the correlation and covariance matrices for the selected features
def regression_analysis():
    df = dp.encoding("Weight")
    df, scaler = dp.standardize(df)

    # Split into 80% training and 20% testing
    train_set, test_set = train_test_split(df, test_size=0.2, shuffle=True, random_state=5)

    # Split into X and y
    X_train = train_set.drop(columns=['Weight'])
    y_train = train_set['Weight']
    X_test = test_set.drop(columns=['Weight'])
    y_test = test_set['Weight']

    pca(X_train)

    rf(X_train, y_train)

    backward_stepwise_regression(X_train, y_train)

    selected_features = ['Height', 'Age', 'Sport_encoded', 'Event_encoded', 'Year', 'is_male', 'Team_encoded']
    correlation_and_covariance_matrices(dp.encoding("Weight"), selected_features)

# Gets encoded data with target 'Sex' and standardizes it
# Performs PCA, RF Feature Importance, and Backward Stepwise Regression
#   to identify important features to select for classification
# As discussed in the report, 'Event' is dropped from the dataset
# Also outputs the correlation and covariance matrices for the selected features
def classification_analysis(drop_event=False):
    df = dp.encoding("is_male")
    df = df.drop(columns=['Event_encoded'])
    df, scaler = dp.standardize(df)

    # Split into 80% training and 20% testing
    train_set, test_set = train_test_split(df, test_size=0.2, shuffle=True, random_state=5)

    # Split into X and y
    X_train = train_set.drop(columns=['is_male'])
    y_train = train_set['is_male']
    X_test = test_set.drop(columns=['is_male'])
    y_test = test_set['is_male']

    pca(X_train)

    rf(X_train, y_train)

    backward_stepwise_regression(X_train, y_train)

    selected_features = ['Weight','Height','Age','Sport_encoded','Year','Team_encoded']
    correlation_and_covariance_matrices(dp.encoding("is_male"), selected_features)

# PCA
def pca(X_train):
    pca = PCA()
    X_pca = pca.fit_transform(X_train)

    cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)
    num_features_95 = np.argmax(cumulative_explained_variance >= 0.95) + 1  # add 1 bc of 0 indexing

    plt.figure()
    plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o',
             linestyle='--')
    plt.axhline(y=0.95, color='r', linestyle='-')
    plt.axvline(x=num_features_95, color='r', linestyle='-')
    plt.xlabel('Number of Features')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Features Needed for 95% Explained Variance')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print(f"Number of features needed to explain more than 95% of the variance: {num_features_95}")
    print()

# RF
def rf(X_train, y_train):
    rf = RandomForestRegressor()
    rf.fit(X_train, y_train)

    # Get feature importances
    feature_importances = rf.feature_importances_
    features = X_train.columns

    importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    print(importance_df)

    plt.figure(figsize=(10, 8))
    plt.barh(importance_df['Feature'], importance_df['Importance'])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance from Random Forest for Olympic Dataset')
    plt.gca().invert_yaxis()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    threshold = 0.05  # i set to 0.05, but can be adjusted
    selected_features = importance_df[importance_df['Importance'] >= threshold]['Feature'].tolist()
    eliminated_features = importance_df[importance_df['Importance'] < threshold]['Feature'].tolist()

    print(f"Eliminated features: {eliminated_features}")
    print(f"Final selected features: {selected_features}")
    print()

# backward stepwise regression
def backward_stepwise_regression(X_train, y_train):
    results = pd.DataFrame(columns=['Eliminated Feature', 'AIC', 'BIC', 'Adjusted R²', 'p-value'])
    X_train = sm.add_constant(X_train)

    # Stepwise elimination process
    while len(X_train.columns) > 0:
        # Fit the model
        model = sm.OLS(y_train, X_train).fit()

        p_values = model.pvalues

        # Get the feature with the highest p-value
        max_p_value = p_values.max()
        feature_to_eliminate = p_values.idxmax()

        # Store the results
        new_row = pd.DataFrame({
            'Eliminated Feature': [feature_to_eliminate],
            'AIC': [model.aic],
            'BIC': [model.bic],
            'Adjusted R²': [model.rsquared_adj],
            'p-value': [max_p_value]
        })
        results = pd.concat([results, new_row], ignore_index=True)

        # Eliminate the feature with the highest p-value
        X_train = X_train.drop(columns=[feature_to_eliminate])

    print(results.round(3))
    print()

    final_features = X_train.columns.tolist()
    if 'const' in final_features:
        final_features.remove('const')
    print(f"Final selected features: {final_features}")

    print()

# Plots the correlation and covariance matrices for the selected features as heatmaps
def correlation_and_covariance_matrices(df, selected_features):
    # Calculate the correlation matrix
    correlation_matrix = df[selected_features].corr()

    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Pearson Correlation Coefficients Heatmap')
    plt.tight_layout()  # Adjust layout to prevent labels from being cut off
    plt.show()

    # Calculate the covariance matrix
    covariance_matrix = df[selected_features].cov()

    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(covariance_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Covariance Matrix Heatmap')
    plt.tight_layout()  # Adjust layout to prevent labels from being cut off
    plt.show()
