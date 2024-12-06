# Performs Regression Analysis on the Olympic Dataset

import data_preprocessing as dp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm


# Finds the optimal degree for polynomial regression
# Plots the RMSE vs polynomial degree
# Plots the first 300 observations of the original weight and predicted weight
#   based on a polynomial regression model with the optimal degree
def optimal_degree(X_train, y_train, X_test, y_test):

    pipeline = make_pipeline(PolynomialFeatures(), LinearRegression())
    param_grid = {'polynomialfeatures__degree': np.arange(1, 8)}
    grid_search = GridSearchCV(pipeline, param_grid, scoring='neg_mean_squared_error') # negative bc GridSearchCV maximizes

    grid_search.fit(X_train, y_train)

    results = pd.DataFrame(grid_search.cv_results_)
    optimum_n = grid_search.best_params_['polynomialfeatures__degree']
    best_rmse = np.sqrt(-grid_search.best_score_)

    plt.figure(figsize=(10, 6))
    # negate and sqrt to get RMSE
    plt.plot(results['param_polynomialfeatures__degree'], np.sqrt(-results['mean_test_score']), marker='o', linestyle='--')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('RMSE')
    plt.title('RMSE vs Polynomial Degree for Olympic Dataset')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print(f"Optimum polynomial degree: {optimum_n}")
    print(f"Best RMSE: {best_rmse:.3f}")
    print()

    final_model = make_pipeline(PolynomialFeatures(degree=optimum_n), LinearRegression())
    final_model.fit(X_train, y_train)

    y_pred = final_model.predict(X_test)

    # Reverse standardization
    # weight is at index 2
    y_test_original = y_test * scaler.scale_[2] + scaler.mean_[2]
    y_pred_original = y_pred * scaler.scale_[2] + scaler.mean_[2]

    plt.figure(figsize=(10, 6))
    plt.plot(y_test_original.values[:300], label='Original Weight')
    plt.plot(y_pred_original[:300], label='Predicted Weight')
    plt.xlabel('Observation')
    plt.ylabel('Weight')
    plt.title('Regression Model Olympic Dataset')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    mse = mean_squared_error(y_test_original, y_pred_original)
    print(f"Mean Squared Error (MSE): {mse:.3f}")

# 95% CI
# Plots the first 100 observations of the original weight and predicted weight
#   based on a polynomial regression model with the optimal degree
#   with a 95% prediction interval
def ci_95(X_train, y_train, X_test, y_test, optimum_n=5):
    # Fit the final model using statsmodels
    X_train_poly = PolynomialFeatures(degree=optimum_n).fit_transform(X_train)
    X_test_poly = PolynomialFeatures(degree=optimum_n).fit_transform(X_test)

    model = sm.OLS(y_train, X_train_poly).fit()
    predictions = model.get_prediction(X_test_poly)
    prediction_summary = predictions.summary_frame(alpha=0.05)

    # Reverse standardization
    y_test_original = y_test * scaler.scale_[2] + scaler.mean_[2]
    y_pred_original = prediction_summary['mean'] * scaler.scale_[2] + scaler.mean_[2]
    lower_bound = prediction_summary['obs_ci_lower'] * scaler.scale_[2] + scaler.mean_[2]
    upper_bound = prediction_summary['obs_ci_upper'] * scaler.scale_[2] + scaler.mean_[2]

    # Plotting
    observations = 100
    plt.figure(figsize=(10, 6))
    plt.plot(y_test_original.values[:observations], label='Original Weight', color='orange')
    plt.plot(y_pred_original.values[:observations], label='Predicted Weight', color='blue')
    plt.fill_between(range(observations), lower_bound[:observations], upper_bound[:observations], color='b', alpha=0.2, label='95% Prediction Interval')
    plt.xlabel('Observation')
    plt.ylabel('Weight')
    plt.title('Regression Model Olympic Dataset with 95% Prediction Interval')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Shows the model summary
def show_model_summary(X_train, y_train, optimum_n=5):
    X_train_poly = PolynomialFeatures(degree=optimum_n).fit_transform(X_train)

    model = sm.OLS(y_train, X_train_poly).fit()
    print(model.summary())
    print()


# This is an example setup for the Olympic dataset
#   to perform regression analysis
df = dp.encoding("Weight")
df, scaler = dp.standardize(df)

selected_features = ['Height', 'Age', 'Sport_encoded', 'Event_encoded', 'Year', 'is_male', 'Team_encoded']

X = df[selected_features]
y = df['Weight']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)