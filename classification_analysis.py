# Performs Classification Analysis on the Olympic Dataset

import data_preprocessing as dp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, recall_score, precision_score, ConfusionMatrixDisplay, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, StackingClassifier, GradientBoostingClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam


import warnings
import time

# Performs a Gridsearch for the best hyperparameters for a Logistic Regression model
# Prints the best parameters found by the Gridsearch
# Trains the best model on the training data
# Plots the confusion matrix and ROC curve for the best model
# Prints the evaluation metrics for the best model
def logistic_regression(X_train, X_test, y_train, y_test):

    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

    # grid search
    param_grid = {
        'penalty': ['l1', 'l2', 'elasticnet'],
        'C': np.logspace(-3, 1, 10),
        'l1_ratio': np.linspace(0, 1, 30)
    }
    grid_search = GridSearchCV(LogisticRegression(solver='saga'), param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    print("Best parameters found by grid search:")
    print(grid_search.best_params_)
    print()

    # best model from grid search
    best_model = grid_search.best_estimator_
    best_model.fit(X_train, y_train)
    y_pred_best = best_model.predict(X_test)
    y_pred_proba_best = best_model.predict_proba(X_test)[:, 1]

    print(f"Train score (best model): {best_model.score(X_train, y_train):.2f}")
    print(f"Test score (best model): {best_model.score(X_test, y_test):.2f}")
    print()

    cm_best = confusion_matrix(y_test, y_pred_best)
    disp_best = ConfusionMatrixDisplay(confusion_matrix=cm_best)
    disp_best.plot()
    plt.title('Confusion Matrix (Best Model)')
    plt.show()

    fpr_best, tpr_best, _ = roc_curve(y_test, y_pred_proba_best)
    roc_auc_best = auc(fpr_best, tpr_best)
    plt.figure()
    plt.plot(fpr_best, tpr_best, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc_best:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (Best Model)')
    plt.legend(loc='lower right')
    plt.show()

    accuracy_best = accuracy_score(y_test, y_pred_best)
    recall_best = recall_score(y_test, y_pred_best)
    precision_best = precision_score(y_test, y_pred_best)
    f1_best = f1_score(y_test, y_pred_best)
    specificity_best = recall_score(y_test, y_pred_best, pos_label=0)

    print(f'Accuracy (best model): {accuracy_best:.2f}')
    print(f'Recall (best model): {recall_best:.2f}')
    print(f'Precision (best model): {precision_best:.2f}')
    print(f'F1 Score (best model): {f1_best:.2f}')
    print(f'Specificity (best model): {specificity_best:.2f}')

# Performs a Gridsearch for the best hyperparameters for a Decision Tree model
# Prints the best parameters found by the Gridsearch
# Finds the best alpha for post-pruning
# Compares the pre-pruned and post-pruned models and selects the best model
# Plots the confusion matrix and ROC curve for the best model
# Prints the evaluation metrics for the best model
def decision_tree(X_train, X_test, y_train, y_test):

    # pre pruning
    tuned_parameters = [{
        'max_depth': [5, 10, 20, 30],
        'min_samples_split': [50, 100, 200],
        'min_samples_leaf': [10, 50, 100],
        'criterion': ['gini', 'entropy'],
        'splitter': ['best', 'random'],
        'max_features': ['sqrt', 'log2']
    }]

    grid_search = GridSearchCV(DecisionTreeClassifier(random_state=5), tuned_parameters, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    print(f"Best parameters for pre-pruned tree: {best_params}")

    best_model_pre_prune = DecisionTreeClassifier(**best_params, random_state=5)
    best_model_pre_prune.fit(X_train, y_train)

    pre_prune_train_accuracy = accuracy_score(y_train, best_model_pre_prune.predict(X_train))
    pre_prune_test_accuracy = accuracy_score(y_test, best_model_pre_prune.predict(X_test))
    print(f"Train Accuracy (pre-pruned): {pre_prune_train_accuracy:.2f}")
    print(f"Test Accuracy (pre-pruned): {pre_prune_test_accuracy:.2f}")
    print()

    # post pruning
    # cost complexity pruning path
    model = DecisionTreeClassifier(random_state=5)
    model.fit(X_train, y_train)
    path = model.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas = path.ccp_alphas

    train_scores = []
    test_scores = []

    # total_len = len(ccp_alphas)
    # start_time = time.time()
    for i, ccp_alpha in enumerate(ccp_alphas):
        # print(f"Alpha: {ccp_alpha} ({i}/{total_len})")
        ccp_alpha = max(0, ccp_alpha)
        clf = DecisionTreeClassifier(random_state=5, ccp_alpha=ccp_alpha)
        clf.fit(X_train, y_train)
        train_scores.append(accuracy_score(y_train, clf.predict(X_train)))
        test_scores.append(accuracy_score(y_test, clf.predict(X_test)))
        # end_time = time.time()
        # print(f"Time elapsed: {end_time - start_time:.2f} seconds")
        # start_time = end_time

    best_alpha = ccp_alphas[np.argmax(test_scores)]
    print(f"Best alpha: {best_alpha}")

    plt.figure(figsize=(10, 6))
    plt.plot(ccp_alphas, train_scores, marker='o', label='Train', drawstyle="steps-post")
    plt.plot(ccp_alphas, test_scores, marker='o', label='Test', drawstyle="steps-post")
    plt.xlabel('Alpha')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Alpha for Training and Testing sets')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # best_alpha = 2.1131593707801425e-06
    post_pruned_model = DecisionTreeClassifier(random_state=5, ccp_alpha=best_alpha)
    post_pruned_model.fit(X_train, y_train)

    post_prune_train_accuracy = accuracy_score(y_train, post_pruned_model.predict(X_train))
    post_prune_test_accuracy = accuracy_score(y_test, post_pruned_model.predict(X_test))
    print(f"Train Accuracy (post-pruning): {post_prune_train_accuracy:.2f}")
    print(f"Test Accuracy (post-pruning): {post_prune_test_accuracy:.2f}")
    print()


    print(f"Pre-Pruned Test Score: {pre_prune_test_accuracy:.2f}")
    print(f"Post-Pruned Test Score: {post_prune_test_accuracy:.2f}")
    if post_prune_test_accuracy > pre_prune_test_accuracy:
        pruned_model = post_pruned_model
        print("Using post-pruned model")
        print(f"Best alpha: {best_alpha}")
    else:
        pruned_model = best_model_pre_prune
        print("Using pre-pruned model")
        print(f"Best parameters: {best_params}")
    print()

    y_pred = pruned_model.predict(X_test)
    y_pred_proba = pruned_model.predict_proba(X_test)[:, 1]

    # Calculate and print evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    specificity = recall_score(y_test, y_pred, pos_label=0)

    print(f'Decision Tree')
    print(f'Accuracy: {accuracy:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'F1 Score: {f1:.2f}')
    print(f'Specificity: {specificity:.2f}')
    print()

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title('Confusion Matrix for Decision Tree')
    plt.show()

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Decision Tree')
    plt.legend(loc='lower right')
    plt.show()

# Finds the best K using the elbow method
# Plots the error rates for each K
# Plots the confusion matrix and ROC curve for the best K
# Prints the evaluation metrics for the best K
def knn(X_train, X_test, y_train, y_test):
    # optimum K using the elbow method
    error_rates = []
    k_values = range(1, 31)

    start_time = time.time()
    for k in k_values:
        print(f"K: {k}")
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        error = mean_squared_error(y_test, y_pred)
        error_rates.append(error)
        end_time = time.time()
        print(f"Time elapsed: {end_time - start_time:.2f} seconds")
        start_time = end_time

    # error rates
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, error_rates, marker='o', linestyle='--')
    plt.xlabel('K')
    plt.ylabel('Mean Squared Error')
    plt.title('Elbow Method for Optimum K')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    best_k = k_values[np.argmin(error_rates)]
    print(f"Best K: {best_k}")

    knn = KNeighborsClassifier(n_neighbors=best_k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    y_pred_proba = knn.predict_proba(X_test)[:, 1]

    # Print the train and test scores
    print(f"Train score (KNN best): {knn.score(X_train, y_train):.2f}")
    print(f"Test score (KNN best): {knn.score(X_test, y_test):.2f}")
    print()

    # Calculate and print evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    specificity = recall_score(y_test, y_pred, pos_label=0)

    print(f'KNN Model')
    print(f'Accuracy: {accuracy:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'F1 Score: {f1:.2f}')
    print(f'Specificity: {specificity:.2f}')
    print()

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title('Confusion Matrix for KNN')
    plt.show()

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for KNN')
    plt.legend(loc='lower right')
    plt.show()

# Trains a Naive Bayes model
# Plots the confusion matrix and ROC curve for the model
# Prints the evaluation metrics for the model
def naive_bayes(X_train, X_test, y_train, y_test):
    # Naive Bayes model
    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    print(f"Train score (Naive Bayes): {model.score(X_train, y_train):.2f}")
    print(f"Test score (Naive Bayes): {model.score(X_test, y_test):.2f}")
    print()


    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    specificity = recall_score(y_test, y_pred, pos_label=0)

    print(f'Naive Bayes Model')
    print(f'Accuracy: {accuracy:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'F1 Score: {f1:.2f}')
    print(f'Specificity: {specificity:.2f}')
    print()

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title('Confusion Matrix for Naive Bayes')
    plt.show()


    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Naive Bayes')
    plt.legend(loc='lower right')
    plt.show()

# Performs a Gridsearch for the best hyperparameters for a SVM model
#   for linear, polynomial, and RBF kernels
# Prints the best parameters found by the Gridsearch for each kernel
# Compares the test scores for each kernel and selects the best model
# Calls evaluate_svm_model to evaluate the best model
def svm(X_train, X_test, y_train, y_test):

    # Grid search for linear kernel
    param_grid_linear = {
        'kernel': ['linear'],
        'C': [0.1, 1, 10, 100]
    }
    grid_search_linear = GridSearchCV(SVC(probability=True), param_grid_linear, cv=5, scoring='accuracy')
    grid_search_linear.fit(X_train, y_train)
    best_linear_model = grid_search_linear.best_estimator_
    linear_test_score = best_linear_model.score(X_test, y_test)
    print(f"Best parameters for linear kernel: {grid_search_linear.best_params_}")
    print(f"Test score (linear kernel): {linear_test_score:.2f}")



    # Grid search for polynomial kernel
    param_grid_poly = {
        'kernel': ['poly'],
        'degree': [2, 3, 4, 5],
        'C': [0.1, 1, 10, 100]
    }
    grid_search_poly = GridSearchCV(SVC(probability=True), param_grid_poly, cv=5, scoring='accuracy')
    grid_search_poly.fit(X_train, y_train)
    best_poly_model = grid_search_poly.best_estimator_
    poly_test_score = best_poly_model.score(X_test, y_test)
    print(f"Best parameters for polynomial kernel: {grid_search_poly.best_params_}")
    print(f"Test score (polynomial kernel): {poly_test_score:.2f}")


    # Grid search for RBF kernel
    param_grid_rbf = {
        'kernel': ['rbf'],
        'gamma': ['scale', 'auto', 0.01, 0.1, 1, 10],
        'C': [0.1, 1, 10, 100]
    }
    grid_search_rbf = GridSearchCV(SVC(probability=True), param_grid_rbf, cv=5, scoring='accuracy')
    grid_search_rbf.fit(X_train, y_train)
    best_rbf_model = grid_search_rbf.best_estimator_
    rbf_test_score = best_rbf_model.score(X_test, y_test)
    print(f"Best parameters for RBF kernel: {grid_search_rbf.best_params_}")
    print(f"Test score (RBF kernel): {rbf_test_score:.2f}")

    # Determine the best model based on test accuracy
    best_model = None
    best_kernel = None
    best_score = max(linear_test_score, poly_test_score, rbf_test_score)

    if best_score == linear_test_score:
        best_model = best_linear_model
        best_kernel = 'linear'
    elif best_score == poly_test_score:
        best_model = best_poly_model
        best_kernel = 'polynomial'
    else:
        best_model = best_rbf_model
        best_kernel = 'RBF'

    # Evaluate the best model
    evaluate_svm_model(best_model, X_test, y_test, best_kernel)

# Evaluates the SVM model
# Plots the confusion matrix and ROC curve for the model
# Prints the evaluation metrics for the model
def evaluate_svm_model(model, X_test, y_test, kernel_name):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Save y_pred_proba and y_test to files
    np.save(f'y_pred_proba_{kernel_name}.npy', y_pred_proba)
    np.save(f'y_test_{kernel_name}.npy', y_test)

    # Print the train and test scores
    print(f"Test score ({kernel_name} kernel): {model.score(X_test, y_test):.2f}")
    print()

    # Calculate and print evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    specificity = recall_score(y_test, y_pred, pos_label=0)

    print(f'SVM Model ({kernel_name} kernel)')
    print(f'Accuracy: {accuracy:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'F1 Score: {f1:.2f}')
    print(f'Specificity: {specificity:.2f}')
    print()

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f'Confusion Matrix for SVM ({kernel_name} kernel)')
    plt.show()

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for SVM ({kernel_name} kernel)')
    plt.legend(loc='lower right')
    plt.show()


# Performs a Gridsearch for the best hyperparameters for a Random Forest model
#   includes Bagging, Stacking, and Boosting
# Prints the best parameters found by the Gridsearch for each model
# Compares the test scores for each model and selects the best model
# Calls evaluate_rf_model to evaluate the best model
def random_forest(X_train, X_test, y_train, y_test):
    # Define parameter grids
    param_grid_rf = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    param_grid_bagging = {
        'n_estimators': [10, 50, 100],
        'estimator__n_estimators': [50, 100, 200],
        'estimator__max_depth': [10, 20],
        'estimator__min_samples_split': [2, 5],
        'estimator__min_samples_leaf': [1, 2]
    }

    param_grid_stacking = {
        'final_estimator__C': [0.1, 1, 10],
        'final_estimator__penalty': ['l2'],
        'final_estimator__solver': ['lbfgs']
    }

    param_grid_boosting = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }

    # Initialize models
    rf = RandomForestClassifier(random_state=5)
    bagging = BaggingClassifier(estimator=RandomForestClassifier(random_state=5), random_state=5)
    stacking = StackingClassifier(
        estimators=[('rf', RandomForestClassifier(random_state=5)), ('svm', SVC(kernel='linear', probability=True))],
        final_estimator=LogisticRegression(), cv=5)
    boosting = GradientBoostingClassifier(random_state=5)

    # Perform grid search for each model
    grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=5, scoring='accuracy', n_jobs=2)
    grid_search_bagging = GridSearchCV(estimator=bagging, param_grid=param_grid_bagging, cv=5, scoring='accuracy', n_jobs=2)
    grid_search_stacking = GridSearchCV(estimator=stacking, param_grid=param_grid_stacking, cv=5, scoring='accuracy', n_jobs=2)
    grid_search_boosting = GridSearchCV(estimator=boosting, param_grid=param_grid_boosting, cv=5, scoring='accuracy', n_jobs=2)

    # Fit the grid search to the data
    print("Fitting Random Forest")
    grid_search_rf.fit(X_train, y_train)
    print("Fitting Bagging")
    grid_search_bagging.fit(X_train, y_train)
    print("Fitting Stacking")
    grid_search_stacking.fit(X_train, y_train)
    print("Fitting Boosting")
    grid_search_boosting.fit(X_train, y_train)

    # Get the best models
    best_rf = grid_search_rf.best_estimator_
    best_bagging = grid_search_bagging.best_estimator_
    best_stacking = grid_search_stacking.best_estimator_
    best_boosting = grid_search_boosting.best_estimator_

    # Evaluate the best models
    models = {
        'Random Forest': (best_rf, grid_search_rf.best_params_),
        'Bagging with Random Forest': (best_bagging, grid_search_bagging.best_params_),
        'Stacking with Random Forest': (best_stacking, grid_search_stacking.best_params_),
        'Boosting with Gradient Boosting': (best_boosting, grid_search_boosting.best_params_)
    }

    best_model_name = None
    best_model = None
    best_score = 0

    for model_name, (model, params) in models.items():
        score = model.score(X_test, y_test)
        print(f"Best score for {model_name}: {score:.2f}")
        print(f"Best parameters for {model_name}: {params}")
        if score > best_score:
            best_score = score
            best_model_name = model_name
            best_model = model

    print(f"\nBest model overall: {best_model_name} with accuracy: {best_score:.2f}")
    evaluate_rf_model(best_model, X_test, y_test, best_model_name)

# Evaluates the Random Forest model
# Plots the confusion matrix and ROC curve for the model
# Prints the evaluation metrics for the model
def evaluate_rf_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Print the test score
    print(f"Test score ({model_name}): {model.score(X_test, y_test):.2f}")
    print()

    # Calculate and print evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    specificity = recall_score(y_test, y_pred, pos_label=0)

    print(f'{model_name} Model')
    print(f'Accuracy: {accuracy:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'F1 Score: {f1:.2f}')
    print(f'Specificity: {specificity:.2f}')
    print()

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f'Confusion Matrix for {model_name}')
    plt.show()

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {model_name}')
    plt.legend(loc='lower right')
    plt.show()

# Trains a Multi-Layer Perceptron model
# Architecture is hardcoded as 3 hidden layers with 512, 256, and 128 neurons respectively
#   Dropout and ReLU activation functions are used
#   Adam optimizer is used with a learning rate of 0.001
#   Binary crossentropy loss function is used
#   Model is trained for 50 epochs with a batch size of 32
#   Hyperparameters can be tuned, but this is what I found to be best with my CPU
# Plots the confusion matrix and ROC curve for the model
# Prints the evaluation metrics for the model
def mlp(X_train, X_test, y_train, y_test, model_save_path='mlp_model.keras', model_name='MLP'):
    # Define the MLP model
    model = Sequential([
        Input(shape=(X_train.shape[1],)),  # Explicit Input layer
        Dense(512, activation='relu'),
        Dropout(0.3),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(X_train, y_train,
                        epochs=50, batch_size=32,
                        validation_split=0.2, verbose=1)

    # Save the model
    model.save(model_save_path)
    print(f'Model saved to {model_save_path}')

    # Evaluate the model
    y_pred_proba = model.predict(X_test).ravel()
    y_pred = (y_pred_proba > 0.5).astype(int)

    # Print the test score
    print(f"Test score ({model_name}): {accuracy_score(y_test, y_pred):.2f}")
    print()

    # Calculate and print evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    specificity = recall_score(y_test, y_pred, pos_label=0)

    print(f'{model_name} Model')
    print(f'Accuracy: {accuracy:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'F1 Score: {f1:.2f}')
    print(f'Specificity: {specificity:.2f}')
    print()

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f'Confusion Matrix for {model_name}')
    plt.show()

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {model_name}')
    plt.legend(loc='lower right')
    plt.show()

# Trains and stores the predictions and probabilities for each model
# Useful for ROC curve plotting
def train_and_store_predictions(X_train, X_test, y_train, y_test):
    # Define hyperparameters for each model
    params_logistic = {
        'penalty': 'l2',
        'C': .007742636826811269,
    }

    ccp_alpha = 2.1131593707801425e-06

    params_knn = {
        'n_neighbors': 1
    }

    params_svm = {
        'kernel': 'rbf',
        'C': 1.0,
        'gamma': 1
    }

    params_boosting = {
        'n_estimators': 200,
        'learning_rate': 0.2,
        'max_depth': 5
    }

    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(**params_logistic),
        'Decision Tree': DecisionTreeClassifier(ccp_alpha=ccp_alpha),
        'KNN': KNeighborsClassifier(**params_knn),
        'RBF SVM': SVC(**params_svm, probability=True),
        'Naive Bayes': GaussianNB(),
        'Boosting': GradientBoostingClassifier(**params_boosting, random_state=5)
    }

    predictions = {}
    probabilities = {}

    for model_name, model in models.items():

        model.fit(X_train, y_train)

        # Predict and store y_pred and y_pred_proba
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

        predictions[model_name] = y_pred
        probabilities[model_name] = y_pred_proba

        # Save y_pred and y_pred_proba to files
        np.save(f'roc/y_pred_{model_name.replace(" ", "_")}.npy', y_pred)
        if y_pred_proba is not None:
            np.save(f'roc/y_pred_proba_{model_name.replace(" ", "_")}.npy', y_pred_proba)

    # MLP
    model = tf.keras.models.load_model('mlp_model.keras')

    # Predict the probabilities
    y_pred_proba = model.predict(X_test).ravel()
    # Predict the classes
    y_pred = (y_pred_proba > 0.5).astype(int)

    # Save the predictions and probabilities with the model name 'mlp'
    np.save('roc/y_pred_MLP.npy', y_pred)
    np.save('roc/y_pred_proba_MLP.npy', y_pred_proba)


    return predictions, probabilities

# Plots the ROC curves for all models on one plot
def plot_roc_curves(y_test):
    model_names = ['Logistic Regression', 'Decision Tree', 'KNN', 'RBF SVM', 'Naive Bayes', 'MLP']

    # Initialize a plot
    plt.figure(figsize=(10, 8))

    # Loop through each model, load the predicted probabilities, and plot the ROC curve
    for model_name in model_names:
        y_pred_proba = np.load(f'roc/y_pred_proba_{model_name.replace(" ", "_")}.npy')
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.2f})')

    # Plot the diagonal line for random guessing
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    # Set plot labels and title
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for All Models')
    plt.legend(loc='lower right', fontsize='x-large')
    plt.grid(True)
    plt.tight_layout()

    # Show the plot
    plt.show()



# This is an example setup for the Olympic dataset
#   to perform classification analysis
df = dp.encoding("is_male")
df, scaler = dp.standardize(df)

selected_features = ['Weight','Height','Age','Sport_encoded','Year','Team_encoded']

X = df[selected_features]
y = df['is_male']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5, stratify=y)

# balance using SMOTE
smote = SMOTE(random_state=5)
X_train, y_train = smote.fit_resample(X_train, y_train)