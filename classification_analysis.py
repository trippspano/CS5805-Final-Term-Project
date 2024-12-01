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


import warnings
import time


def logistic_regression(X_train, X_test, y_train, y_test):

    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

    # basic logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    print(f"Train score: {model.score(X_train, y_train):.2f}")
    print(f"Test score: {model.score(X_test, y_test):.2f}")
    print()

    # cm = confusion_matrix(y_test, y_pred)
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    # disp.plot()
    # plt.title('Confusion Matrix')
    # plt.show()

    # fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    # roc_auc = auc(fpr, tpr)
    # plt.figure()
    # plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('ROC Curve')
    # plt.legend(loc='lower right')
    # plt.show()

    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f'Accuracy: {accuracy:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'F1 Score: {f1:.2f}')
    print()

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


def decision_tree(X_train, X_test, y_train, y_test):
    model = DecisionTreeClassifier(random_state=5)
    model.fit(X_train, y_train)

    # train_accuracy = accuracy_score(y_train, model.predict(X_train))
    # test_accuracy = accuracy_score(y_test, model.predict(X_test))
    # print(f"Train Accuracy (base): {train_accuracy:.2f}")
    # print(f"Test Accuracy (base): {test_accuracy:.2f}")
    #
    # print("Base Decision Tree Parameters:")
    # params = model.get_params()
    # print(params)
    # print()

    # plt.figure()
    # plot_tree(model, filled=True, class_names=['Female', 'Male'])
    # plt.title("Decision Tree for Olympic Dataset")
    # plt.show()

    # # pre pruning
    # tuned_parameters = [{
    #     'max_depth': [5, 10, 20, 30],
    #     'min_samples_split': [50, 100, 200],
    #     'min_samples_leaf': [10, 50, 100],
    #     'criterion': ['gini', 'entropy'],
    #     'splitter': ['best', 'random'],
    #     'max_features': ['sqrt', 'log2']
    # }]
    #
    # grid_search = GridSearchCV(DecisionTreeClassifier(random_state=5805), tuned_parameters, cv=5, scoring='accuracy')
    # grid_search.fit(X_train, y_train)
    #
    # best_params = grid_search.best_params_
    # print(f"Best parameters for pre-pruned tree: {best_params}")
    #
    # best_model = DecisionTreeClassifier(**best_params, random_state=5805)
    # best_model.fit(X_train, y_train)
    #
    # train_accuracy = accuracy_score(y_train, best_model.predict(X_train))
    # test_accuracy = accuracy_score(y_test, best_model.predict(X_test))
    # print(f"Train Accuracy (pre-pruned): {train_accuracy:.2f}")
    # print(f"Test Accuracy (pre-pruned): {test_accuracy:.2f}")
    # print()

    # plt.figure()
    # plot_tree(best_model, filled=True, feature_names=numerical_features, class_names=['Not Survived', 'Survived'])
    # plt.title("Pre-Pruned Decision Tree")
    # plt.show()

    # post pruning
    # cost complexity pruning path
    path = model.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas = path.ccp_alphas

    train_scores = []
    test_scores = []

    total_len = len(ccp_alphas)
    start_time = time.time()
    for idx, ccp_alpha in enumerate(ccp_alphas, start=1):
        print(f"Alpha: {idx} of {total_len}")
        ccp_alpha = max(0, ccp_alpha)
        clf = DecisionTreeClassifier(random_state=5, ccp_alpha=ccp_alpha)
        clf.fit(X_train, y_train)
        train_scores.append(accuracy_score(y_train, clf.predict(X_train)))
        test_scores.append(accuracy_score(y_test, clf.predict(X_test)))
        end_time = time.time()
        print(f"Time: {end_time - start_time:.2f}")
        start_time = end_time


    output_file = open('classification_log.txt', 'w')
    best_alpha = ccp_alphas[np.argmax(test_scores)]
    output_file.write(f"Best alpha: {best_alpha}\n")

    pruned_model = DecisionTreeClassifier(random_state=5, ccp_alpha=best_alpha)
    pruned_model.fit(X_train, y_train)

    train_accuracy = accuracy_score(y_train, pruned_model.predict(X_train))
    test_accuracy = accuracy_score(y_test, pruned_model.predict(X_test))
    output_file.write(f"Train Accuracy (post-pruning): {train_accuracy:.2f}\n")
    output_file.write(f"Test Accuracy (post-pruning): {test_accuracy:.2f}\n")
    output_file.write("\n")

    plt.figure(figsize=(10, 6))
    plt.plot(ccp_alphas, train_scores, marker='o', label='Train', drawstyle="steps-post")
    plt.plot(ccp_alphas, test_scores, marker='o', label='Test', drawstyle="steps-post")
    plt.xlabel('Alpha')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Alpha for Training and Testing sets')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('post_pruned_clf.png')


    #
    # plt.figure()
    # plot_tree(pruned_model, filled=True, feature_names=numerical_features, class_names=['Not Survived', 'Survived'])
    # plt.title("Post-Pruned Decision Tree")
    # plt.show()


def knn(X_train, X_test, y_train, y_test):
    # optimum K using the elbow method
    error_rates = []
    k_values = range(1, 31)

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        error = mean_squared_error(y_test, y_pred)
        error_rates.append(error)

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

def svm(X_train, X_test, y_train, y_test):

    output_file = open('svm_log.txt', 'w')
    # Grid search for linear kernel
    param_grid_linear = {
        'kernel': ['linear'],
        'C': [0.1, 1, 10, 100]
    }
    grid_search_linear = GridSearchCV(SVC(probability=True), param_grid_linear, cv=5, scoring='accuracy')
    grid_search_linear.fit(X_train, y_train)
    best_linear_model = grid_search_linear.best_estimator_
    linear_test_score = best_linear_model.score(X_test, y_test)
    output_file.write(f"Best parameters for linear kernel: {grid_search_linear.best_params_}\n")
    output_file.write(f"Test score (linear kernel): {linear_test_score:.2f}\n\n")

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
    output_file.write(f"Best parameters for polynomial kernel: {grid_search_poly.best_params_}\n")
    output_file.write(f"Test score (polynomial kernel): {poly_test_score:.2f}\n\n")

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
    output_file.write(f"Best parameters for RBF kernel: {grid_search_rbf.best_params_}\n")
    output_file.write(f"Test score (RBF kernel): {rbf_test_score:.2f}\n\n")

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

def evaluate_svm_model(model, X_test, y_test, kernel_name):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

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
    plt.savefig("cm_svm.png")

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
    plt.savefig("roc_curve_svm.png")


def random_forest(X_train, X_test, y_train, y_test):
    # Random Forest
    rf_model = RandomForestClassifier(random_state=5)
    rf_model.fit(X_train, y_train)
    rf_test_score = rf_model.score(X_test, y_test)
    print(f"Test score (Random Forest): {rf_test_score:.2f}")

    # Bagging with Random Forest
    bagging_model = BaggingClassifier(base_estimator=RandomForestClassifier(random_state=5), random_state=5)
    bagging_model.fit(X_train, y_train)
    bagging_test_score = bagging_model.score(X_test, y_test)
    print(f"Test score (Bagging with Random Forest): {bagging_test_score:.2f}")

    # Stacking with Random Forest
    estimators = [
        ('rf', RandomForestClassifier(random_state=5)),
        ('svm', SVC(kernel='linear', probability=True))
    ]
    stacking_model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(), cv=5)
    stacking_model.fit(X_train, y_train)
    stacking_test_score = stacking_model.score(X_test, y_test)
    print(f"Test score (Stacking with Random Forest): {stacking_test_score:.2f}")

    # Boosting with Gradient Boosting
    boosting_model = GradientBoostingClassifier(random_state=5)
    boosting_model.fit(X_train, y_train)
    boosting_test_score = boosting_model.score(X_test, y_test)
    print(f"Test score (Boosting with Gradient Boosting): {boosting_test_score:.2f}")

    # Determine the best model based on test accuracy
    best_model = None
    best_model_name = None
    best_score = max(rf_test_score, bagging_test_score, stacking_test_score, boosting_test_score)

    if best_score == rf_test_score:
        best_model = rf_model
        best_model_name = 'Random Forest'
    elif best_score == bagging_test_score:
        best_model = bagging_model
        best_model_name = 'Bagging with Random Forest'
    elif best_score == stacking_test_score:
        best_model = stacking_model
        best_model_name = 'Stacking with Random Forest'
    else:
        best_model = boosting_model
        best_model_name = 'Boosting with Gradient Boosting'

    # Evaluate the best model
    evaluate_rf_model(best_model, X_test, y_test, best_model_name)


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

df = dp.encoding("is_male")
df, scaler = dp.standardize(df)

# print(df.head().to_string())

## WITH EVENT
# selected_features = ['Height', 'Age', 'Weight', 'Event_encoded', 'Year', 'Medal_encoded', 'Team_encoded']

## WITHOUT EVENT
selected_features = ['Weight','Height','Age','Sport_encoded','Year','Team_encoded']

X = df[selected_features]
y = df['is_male']

# # plot imbalance
# plt.figure()
# y.value_counts().plot(kind='bar', title='Imbalanced Olympic Dataset')
# plt.xlabel('is_male')
# plt.ylabel('Count')
# plt.tight_layout()
# plt.show()

# balance using SMOTE
smote = SMOTE(random_state=5)
X_res, y_res = smote.fit_resample(X, y)

# # plot balance
# plt.figure()
# pd.Series(y_res).value_counts().plot(kind='bar', title='Balanced Olympic Dataset')
# plt.xlabel('is_male')
# plt.ylabel('Count')
# plt.tight_layout()
# plt.show()


X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=5, stratify=y_res)

# logistic_regression(X_train, X_test, y_train, y_test)
# decision_tree(X_train, X_test, y_train, y_test)
svm(X_train[:100], X_test, y_train[:100], y_test)
