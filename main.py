# Run to generate all the results for the project

import classification_analysis, clustering_analysis, data_preprocessing as dp, feature_selection, regression_analysis
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Feature Selection
print("Running data preprocessing...")
print("Regression Feature Selection")
feature_selection.regression_analysis()
print()
print("Classification Feature Selection")
feature_selection.classification_analysis()

# Regression Analysis
df_regression = dp.encoding("Weight")
df_regression, scaler = dp.standardize(df_regression)
selected_features = ['Height', 'Age', 'Sport_encoded', 'Event_encoded', 'Year', 'is_male', 'Team_encoded']

X = df_regression[selected_features]
y = df_regression['Weight']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

regression_analysis.optimal_degree(X_train, y_train, X_test, y_test, scaler)
regression_analysis.ci_95(X_train, y_train, X_test, y_test, scaler, 5)
regression_analysis.show_model_summary(X_train, y_train, 5)


# Classification Analysis
df = dp.encoding("is_male")
df, scaler = dp.standardize(df)

selected_features = ['Weight','Height','Age','Sport_encoded','Year','Team_encoded']

X = df[selected_features]
y = df['is_male']

# balance using SMOTE
smote = SMOTE(random_state=5)
X, y = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5, stratify=y)

classification_analysis.logistic_regression(X_train, X_test, y_train, y_test)
classification_analysis.knn(X_train, X_test, y_train, y_test)
classification_analysis.decision_tree(X_train, X_test, y_train, y_test)
classification_analysis.svm(X_train, X_test, y_train, y_test)
classification_analysis.mlp(X_train, X_test, y_train, y_test)
classification_analysis.naive_bayes(X_train, X_test, y_train, y_test)
classification_analysis.plot_roc_curves(y_test)
print("Accuracy Scores:")
classification_analysis.print_accuracy_file()



# Clustering Analysis
df = dp.encoding_no_target()
df, _ = dp.standardize(df)
clustering_analysis.kmeans_clustering_analysis(df)
clustering_analysis.dbscan_clustering_analysis(df)

df = dp.encoding_association()
clustering_analysis.my_apriori(df)




