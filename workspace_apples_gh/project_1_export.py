# %% [markdown]
# ___
# # Loading the data and libraries
# ___

# %%
%load_ext autoreload
%autoreload 2

import os
import sys
import warnings

while any(marker in os.getcwd() for marker in ['workspace_apples_gh']):
    os.chdir("..")

sys.path.append('classes_and_functions_apples_gh')

current_directory = os.getcwd()
current_directory

# %%
# Base packages
from colorama import Fore, Back, Style
from typing import Optional, List

# Third-party packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import f_oneway
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split

# Internal packages
from classes_and_functions_apples_gh.custom_transformers import (
    DropColumnTransformer,
    CustomImputer,
    CustomStandardScaler,
    CustomLabelEncoder,
    CustomOneHotEncoder,
    CustomOutlierDetector,
    CustomOutlierRemover,
    CustomMinMaxScaler,
)
from classes_and_functions_apples_gh.visualisation import classification_report_print
from classes_and_functions_apples_gh.visualisation import compute_and_plot_confusion_matrix

from classes_and_functions_apples_gh.classification import CustomClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import xgboost as xgb

# %%
raw_data = pd.read_csv('attachments_apples_gh//apple_quality.csv')
raw_data.head()

# %% [markdown]
# ___
# # EDA Before Cleaning & Preprocessing
# ___

# %%
# Data Types
raw_data.dtypes

# %%
# Summary Statistics
print("\nSummary statistics for numerical variables:")
raw_data.describe()

# %%
print("\nSummary statistics for categorical variables:")
for column in raw_data.select_dtypes(include=['object']).columns:
    print("\n", column, ":")
    print(raw_data[column].value_counts())

# %%
# Null Values?
raw_data.isnull().sum()

# %%
# Outliers?

outlier_detector = CustomOutlierDetector(raw_data)

num_cols = ['Size','Weight','Sweetness','Crunchiness','Juiciness','Ripeness']

for var in num_cols:
    outliers = outlier_detector.detect_outliers_iqr(var)

print("Outliers identified using the IQR method:")
outliers


# %%
# Boxplots!

plt.figure(figsize=(10, 6))
sns.boxplot(data=raw_data[num_cols])
plt.title('Boxplots of Numerical Variables')
plt.xlabel('Variable')
plt.ylabel('Value')
plt.show()

# %%
# Histograms!

for var in num_cols:
    plt.figure(figsize=(8, 6))
    plt.hist(raw_data[var], bins=20, color='skyblue', edgecolor='black')
    plt.title(f'Histogram of {var}')
    plt.xlabel(var)
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

# %%
# Bar Plot!
category_counts = raw_data['Quality'].value_counts()

plt.figure(figsize=(8, 6))
category_counts.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Bar plot of Quality')
plt.xlabel('Quality')
plt.ylabel('Frequency')
plt.grid(axis='y')
plt.show()

# %%
# Heatmap!
correlation_matrix = raw_data[num_cols].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap of Numerical Variables')
plt.show()

# %%
# THE pairplot!
sns.pairplot(raw_data, hue="Quality")
plt.show()

# %%
# Does the index somehow influence the variable? It doesn't!
def indexed_isit_sorted_test(df,features):

    for feature in features:
        plt.scatter(df.index, df[feature], c='skyblue')
        plt.xlabel("index")
        plt.ylabel(feature)
        plt.title(f"{feature} and index")
        plt.show()


indexed_isit_sorted_test(raw_data, num_cols)

# %% [markdown]
# ___
# # Cleaning
# ___

# %%
raw_data.drop(raw_data[raw_data.isnull().any(axis=1)].index, inplace=True)
raw_data['Acidity'] = pd.to_numeric(raw_data['Acidity'], errors='coerce')

# %%
data_cleaning = make_pipeline(
    DropColumnTransformer(columns=["A_id"]),
    FunctionTransformer(lambda X: X.drop_duplicates(), validate=False),
    CustomOutlierRemover(),
)

df_cleaned = data_cleaning.fit_transform(raw_data)
df_cleaned.head()

# %% [markdown]
# ___
# # Preprocessing
# ___

# %%
preprocessing_pipeline = make_pipeline(
    CustomStandardScaler(columns=["Size", "Weight", "Sweetness", "Crunchiness", "Juiciness", "Ripeness", "Acidity"]),
)

# %% [markdown]
# ___
# # EDA After Cleaning & Preprocessing
# ___

# %%
print("\nSummary statistics for numerical variables:")
df_cleaned.describe()

# %%
print("\nSummary statistics for categorical variables:")
for column in df_cleaned.select_dtypes(include=['object']).columns:
    print("\n", column, ":")
    print(df_cleaned[column].value_counts())

# %%
# Boxplots

num_cols_cleaned = ['Size','Weight','Sweetness','Crunchiness','Juiciness','Ripeness','Acidity']

plt.figure(figsize=(10, 6))
sns.boxplot(data=df_cleaned[num_cols_cleaned])
plt.title('Boxplots of Numerical Variables')
plt.xlabel('Variable')
plt.ylabel('Value')
plt.show()

# %%
# Histograms

for var in num_cols_cleaned:
    plt.figure(figsize=(8, 6))
    plt.hist(df_cleaned[var], bins=20, color='skyblue', edgecolor='black')
    plt.title(f'Histogram of {var}')
    plt.xlabel(var)
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

# %%
# Bar Plot
category_counts = df_cleaned['Quality'].value_counts()

plt.figure(figsize=(8, 6))
category_counts.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Bar plot of Quality')
plt.xlabel('Quality')
plt.ylabel('Frequency')
plt.grid(axis='y')
plt.show()

# %%
# Heatmap
correlation_matrix = df_cleaned[num_cols_cleaned].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap of Numerical Variables')
plt.show()

# %%
#Pairplot

sns.pairplot(df_cleaned, hue="Quality")
plt.show()

# %%
# ANOVA

anova_results = {}
for feature in num_cols_cleaned:
    good_quality = df_cleaned[df_cleaned['Quality'] == 'good'][feature]
    bad_quality = df_cleaned[df_cleaned['Quality'] == 'bad'][feature]
    anova_result = f_oneway(good_quality, bad_quality)
    anova_results[feature] = {
        'F-value': anova_result.statistic,
        'p-value': anova_result.pvalue
    }

for feature, result in anova_results.items():
    print(f"ANOVA results for {feature}:")
    print("F-value:", result['F-value'])
    print("p-value:", result['p-value'])
    print()


# %% [markdown]
# - For Size, Sweetness, Juiciness, and Ripeness, the F-values are high, and the p-values are extremely low (close to 0). This suggests that there are significant differences between the groups in terms of these attributes.
# 
# - For Weight, Crunchiness, and Acidity, the F-values are low, and the p-values are relatively high (close to 1 or above 0.05). This indicates that there are no significant differences between the groups for these attributes.
# 
# We can conclude that Size, Sweetness, Juiciness, and Ripeness seem to be important attributes, which significantly affect the dependent variable, while Weight, Crunchiness, and Acidity do not appear to have a significant impact.

# %%
# Pairplot for the significant features

significant_features = ['Size', 'Sweetness', 'Juiciness', 'Ripeness', 'Quality']

sns.pairplot(df_cleaned[significant_features], hue = 'Quality')
plt.show()

# %% [markdown]
# ___
# # Classifiers
# ___

# %% [markdown]
# ## Naive Bayes

# %%
# Extract features
#initial_X = df_cleaned.iloc[:, :-1]
initial_X = df_cleaned[['Size', 'Weight', 'Sweetness', 'Crunchiness',
                       'Juiciness', 'Ripeness', 'Acidity']]

# Extract target
#initial_y = df_cleaned.iloc[:, -1]
initial_y = df_cleaned['Quality']

# Reserve 10% of the dataset as a validation set
# This is for the prediction purposes at the end (Confusion matrices)
X, X_validation, y, y_validation = train_test_split(initial_X, initial_y, 
                                                    test_size=0.1, random_state=42)

# %%
# Tune and Test NB
nb = GaussianNB()

nb_classifier = CustomClassifier(preprocessing_pipeline, nb, k_count=10)
nb_classifier.fit(X, y)

# Print Classification Report
classification_report_print(nb_classifier)


# %% [markdown]
# ## MLPerceptron

# %%
# Test MLP 
mlp = MLPClassifier(
    hidden_layer_sizes=(16, 32, 64),
    activation="relu",
    solver="adam",
    max_iter=1000,
    random_state=42,
    )

mlp_classifier = CustomClassifier(preprocessing_pipeline, mlp, k_count=10)
mlp_classifier.fit(X, y)

# Print Classification Report
classification_report_print(mlp_classifier)

# %% [markdown]
# ## Random Forest

# %%
rf = RandomForestClassifier(random_state=42)

params_grid_rf = {
    'n_estimators': [245, 250, 255],
    'min_samples_split': [3],
    'min_samples_leaf': [1, 2],
}

rf = GridSearchCV(estimator=rf, 
                 param_grid=params_grid_rf, 
                 cv=2,   
                 verbose=1, 
                 scoring='accuracy',
                 n_jobs=-1)

rf_classifier = CustomClassifier(preprocessing_pipeline, rf, k_count=5)
rf_classifier.fit(X, y)

print("Best Parameters:", rf.best_params_)
print(f"Average Accuracy of RF: {rf_classifier.accuracy:.3f}")
print(f"Average F1 Score of RF: {rf_classifier.f1:.3f}")

# %%
rf_best_params = rf.best_params_
rf = RandomForestClassifier(**rf_best_params, random_state=42)

rf_classifier = CustomClassifier(preprocessing_pipeline, rf, k_count=10)
rf_classifier.fit(X, y)

classification_report_print(rf_classifier)

# %% [markdown]
# ## Support Vector Machine

# %%
svm = SVC(random_state=42)

params_grid_svm = {
    'C': [0.1, 0.5, 1, 2, 10],  
    'gamma': ['scale', 'auto', 1, 0.1, 0.01],
    'kernel': ['rbf'],
    'degree': [0, 1, 2, 3, 4],
    'coef0': [0.0, 0.1, 0.5]
}

svm = GridSearchCV(estimator=svm, 
                 param_grid=params_grid_svm, 
                 cv=2,   
                 verbose=1, 
                 scoring='accuracy',
                 n_jobs=-1)

svm_classifier = CustomClassifier(preprocessing_pipeline, svm, k_count=5)
svm_classifier.fit(X, y)

print("Best Parameters:", svm.best_params_)
print(f"Average Accuracy of SVM: {svm_classifier.accuracy:.3f}")
print(f"Average F1 Score of SVM: {svm_classifier.f1:.3f}")

# %%
svm_best_params = svm.best_params_
svm = SVC(**svm_best_params, random_state=42)

svm_classifier = CustomClassifier(preprocessing_pipeline, svm, k_count=10)
svm_classifier.fit(X, y)

classification_report_print(svm_classifier)

# %% [markdown]
# ## Logistic Regression

# %%
log_reg = LogisticRegression(random_state=42)

log_reg_clf = CustomClassifier(preprocessing_pipeline, log_reg, k_count=10)
log_reg_clf.fit(X, y)



classification_report_print(log_reg_clf)

# %% [markdown]
# ## XGBoost

# %%
# Encode the target 
le = LabelEncoder()
encoded_y = pd.DataFrame(le.fit_transform(y))

xgboost = xgb.XGBClassifier(tree_method="hist")

params_grid_xgb = {
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 5, 8],
    'n_estimators': [100, 200, 300],
    'subsample': [0.7, 0.8, 1.0]
}

xgboost = GridSearchCV(estimator=xgboost, 
                 param_grid=params_grid_xgb, 
                 cv=2,   
                 verbose=0, 
                 scoring='accuracy',
                 n_jobs=-1)

xgboost_classifier = CustomClassifier(preprocessing_pipeline, xgboost, k_count=20)
xgboost_classifier.fit(X, encoded_y)

print("Best Parameters for xgboost:", xgboost.best_params_)
print(f"Average Accuracy of xgboost: {xgboost_classifier.accuracy:.3f}")
print(f"Average F1 Score of xgboost: {xgboost_classifier.f1:.3f}")


# %% [markdown]
# ## AdaBoost

# %%
param_grid_ada = {
    'n_estimators': [500],
    'learning_rate':  [0.1],
    'algorithm': ['SAMME']
}
tree = DecisionTreeClassifier(criterion='entropy',
                              max_depth=5,
                              random_state=1)

ada = AdaBoostClassifier(estimator=tree,
                         n_estimators=500,
                         learning_rate=0.1,
                         random_state=1)

ada_grid_search = GridSearchCV(
    estimator=ada,
    param_grid=param_grid_ada,
    cv=5,
    verbose=1,
    scoring='accuracy',
    n_jobs=-1
)

ada_classifier = CustomClassifier(preprocessing_pipeline, ada_grid_search, k_count=10)

ada_classifier.fit(X, y)

print("Best Parameters for AdaBoost:", ada_grid_search.best_params_)
print(f"Average Accuracy of AdaBoost: {ada_classifier.accuracy:.3f}")
print(f"Average F1 Score of AdaBoost: {ada_classifier.f1:.3f}")


# %% [markdown]
# ## kNN

# %%
kNN = KNeighborsClassifier()
k_range = list(range(1, 31))
kNN = GridSearchCV(estimator=kNN, 
                 param_grid= dict(n_neighbors=k_range), 
                 cv=2,   
                 verbose=1, 
                 scoring='accuracy',
                 n_jobs=-1)

kNN_classifier = CustomClassifier(preprocessing_pipeline, kNN, k_count=10)
kNN_classifier.fit(X, y)

print("Best Parameters for knn:", kNN.best_params_)
print(f"Average Accuracy of knn: {kNN_classifier.accuracy:.3f}")
print(f"Average F1 Score of knn: {kNN_classifier.f1:.3f}")

# %% [markdown]
# ## Comparison and Visualization

# %%
# Add classifiers to the list
classifiers = {'Naive Bayes': nb_classifier, 'MLPerceptron': mlp_classifier, 
               'Random Forest': rf_classifier, 'Support Vector Machine': svm_classifier, 
               'LogisticRegression': log_reg_clf, 'XGBoost': xgboost_classifier, 
               'AdaBoost': ada_classifier, 'KNN': kNN_classifier}

scores = [ 
   {
      "classifiers" : classifier_name,
      "accuracy" : f'{classifier.accuracy:.2f}',
      "f1_score" : f'{classifier.f1:.2f}',
   } 
   for classifier_name, classifier 
   in classifiers.items()
]

scores_df = pd.DataFrame(scores)
scores_df = scores_df.sort_values(by='accuracy', ascending=False)
scores_df

# %% [markdown]
# ### Overall Performance Metrics
# 

# %%
scores_df.set_index('classifiers', inplace=True)

# %%
# Creating a heatmap for accuracy
plt.figure(figsize=(10, 6))
sns.heatmap(scores_df[['accuracy']].astype(float), annot=True, cmap='crest', cbar=True, linewidths=.5, fmt='.2f', annot_kws={"size": 15})

# %%
plt.figure(figsize=(10, 6))
sns.heatmap(scores_df[['f1_score']].astype(float), annot=True, cmap='crest', cbar=True, linewidths=.5, fmt='.2f', annot_kws={"size": 15})

# %%
classifiers = {'Naive Bayes': nb_classifier, 'MLPerceptron': mlp_classifier, 
               'Random Forest': rf_classifier, 'Support Vector Machine': svm_classifier, 
               'LogisticRegression': log_reg_clf, 'XGBoost': xgboost_classifier, 
               'AdaBoost': ada_classifier, 'KNN': kNN_classifier}

# %% [markdown]
# ### Confusion Matrices
# We can use our reserved validation set here (`X_validation` and `y_validation`)

# %%
# Make Predictions
for name, classifier in classifiers.items():
    if name != 'XGBoost':
        y_pred = classifier.predict(X_validation)
        compute_and_plot_confusion_matrix(y_test=y_validation, y_pred=y_pred, name=name)


# Encode the target 
encoded_y_val = pd.DataFrame(le.fit_transform(y_validation))
y_pred = xgboost_classifier.predict(X_validation)
compute_and_plot_confusion_matrix(y_test=encoded_y_val, y_pred=y_pred, name='XGBoost')

# %% [markdown]
# ### We can see that the Multi-layer Perceptron did the best job with classification.


