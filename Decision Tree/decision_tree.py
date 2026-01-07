import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, KFold
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV

# Load the dataset (apparently it can contain ? values instead of nan so convert them to nan as well)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data"
columns = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'Target']
df = pd.read_csv(url, header=None, names=columns, na_values='?')

# Separate Features and Target
X = df.drop('Target', axis=1)
y = df['Target']

# Encode the Target (currently '+' and '-') to 1 and 0
le = LabelEncoder()
y = le.fit_transform(y)

# Identify numeric and categorical columns
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# Create transformers for imputation and encoding
# Numeric: Fill missing with Mean
numeric_transformer = SimpleImputer(strategy='mean')

# Categorical: Fill missing with Mode (most_frequent) AND One-Hot Encode
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Combine into a preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Base pipeline
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', DecisionTreeClassifier(random_state=42))])

print("Data Loaded and Pipeline Created.")

# Grid of parameters
param_grid = {
    'classifier__criterion': ['gini', 'entropy'],
    'classifier__max_depth': [3, 5, 7, 10, None],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4],
    'classifier__class_weight': [None, 'balanced']
}

# Grid search. Can do refit=True so that it retrains the best model on the full data
grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)

print("Starting Grid Search...")
grid_search.fit(X, y)


# Results
print(f"Best ROC AUC Score: {grid_search.best_score_:.2f}")
print("Best Parameters found:")
print(grid_search.best_params_)

best_model = grid_search.best_estimator_

# Define 5-Fold CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Calculate scores
cv_scores = cross_val_score(best_model, X, y, cv=kf, scoring='accuracy')
roc_auc_scores = cross_val_score(best_model, X, y, cv=kf, scoring='roc_auc')
f1_scores = cross_val_score(best_model, X, y, cv=kf, scoring='f1')

print(f"Cross-Validation Scores: {cv_scores}")
print(f"Mean Accuracy: {cv_scores.mean():.2f}")
print(f"Accuracy Standard Deviation: {cv_scores.std():.2f}")
print(f"ROC AUC Scores: {roc_auc_scores}")
print(f"Mean ROC AUC: {roc_auc_scores.mean():.2f}")
print(f"ROC AUC Standard Deviation: {roc_auc_scores.std():.2f}")
print(f"F1 Scores: {f1_scores}")
print(f"Mean F1: {f1_scores.mean():.2f}")
print(f"F1 Standard Deviation: {f1_scores.std():.2f}")


best_model.fit(X, y)

# Get the names of the features after One-Hot Encoding
ohe = best_model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['encoder']
feature_names = list(numeric_features) + list(ohe.get_feature_names_out(categorical_features))

# Extract Rules
tree_rules = export_text(best_model.named_steps['classifier'], feature_names=feature_names)

print("\n--- DECISION RULES ---")
print(tree_rules)


# Tree visualizer
plt.figure(figsize=(20,10))
plot_tree(best_model.named_steps['classifier'], 
          feature_names=feature_names, 
          class_names=['Rejected', 'Approved'], 
          filled=True, 
          rounded=True, 
          fontsize=10)
plt.title("Credit Approval Decision Tree")
plt.savefig('tree.png')
plt.show()