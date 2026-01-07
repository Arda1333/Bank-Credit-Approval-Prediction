import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score
)
from sklearn import tree
import seaborn as sns


data_path = "crx.data"

column_names = [
    "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8",
    "A9", "A10", "A11", "A12", "A13", "A14", "A15", "Class"
]

df = pd.read_csv(
    data_path,
    header=None,
    names=column_names,
    na_values="?"
)

df = df.dropna(subset=["Class"])
df["Class"] = df["Class"].map({"+": 1, "-": 0})

X = df.drop("Class", axis=1)
y = df["Class"]

numeric_features = ["A2", "A3", "A8", "A11", "A14", "A15"]
categorical_features = [col for col in X.columns if col not in numeric_features]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

rf_clf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("classifier", rf_clf)
])

# ---------------- Cross-validation ----------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_acc = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
cv_auc = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")


print(f"Cross-Validation Scores: {cv_acc}")
print(f"Accuracy: mean={cv_acc.mean():.4f}, std={cv_acc.std():.4f}")
print(f"ROC-AUC:  mean={cv_auc.mean():.4f}, std={cv_auc.std():.4f}")

print("-" * 50)

# ---------------- Train / Test Split ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("Test Performance")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ---------------- Confusion Matrix ----------------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Not Approved", "Approved"],
            yticklabels=["Not Approved", "Approved"])
plt.title("Confusion Matrix - Credit Approval")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ---------------- ROC Curve ----------------
fpr, tpr, thresholds = roc_curve(y_test, y_proba)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_proba):.3f}")
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Credit Approval")
plt.legend()
plt.grid(True)
plt.show()

# ---------------- Feature Importance ----------------
rf = model.named_steps["classifier"]
ohe = model.named_steps["preprocess"].named_transformers_["cat"].named_steps["onehot"]

cat_names = ohe.get_feature_names_out(categorical_features)
feature_names = np.concatenate([numeric_features, cat_names])

importances = rf.feature_importances_

fi_df = (
    pd.DataFrame({"feature": feature_names, "importance": importances})
    .sort_values("importance", ascending=False)
    .head(15)
)

plt.figure(figsize=(8, 6))
sns.barplot(x="importance", y="feature", data=fi_df, palette="viridis")
plt.title("Top Feature Importances - Random Forest")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.show()

# ---------------- Tree Visualization ----------------
estimator = rf.estimators_[0]

plt.figure(figsize=(22, 10))
tree.plot_tree(
    estimator,
    feature_names=feature_names,
    class_names=["Not Approved", "Approved"],
    filled=True,
    rounded=True,
    max_depth=3,
    fontsize=8
)
plt.title("Random Forest - Example Decision Tree (max_depth=3)")
plt.show()
