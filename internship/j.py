# HR Analytics - Predict Employee Attrition
# Author: Veerkumar Savant
# Updated: Fixed bugs, improved robustness, added enhancements

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import shap
import warnings
warnings.filterwarnings("ignore")

# 1️⃣ Load Data
try:
    df = pd.read_csv("data.csv")
except FileNotFoundError:
    print("Error: 'data.csv' not found. Please ensure the file exists.")
    exit()

# Optional rename for clarity (depends on dataset)
if 'sales' in df.columns:
    df.rename(columns={'sales': 'Department', 'salary': 'Salary'}, inplace=True)

print("Dataset shape:", df.shape)

# Convert 'Attrition' to numeric (Yes=1, No=0)
df['Attrition_num'] = df['Attrition'].map({'Yes': 1, 'No': 0})

# Attrition rate
attrition_rate = df['Attrition_num'].mean() * 100
print(f"Overall attrition rate: {attrition_rate:.2f}%")

# 2️⃣ Basic EDA
dept_tab = df.groupby('Department')['Attrition_num'].agg(['mean', 'count']).reset_index()
print("\nAttrition by Department:\n", dept_tab)

# Create income bands
df['IncomeBand'] = pd.cut(df['MonthlyIncome'],
                          bins=4,
                          labels=['Low', 'MedLow', 'MedHigh', 'High'])

income_tab = df.groupby('IncomeBand')['Attrition_num'].mean().reset_index()
print("\nAttrition by Income Band:\n", income_tab)

# Plots
plt.figure(figsize=(6,4))
sns.barplot(x='Department', y='Attrition_num', data=df)
plt.title('Attrition by Department')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("attrition_by_department.png")
plt.close()

plt.figure(figsize=(6,4))
sns.barplot(x='IncomeBand', y='Attrition_num', data=df)
plt.title('Attrition by Income Band')
plt.tight_layout()
plt.savefig("attrition_by_income.png")
plt.close()

# 3️⃣ Preprocessing & Split
X = df.drop(['Attrition', 'Attrition_num'], axis=1)  # Drop both original and numeric target
y = df['Attrition_num']

# Identify categorical and numeric columns
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Preprocessing pipeline
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
])

# 4️⃣ Logistic Regression with Tuning
log_pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

# Hyperparameter tuning
log_param_grid = {'classifier__C': [0.1, 1, 10]}
log_grid = GridSearchCV(log_pipeline, log_param_grid, cv=5, scoring='roc_auc')
log_grid.fit(X_train, y_train)
log_model = log_grid.best_estimator_

y_pred_log = log_model.predict(X_test)
y_pred_proba_log = log_model.predict_proba(X_test)[:, 1]

print("\n--- Logistic Regression ---")
print("Best params:", log_grid.best_params_)
print(classification_report(y_test, y_pred_log))
print("ROC AUC:", roc_auc_score(y_test, y_pred_proba_log))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_log))

# Cross-validation score
log_cv_scores = cross_val_score(log_model, X_train, y_train, cv=5, scoring='roc_auc')
print("CV ROC AUC mean:", log_cv_scores.mean())

# 5️⃣ Decision Tree with Tuning
tree_pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('classifier', DecisionTreeClassifier(random_state=42))
])

# Hyperparameter tuning
tree_param_grid = {'classifier__max_depth': [3, 5, 7]}
tree_grid = GridSearchCV(tree_pipeline, tree_param_grid, cv=5, scoring='roc_auc')
tree_grid.fit(X_train, y_train)
tree_model = tree_grid.best_estimator_

y_pred_tree = tree_model.predict(X_test)
y_pred_proba_tree = tree_model.predict_proba(X_test)[:, 1]

print("\n--- Decision Tree ---")
print("Best params:", tree_grid.best_params_)
print(classification_report(y_test, y_pred_tree))
print("ROC AUC:", roc_auc_score(y_test, y_pred_proba_tree))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_tree))

# Cross-validation score
tree_cv_scores = cross_val_score(tree_model, X_train, y_train, cv=5, scoring='roc_auc')
print("CV ROC AUC mean:", tree_cv_scores.mean())

# ============================================================
# 📘 SHAP Analysis for Decision Tree and Logistic Regression
# ============================================================

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# 1️⃣ Transform Data using Preprocessor
# (Assuming preprocessor, X_train, X_test, y_train, y_test already exist)
# ============================================================

X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)
feature_names = preprocessor.get_feature_names_out()

# Convert to DataFrames (fixes SHAP shape mismatch issues)
X_train_transformed_df = pd.DataFrame(X_train_transformed, columns=feature_names)
X_test_transformed_df = pd.DataFrame(X_test_transformed, columns=feature_names)

# ============================================================
# 2️⃣ Train Models (Decision Tree + Logistic Regression)
# (Assuming tree_model and log_model pipelines are already defined)
# ============================================================

tree_model.fit(X_train, y_train)
log_model.fit(X_train, y_train)

y_pred_proba_tree = tree_model.predict_proba(X_test)[:, 1]
y_pred_proba_log = log_model.predict_proba(X_test)[:, 1]

# ============================================================
# 3️⃣ SHAP for Decision Tree
# ============================================================

print("🔍 Running SHAP for Decision Tree...")

tree_classifier = tree_model.named_steps['classifier']
explainer_tree = shap.TreeExplainer(tree_classifier)
shap_values_tree = explainer_tree(X_test_transformed_df).values  # modern SHAP API
shap_values_tree_class1 = shap_values_tree if shap_values_tree.ndim == 2 else shap_values_tree[:, :, 1]

# SHAP summary plot
shap.summary_plot(shap_values_tree_class1, X_test_transformed_df, show=False, cmap="coolwarm")
plt.title("SHAP Summary Plot - Decision Tree")
plt.tight_layout()
plt.savefig("shap_summary_tree.png")
plt.close()

# Top 10 SHAP features
mean_abs_shap_tree = np.mean(np.abs(shap_values_tree_class1), axis=0)
feat_imp_tree = pd.DataFrame({
    'feature': feature_names,
    'mean_abs_shap': mean_abs_shap_tree
}).sort_values(by='mean_abs_shap', ascending=False)

plt.figure(figsize=(8, 5))
plt.barh(feat_imp_tree['feature'][:10], feat_imp_tree['mean_abs_shap'][:10])
plt.gca().invert_yaxis()
plt.title("Top 10 SHAP Feature Importances (Decision Tree)")
plt.xlabel("Mean |SHAP value|")
plt.tight_layout()
plt.savefig("shap_feature_importance_tree.png")
plt.close()

# ============================================================
# 4️⃣ SHAP for Logistic Regression
# ============================================================

print("🔍 Running SHAP for Logistic Regression...")

log_classifier = log_model.named_steps['classifier']
explainer_log = shap.LinearExplainer(log_classifier, X_train_transformed_df)
shap_values_log = explainer_log(X_test_transformed_df).values  # modern SHAP API

# SHAP summary plot
shap.summary_plot(shap_values_log, X_test_transformed_df, show=False, cmap="coolwarm")
plt.title("SHAP Summary Plot - Logistic Regression")
plt.tight_layout()
plt.savefig("shap_summary_log.png")
plt.close()

# Top 10 SHAP features
mean_abs_shap_log = np.mean(np.abs(shap_values_log), axis=0)
feat_imp_log = pd.DataFrame({
    'feature': feature_names,
    'mean_abs_shap': mean_abs_shap_log
}).sort_values(by='mean_abs_shap', ascending=False)

plt.figure(figsize=(8, 5))
plt.barh(feat_imp_log['feature'][:10], feat_imp_log['mean_abs_shap'][:10])
plt.gca().invert_yaxis()
plt.title("Top 10 SHAP Feature Importances (Logistic Regression)")
plt.xlabel("Mean |SHAP value|")
plt.tight_layout()
plt.savefig("shap_feature_importance_log.png")
plt.close()

# ============================================================
# 5️⃣ Compare SHAP Importances Side by Side
# ============================================================

comparison_df = pd.merge(
    feat_imp_tree[['feature', 'mean_abs_shap']].rename(columns={'mean_abs_shap': 'Decision_Tree'}),
    feat_imp_log[['feature', 'mean_abs_shap']].rename(columns={'mean_abs_shap': 'Logistic_Regression'}),
    on='feature',
    how='outer'
).fillna(0).sort_values(by='Decision_Tree', ascending=False)

comparison_df.to_csv("shap_feature_comparison.csv", index=False)

# ============================================================
# 6️⃣ Model Metrics
# ============================================================

metrics = {
    'Model': ['Logistic Regression', 'Decision Tree'],
    'ROC_AUC': [
        roc_auc_score(y_test, y_pred_proba_log),
        roc_auc_score(y_test, y_pred_proba_tree)
    ]
}

metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv("model_metrics.csv", index=False)

# ============================================================
# 7️⃣ Save SHAP Values for Future Use
# ============================================================

np.save("shap_values_tree.npy", shap_values_tree_class1)
np.save("shap_values_log.npy", shap_values_log)

# ============================================================
# ✅ Done
# ============================================================

print("\n✅ SHAP analysis complete! All plots, CSVs, and metrics saved.")
print("📂 Generated files:")
print("- shap_summary_tree.png")
print("- shap_feature_importance_tree.png")
print("- shap_summary_log.png")
print("- shap_feature_importance_log.png")
print("- shap_feature_comparison.csv")
print("- model_metrics.csv")
