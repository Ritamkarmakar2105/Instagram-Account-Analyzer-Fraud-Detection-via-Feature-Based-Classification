import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
data = pd.read_csv(r'C:\Users\Ritam Karmakar\Downloads\archive\Preprocessed_Instagram_Dataset.csv')

# Data Preprocessing
# Replace "#DIV/0!" with a large value (e.g., 1e6) for ratio columns
data['Following/Followers'] = data['Following/Followers'].replace('#DIV/0!', 1e6).astype(float)
data['Posts/Followers'] = data['Posts/Followers'].replace('#DIV/0!', 1e6).astype(float)

# Convert categorical columns to binary (Yes=1, No=0)
categorical_cols = ['Bio', 'Profile Picture', 'External Link', 'Threads']
for col in categorical_cols:
    data[col] = data[col].str.lower().map({'yes': 1, 'n': 0})

# Encode the target variable
label_mapping = {'Bot': 0, 'Scam': 1, 'Real': 2, 'Spam': 3}
data['Labels'] = data['Labels'].map(label_mapping)

# Define features and target
features = ['Followers', 'Following', 'Following/Followers', 'Posts', 'Posts/Followers', 
            'Bio', 'Profile Picture', 'External Link', 'Mutual Friends', 'Threads']
X = data[features]
y = data['Labels']

# Handle any remaining missing values
X = X.fillna(0)

# Standardize numerical features
scaler = StandardScaler()
numerical_cols = ['Followers', 'Following', 'Following/Followers', 'Posts', 'Posts/Followers', 'Mutual Friends']
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train Random Forest Classifier with Grid Search
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_
print("Best Hyperparameters:", grid_search.best_params_)

# Evaluate the model
y_pred = best_model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Bot', 'Scam', 'Real', 'Spam']))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Feature Importance
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': best_model.feature_importances_
}).sort_values(by='Importance', ascending=False)
print("\nFeature Importance:")
print(feature_importance)

# ============== VISUALIZATION SECTION ==============
plt.figure(figsize=(15, 10))

# 1. Feature Distribution by Account Type
plt.subplot(2, 2, 1)
for label in [0, 1, 2, 3]:
    subset = data[data['Labels'] == label]
    plt.plot(subset['Following/Followers'].values[:100], 
             label=['Bot', 'Scam', 'Real', 'Spam'][label], alpha=0.7)
plt.title('Following/Followers Ratio by Account Type')
plt.xlabel('Sample Index')
plt.ylabel('Following/Followers Ratio')
plt.legend()
plt.yscale('log')  # Using log scale due to large value range

# 2. Posts Distribution
plt.subplot(2, 2, 2)
for label in [0, 1, 2, 3]:
    subset = data[data['Labels'] == label]
    plt.plot(subset['Posts'].values[:100], 
             label=['Bot', 'Scam', 'Real', 'Spam'][label], alpha=0.7)
plt.title('Number of Posts by Account Type')
plt.xlabel('Sample Index')
plt.ylabel('Number of Posts')
plt.legend()

# 3. Feature Importance
plt.subplot(2, 2, 3)
plt.plot(feature_importance['Feature'], feature_importance['Importance'], marker='o')
plt.title('Feature Importance')
plt.xlabel('Features')
plt.ylabel('Importance Score')
plt.xticks(rotation=45)
plt.grid(True)

# 4. Model Performance Metrics
metrics = classification_report(y_test, y_pred, target_names=['Bot', 'Scam', 'Real', 'Spam'], output_dict=True)
metrics_df = pd.DataFrame(metrics).transpose()
plt.subplot(2, 2, 4)
for metric in ['precision', 'recall', 'f1-score']:
    plt.plot(metrics_df.index[:-3], metrics_df[metric][:-3], marker='o', label=metric)
plt.title('Model Performance by Class')
plt.xlabel('Class')
plt.ylabel('Score')
plt.legend()
plt.ylim(0, 1.1)
plt.grid(True)

plt.tight_layout()
plt.show()

# ============== PREDICTION FUNCTION ==============
def predict_account(account_data, scaler=scaler, model=best_model):
    """
    Predict the label for a new Instagram account.
    account_data: dict with keys matching feature names
    Returns: Predicted label (Bot, Scam, Real, Spam)
    """
    # Create DataFrame for the new account
    df = pd.DataFrame([account_data], columns=features)
    
    # Preprocess categorical columns
    for col in categorical_cols:
        df[col] = df[col].str.lower().map({'yes': 1, 'no': 0})
    
    # Fill missing values
    df = df.fillna(0)
    
    # Standardize numerical features
    df[numerical_cols] = scaler.transform(df[numerical_cols])
    
    # Predict
    pred = model.predict(df)[0]
    reverse_mapping = {0: 'Bot', 1: 'Scam', 2: 'Real', 3: 'Spam'}
    return reverse_mapping[pred]

# Example prediction
new_account = {
    'Followers': 100,
    'Following': 5000,
    'Following/Followers': 50.0,
    'Posts': 0,
    'Posts/Followers': 0.0,
    'Bio': 'No',
    'Profile Picture': 'No',
    'External Link': 'No',
    'Mutual Friends': 0,
    'Threads': 'No'
}
prediction = predict_account(new_account)
print(f"\nPredicted Label for New Account: {prediction}")

# Load the dataset
data = pd.read_csv(r'C:\Users\Ritam Karmakar\Downloads\archive\Preprocessed_Instagram_Dataset.csv')

# Data Preprocessing
# Replace "#DIV/0!" with a large value (e.g., 1e6) for ratio columns
data['Following/Followers'] = data['Following/Followers'].replace('#DIV/0!', 1e6).astype(float)
data['Posts/Followers'] = data['Posts/Followers'].replace('#DIV/0!', 1e6).astype(float)

# Convert categorical columns to binary (Yes=1, No=0)
categorical_cols = ['Bio', 'Profile Picture', 'External Link', 'Threads']
for col in categorical_cols:
    data[col] = data[col].str.lower().map({'yes': 1, 'n': 0})

# Encode the target variable
label_mapping = {'Bot': 0, 'Scam': 1, 'Real': 2, 'Spam': 3}
data['Labels'] = data['Labels'].map(label_mapping)

# Define features and target
features = ['Followers', 'Following', 'Following/Followers', 'Posts', 'Posts/Followers', 
            'Bio', 'Profile Picture', 'External Link', 'Mutual Friends', 'Threads']
X = data[features]
y = data['Labels']

# Handle any remaining missing values
X = X.fillna(0)

# Standardize numerical features
scaler = StandardScaler()
numerical_cols = ['Followers', 'Following', 'Following/Followers', 'Posts', 'Posts/Followers', 'Mutual Friends']
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train Random Forest Classifier with Grid Search
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_
print("Best Hyperparameters:", grid_search.best_params_)

# Evaluate the model
y_pred = best_model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Bot', 'Scam', 'Real', 'Spam']))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Feature Importance
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': best_model.feature_importances_
}).sort_values(by='Importance', ascending=False)
print("\nFeature Importance:")
print(feature_importance)

# ============== VISUALIZATION SECTION ==============
plt.figure(figsize=(15, 10))

# 1. Feature Distribution by Account Type
plt.subplot(2, 2, 1)
for label in [0, 1, 2, 3]:
    subset = data[data['Labels'] == label]
    plt.plot(subset['Following/Followers'].values[:100], 
             label=['Bot', 'Scam', 'Real', 'Spam'][label], alpha=0.7)
plt.title('Following/Followers Ratio by Account Type')
plt.xlabel('Sample Index')
plt.ylabel('Following/Followers Ratio')
plt.legend()
plt.yscale('log')  # Using log scale due to large value range

# 2. Posts Distribution
plt.subplot(2, 2, 2)
for label in [0, 1, 2, 3]:
    subset = data[data['Labels'] == label]
    plt.plot(subset['Posts'].values[:100], 
             label=['Bot', 'Scam', 'Real', 'Spam'][label], alpha=0.7)
plt.title('Number of Posts by Account Type')
plt.xlabel('Sample Index')
plt.ylabel('Number of Posts')
plt.legend()

# 3. Feature Importance
plt.subplot(2, 2, 3)
plt.plot(feature_importance['Feature'], feature_importance['Importance'], marker='o')
plt.title('Feature Importance')
plt.xlabel('Features')
plt.ylabel('Importance Score')
plt.xticks(rotation=45)
plt.grid(True)

# 4. Model Performance Metrics
metrics = classification_report(y_test, y_pred, target_names=['Bot', 'Scam', 'Real', 'Spam'], output_dict=True)
metrics_df = pd.DataFrame(metrics).transpose()
plt.subplot(2, 2, 4)
for metric in ['precision', 'recall', 'f1-score']:
    plt.plot(metrics_df.index[:-3], metrics_df[metric][:-3], marker='o', label=metric)
plt.title('Model Performance by Class')
plt.xlabel('Class')
plt.ylabel('Score')
plt.legend()
plt.ylim(0, 1.1)
plt.grid(True)

plt.tight_layout()
plt.show()

# ============== PREDICTION FUNCTION ==============
def predict_account(account_data, scaler=scaler, model=best_model):
    """
    Predict the label for a new Instagram account.
    account_data: dict with keys matching feature names
    Returns: Predicted label (Bot, Scam, Real, Spam)
    """
    # Create DataFrame for the new account
    df = pd.DataFrame([account_data], columns=features)
    
    # Preprocess categorical columns
    for col in categorical_cols:
        df[col] = df[col].str.lower().map({'yes': 1, 'no': 0})
    
    # Fill missing values
    df = df.fillna(0)
    
    # Standardize numerical features
    df[numerical_cols] = scaler.transform(df[numerical_cols])
    
    # Predict
    pred = model.predict(df)[0]
    reverse_mapping = {0: 'Bot', 1: 'Scam', 2: 'Real', 3: 'Spam'}
    return reverse_mapping[pred]

# Example prediction
new_account = {
    'Followers': 100,
    'Following': 5000,
    'Following/Followers': 50.0,
    'Posts': 0,
    'Posts/Followers': 0.0,
    'Bio': 'No',
    'Profile Picture': 'No',
    'External Link': 'No',
    'Mutual Friends': 0,
    'Threads': 'No'
}
prediction = predict_account(new_account)
print(f"\nPredicted Label for New Account: {prediction}")