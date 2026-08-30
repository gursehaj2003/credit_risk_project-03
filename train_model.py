import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score, classification_report
import joblib
import warnings

warnings.filterwarnings('ignore')

# ==============================
# Load Dataset
# ==============================
CSV_PATH = "german_credit_data.csv"
print("🔍 Loading German Credit Dataset...")
df = pd.read_csv(CSV_PATH)
print(f"✅ Dataset loaded! Shape: {df.shape}")

# ==============================
# Data Cleaning
# ==============================
df.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')
df['Saving accounts'].fillna('unknown', inplace=True)
df['Checking account'].fillna('unknown', inplace=True)
df.dropna(subset=['Risk'], inplace=True)

# Encode target
le_target = LabelEncoder()
df['Risk'] = le_target.fit_transform(df['Risk'])

# ==============================
# Features
# ==============================
num_cols = ['Age', 'Credit amount', 'Duration']
cat_cols = ['Sex', 'Job', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']

X = df[num_cols + cat_cols]
y = df['Risk']

print(f"📊 Features: {len(X.columns)} | Classes: {y.nunique()}")
print(f"🎯 Class distribution:\n{y.value_counts()}")

# ==============================
# Train-Test Split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==============================
# Preprocessing Pipeline
# ==============================
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), cat_cols)
    ]
)

# Transform data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# ==============================
# Handle Imbalance with SMOTE
# ==============================
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_processed, y_train)

# ==============================
# Train Logistic Regression Model
# ==============================
model = LogisticRegression(
    C=1.0, penalty='l2', random_state=42, max_iter=1000
)
model.fit(X_train_smote, y_train_smote)

# ==============================
# Evaluate Model
# ==============================
y_pred_proba = model.predict_proba(X_test_processed)[:, 1]
y_pred = model.predict(X_test_processed)

print("\n" + "="*60)
print("🏆 MODEL RESULTS")
print("="*60)
print(f"📈 ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Good Credit', 'Bad Credit']))

# ==============================
# Save Model & Preprocessor
# ==============================
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/credit_model.pkl')
joblib.dump({
    'preprocessor': preprocessor,
    'num_cols': num_cols,
    'cat_cols': cat_cols,
    'target_encoder': le_target,
    'feature_names': preprocessor.get_feature_names_out()
}, 'models/feature_info.pkl')

print("\n✅ MODEL SAVED SUCCESSFULLY!")
print("🚀 Run: streamlit run app.py")
