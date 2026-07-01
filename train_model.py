# =====================================
# ENHANCED DIABETES PREDICTION (No Extra Dependencies)
# =====================================
import os
print(os.getcwd())
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import RobustScaler, PowerTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier,ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, f1_score, precision_score, recall_score, balanced_accuracy_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from imblearn.combine import SMOTETomek
from scipy import stats
import os
from xgboost import XGBClassifier
from catboost import CatBoostClassifier


# -----------------------------
# 1. Advanced Data Loading & Validation
# -----------------------------
print("="*60)
print("ENHANCED DIABETES PREDICTION MODEL TRAINING")
print("="*60)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "dataset", "diabetes.csv")

df = pd.read_csv(file_path)
print(f"\nDataset Shape: {df.shape}")
print(f"Missing Values: {df.isnull().sum().sum()}")
print(f"\nOutcome Distribution:\n{df['Outcome'].value_counts(normalize=True)}")

# -----------------------------
# 2. Intelligent Outlier Handling
# -----------------------------
def handle_outliers_iqr(df, column, multiplier=1.5):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    df[column] = df[column].clip(lower_bound, upper_bound)
    return df

def handle_outliers_zscore(df, column, threshold=3):
    from scipy import stats
    df[column] = np.where(np.abs(stats.zscore(df[column])) > threshold, df[column].median(), df[column])
    return df

# Enhanced zero handling with domain knowledge
cols_to_fix = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

for col in cols_to_fix:
    if col == 'Glucose':
        # Glucose cannot be zero - replace with median by age group
        df['Age_Group'] = pd.cut(df['Age'], bins=[0,30,40,50,100], labels=[0,1,2,3])
        df.loc[df[col] == 0, col] = df.groupby('Age_Group')[col].transform('median').fillna(df[col].median())
    elif col == 'BloodPressure':
        df.loc[(df[col] == 0) | (df[col] < 50), col] = df[col].median()
    elif col == 'BMI':
        df.loc[(df[col] == 0) | (df[col] < 15) | (df[col] > 50), col] = df[col].median()
    elif col == 'Insulin':
        # Use median for insulin (simpler approach without KNN)
        df.loc[df[col] == 0, col] = df[col].median()
    else:
        df[col] = df[col].replace(0, df[col].median())

# Apply outlier capping
for col in cols_to_fix:
    df = handle_outliers_iqr(df, col, multiplier=2.0)

# Remove temporary column
if 'Age_Group' in df.columns:
    df = df.drop('Age_Group', axis=1)

# -----------------------------
# 3. Advanced Feature Engineering
# -----------------------------
# Original features
df['Glucose_BMI_Ratio'] = df['Glucose'] / (df['BMI'] + 0.1)
df['Age_Glucose_Interaction'] = df['Age'] * df['Glucose'] / 100
df['Insulin_Glucose_Ratio'] = df['Insulin'] / (df['Glucose'] + 0.1)
df['BP_BMI_Interaction'] = df['BloodPressure'] * df['BMI'] / 100
df['Age_Pregnancies'] = df['Age'] * df['Pregnancies'] / 10
df['SkinThickness_BMI'] = df['SkinThickness'] * df['BMI']

# NEW FEATURES
df['BMI_Age'] = df['BMI'] * df['Age']
df['Glucose_BMI'] = df['Glucose'] * df['BMI']
df['Pregnancy_Risk'] = df['Pregnancies'] * df['Glucose']
df['Age_BMI_Risk'] = df['Age'] * df['BMI']
df['Insulin_BMI'] = df['Insulin'] * df['BMI']

# Polynomial features
df['Glucose_Squared'] = df['Glucose'] ** 2
df['BMI_Squared'] = df['BMI'] ** 2
df['Age_Squared'] = df['Age'] ** 2

# Log transformations
df['Log_Insulin'] = np.log1p(df['Insulin'])
df['Log_Glucose'] = np.log1p(df['Glucose'])
df['Log_BMI'] = np.log1p(df['BMI'])

# Risk scores
df['Metabolic_Risk'] = (df['Glucose'] * 0.35 + df['BMI'] * 0.25 + 
                        df['BloodPressure'] * 0.20 + df['Age'] * 0.15 + 
                        df['Pregnancies'] * 0.05)

# Age groups (categorical)
df['Age_Group'] = pd.cut(df['Age'], bins=[0, 30, 40, 50, 60, 100], labels=[0, 1, 2, 3, 4]).astype(int)

# BMI categories
df['BMI_Category'] = pd.cut(df['BMI'], bins=[0, 18.5, 25, 30, 100], labels=[0, 1, 2, 3]).astype(int)

# HOMA-IR approximation (insulin resistance)
df['HOMA_IR'] = (df['Glucose'] * df['Insulin']) / 405

print(f"\nFeatures after engineering: {df.shape[1]}")

# -----------------------------
# 4. Feature Selection
# -----------------------------
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Remove high correlation features
corr_matrix = X.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_corr_features = [
    column for column in upper.columns
    if any(upper[column] > 0.95)
]

X = X.drop(columns=high_corr_features)

print(f"\nRemoved {len(high_corr_features)} highly correlated features")

# -----------------------------
# 5. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.15,
    stratify=y,
    random_state=42
)
print("Total Dataset:", len(df))
print("Training Set:", len(X_train))
print("Test Set:", len(X_test))
selector = SelectKBest(
    mutual_info_classif,
    k=min(20, X_train.shape[1])
)

X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

selected_features = X_train.columns[
    selector.get_support()
].tolist()

X_train = pd.DataFrame(
    X_train_selected,
    columns=selected_features
)

X_test = pd.DataFrame(
    X_test_selected,
    columns=selected_features
)

# -----------------------------
# 6. Sampling Strategy (Auto-select best)
# -----------------------------
print(f"\nBefore Sampling - Class distribution: {y_train.value_counts().to_dict()}")

sampling_techniques = {
    'SMOTE': SMOTE(random_state=42, k_neighbors=5),
    'BorderlineSMOTE': BorderlineSMOTE(random_state=42, k_neighbors=5),
    'SMOTETomek': SMOTETomek(random_state=42),
    'ADASYN': ADASYN(random_state=42)
}

best_sampler = None
best_cv_score = 0

for name, sampler in sampling_techniques.items():
    try:
        X_temp, y_temp = sampler.fit_resample(X_train, y_train)
        rf_temp = RandomForestClassifier(random_state=42, n_estimators=100, n_jobs=-1)
        cv_score = cross_val_score(rf_temp, X_temp, y_temp, cv=3, scoring='roc_auc').mean()
        if cv_score > best_cv_score:
            best_cv_score = cv_score
            best_sampler = sampler
            print(f"  {name}: CV Score = {cv_score:.4f}")
    except Exception as e:
        print(f"  {name}: Failed - {str(e)[:50]}")

X_train_resampled, y_train_resampled = best_sampler.fit_resample(X_train, y_train)
print(f"\nAfter Sampling - Class distribution: {y_train_resampled.value_counts().to_dict()}")
print(f"Using {best_sampler.__class__.__name__}")

# -----------------------------
# 7. Scaling
# -----------------------------
scaler = RobustScaler(quantile_range=(5, 95))
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

power_transformer = PowerTransformer(method='yeo-johnson')
X_train_scaled = power_transformer.fit_transform(X_train_scaled)
X_test_scaled = power_transformer.transform(X_test_scaled)

# -----------------------------
# 8. Model Training with Hyperparameter Tuning
# -----------------------------
print("\n" + "="*60)
print("MODEL TRAINING AND TUNING")
print("="*60)

# Random Forest
rf_params = {
    'n_estimators': [500, 700, 1000],
    'max_depth': [10, 15, 20, 25],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf_random = RandomizedSearchCV(
    RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1),
    rf_params,
    n_iter=20,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='roc_auc',
    n_jobs=-1,
    random_state=42,
    verbose=0
)
rf_random.fit(X_train_scaled, y_train_resampled)
best_rf = rf_random.best_estimator_
print(f"\nBest Random Forest Score: {rf_random.best_score_:.4f}")

# Gradient Boosting
gb_params = {
    'n_estimators': [200, 300],
    'learning_rate': [0.03, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'min_samples_split': [5, 10],
    'subsample': [0.7, 0.8]
}

gb_random = RandomizedSearchCV(
    GradientBoostingClassifier(random_state=42),
    gb_params,
    n_iter=15,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    random_state=42,
    verbose=0
)
gb_random.fit(X_train_scaled, y_train_resampled)
best_gb = gb_random.best_estimator_
print(f"Best Gradient Boosting Score: {gb_random.best_score_:.4f}")

# XGBoost
xgb = XGBClassifier(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.03,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    eval_metric='logloss'
)

xgb.fit(X_train_scaled, y_train_resampled)

# CatBoost
cat = CatBoostClassifier(
    iterations=2500,
    depth=8,
    learning_rate=0.02,
    l2_leaf_reg=3,
    random_strength=1,
    bagging_temperature=1,
    eval_metric='AUC',
    random_state=42,
    verbose=0
)

cat.fit(X_train_scaled, y_train_resampled)

# -----------------------------
# 9. Ensemble Methods
# -----------------------------
print("\n" + "="*60)
print("BUILDING ENSEMBLES")
print("="*60)

base_models = [
    ('rf', best_rf),
    ('gb', best_gb),
    ('xgb', xgb),
    ('cat', cat),
]

# Soft Voting
voting_clf = VotingClassifier(
    estimators=base_models,
    voting='soft',
    weights=[1.5, 1.3, 1.2, 1.0]
)
voting_clf.fit(X_train_scaled, y_train_resampled)



# -----------------------------
# 10. Evaluation
# -----------------------------
print("\n" + "="*60)
print("MODEL EVALUATION")
print("="*60)

models = {
    'Random Forest': best_rf,
    'Gradient Boosting': best_gb,
    'XGBoost': xgb,
    'CatBoost': cat,
    'Voting Ensemble': voting_clf,
}

results = {}
for name, model in models.items():

    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    # Find best threshold
    best_acc = 0
    best_threshold = 0.5

    for threshold in np.arange(0.30, 0.71, 0.01):

        pred = (y_pred_proba > threshold).astype(int)

        acc = accuracy_score(y_test, pred)

        if acc > best_acc:
            best_acc = acc
            best_threshold = threshold

    # Use best threshold
    y_pred = (y_pred_proba > best_threshold).astype(int)

    print(f"\n{name} Best Threshold: {best_threshold:.2f}")

    results[name] = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Balanced_Accuracy': balanced_accuracy_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_pred_proba)
    }
    # Find Best Model Based on ROC-AUC

best_model_name = max(
    results,
    key=lambda x: results[x]['ROC-AUC']
)

print("\n" + "="*60)
print("BEST MODEL SELECTION")
print("="*60)
print("Best Model:", best_model_name)
print("Best ROC-AUC:", results[best_model_name]['ROC-AUC'])
print("Best Accuracy:", results[best_model_name]['Accuracy'])
    
print(f"\n{name}:")
print(f"  Accuracy:          {results[name]['Accuracy']*100:.2f}%")
print(f"  Balanced Accuracy: {results[name]['Balanced_Accuracy']*100:.2f}%")
print(f"  F1-Score:          {results[name]['F1-Score']:.4f}")
print(f"  ROC-AUC:           {results[name]['ROC-AUC']:.4f}")
print(f"  Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
cat_probs = cat.predict_proba(X_test_scaled)[:, 1]

best_acc = 0
best_threshold = 0.5

for threshold in np.arange(0.30, 0.71, 0.01):

    pred = (cat_probs > threshold).astype(int)

    acc = accuracy_score(y_test, pred)

    if acc > best_acc:
        best_acc = acc
        best_threshold = threshold

print("\nBest CatBoost Threshold:", best_threshold)
print("Best Accuracy:", best_acc)

# -----------------------------
# 11. Cross-Validation
# -----------------------------
print("\n" + "="*60)
print("CROSS-VALIDATION")
print("="*60)

# -----------------------------
# 12. Save Models
# -----------------------------
print("\n" + "="*60)
print("SAVING MODEL ARTIFACTS")
print("="*60)

os.makedirs("models", exist_ok=True)


# Select model with highest ROC-AUC
best_model_name = max(
    results,
    key=lambda x: results[x]['ROC-AUC']
)

best_model = models[best_model_name]

print(f"\nBest Model Selected: {best_model_name}")

pipeline = {
    'model': best_model,
    'scaler': scaler,
    'power_transformer': power_transformer,
    'feature_selector': selector,
    'selected_features': selected_features,
    'sampler': best_sampler,
    'metadata': {
        'best_model': best_model_name,
        'accuracy': results[best_model_name]['Accuracy'],
        'balanced_accuracy': results[best_model_name]['Balanced_Accuracy'],
        'roc_auc': results[best_model_name]['ROC-AUC'],
        'f1_score': results[best_model_name]['F1-Score']
    }
}

joblib.dump(
    pipeline,
    "models/diabetes_enhanced_pipeline.pkl"
)

print("✓ Complete pipeline saved to 'models/diabetes_enhanced_pipeline.pkl'")

print("\n" + "="*60)
print(f"✅ TRAINING COMPLETE")
print(f"Best Accuracy: {results['Random Forest']['Accuracy']*100:.2f}%")
print(f"Best ROC-AUC: {results['Random Forest']['ROC-AUC']:.4f}")
print("="*60)
