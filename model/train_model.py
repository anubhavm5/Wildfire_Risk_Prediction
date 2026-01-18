import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from scripts.data_loader import load_data
from scripts.feature_engineering import add_features

print("Loading data...")
df = load_data()
df = add_features(df)

if "fire_occurred" not in df.columns:
    raise ValueError("'fire_occurred' column missing!")

X = df.drop(columns=["fire_occurred", "date"], errors="ignore")
y = df["fire_occurred"]

print("Fire counts (before balancing):", y.value_counts().to_dict())

# Handle imbalance
if y.nunique() < 2:
    print("Only one class detected! Injecting synthetic non-fire samples...")
    synthetic = X.sample(n=min(50, len(X)), random_state=42).copy()
    synthetic["fire_occurred"] = 0
    df_balanced = pd.concat([X, y], axis=1)
    df_balanced = pd.concat([df_balanced, synthetic], ignore_index=True)
    X = df_balanced.drop(columns=["fire_occurred"])
    y = df_balanced["fire_occurred"]
    print("Fire counts (after injection):", y.value_counts().to_dict())
else:
    print("Applying SMOTE balancing...")
    smote = SMOTE(random_state=42)
    X, y = smote.fit_resample(X, y)
    print("Fire counts (after SMOTE):", y.value_counts().to_dict())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scale_pos_weight = (y_train.value_counts().get(0, 1) /
                    max(y_train.value_counts().get(1, 1), 1))

print(f"\nTraining model with scale_pos_weight={scale_pos_weight:.2f}...")

model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss"
)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print("\n📊 Model Evaluation:")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, zero_division=0))
print("Recall   :", recall_score(y_test, y_pred, zero_division=0))
print("F1 Score :", f1_score(y_test, y_pred, zero_division=0))

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))

pickle.dump(model, open("artifacts/wildfire_model.pkl", "wb"))
print("\n✅ Model trained and saved at artifacts/wildfire_model.pkl")
