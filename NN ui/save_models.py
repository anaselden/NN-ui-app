"""
Script to save trained models and preprocessors for the UI
Run this script once after training your models to save them
"""
import pandas as pd
import numpy as np
import pickle
import joblib

from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load Dataset
print("Loading dataset...")
car = fetch_ucirepo(id=19)
X = car.data.features.copy()
y = car.data.targets.copy().iloc[:, 0]

df = X.copy()
df["class"] = y

# Handle missing values
for col in df.columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# Save original feature values before encoding (for UI dropdowns)
feature_cols = df.columns[:-1]
original_features = {}
for col in feature_cols:
    original_features[col] = sorted(df[col].unique().tolist())

# Encoding
encoder = OrdinalEncoder()
df[feature_cols] = encoder.fit_transform(df[feature_cols])

label_enc = LabelEncoder()
df["class"] = label_enc.fit_transform(df["class"])

# Splitting Data
X = df.drop("class", axis=1)
y = df["class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model 1: Neural Network (MLP)
print("Training Neural Network...")
num_classes = len(np.unique(y_train))

model = Sequential([
    Dense(32, activation='relu', input_dim=X_train_scaled.shape[1]),
    Dense(16, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    verbose=1
)

# Model 2: SVM
print("Training SVM...")
svm_model = SVC(kernel='rbf', gamma='scale', C=1.0)
svm_model.fit(X_train_scaled, y_train)

# Save models and preprocessors
print("Saving models and preprocessors...")

# Save Neural Network model
model.save('neural_network_model.h5')
print("✓ Saved neural_network_model.h5")

# Save SVM model
joblib.dump(svm_model, 'svm_model.joblib')
print("✓ Saved svm_model.joblib")

# Save preprocessors
joblib.dump(encoder, 'ordinal_encoder.joblib')
print("✓ Saved ordinal_encoder.joblib")

joblib.dump(label_enc, 'label_encoder.joblib')
print("✓ Saved label_encoder.joblib")

joblib.dump(scaler, 'standard_scaler.joblib')
print("✓ Saved standard_scaler.joblib")

# Save feature column names for reference
with open('feature_columns.pkl', 'wb') as f:
    pickle.dump(feature_cols.tolist(), f)
print("✓ Saved feature_columns.pkl")

# Save original feature values for UI dropdowns (already captured before encoding)
with open('original_feature_values.pkl', 'wb') as f:
    pickle.dump(original_features, f)
print("✓ Saved original_feature_values.pkl")

# Get original class labels
class_labels = label_enc.classes_.tolist()
with open('class_labels.pkl', 'wb') as f:
    pickle.dump(class_labels, f)
print("✓ Saved class_labels.pkl")

print("\n✅ All models and preprocessors saved successfully!")
print("\nYou can now run the UI with: streamlit run app.py")

