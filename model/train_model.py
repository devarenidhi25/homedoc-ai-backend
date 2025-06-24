import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
import joblib
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("dataset/DiseaseAndSymptoms.csv")
df.fillna("", inplace=True)
df["Disease"] = df["Disease"].str.replace(r"[^a-zA-Z0-9\s]", "", regex=True).str.strip()

# Extract diseases and symptoms
diseases = df["Disease"]
symptoms_list = df.drop("Disease", axis=1).values.tolist()

# Convert to format like [["Fever", "Cough"], ["Headache"], ...]
symptoms = [[sym.strip() for sym in row if sym] for row in symptoms_list]

# Encode symptoms using MultiLabelBinarizer
mlb = MultiLabelBinarizer()
X = mlb.fit_transform(symptoms)

# Encode disease labels
le = LabelEncoder()
y_encoded = le.fit_transform(diseases)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Define the parameter grid for GridSearchCV
param_grid = {
    'n_neighbors': range(1, 21),  # Test k from 1 to 20
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

# Initialize GridSearchCV
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)

# Fit GridSearchCV to the training data
print("Starting GridSearchCV for hyperparameter tuning...")
grid_search.fit(X_train, y_train)

# Get the best model
model = grid_search.best_estimator_

print(f"\n✅ Best parameters found: {grid_search.best_params_}")
print(f"✅ Best cross-validation accuracy: {grid_search.best_score_:.4f}")

# Evaluate the best model on the test set
y_pred = model.predict(X_test)

print("\n✅ Classification Report on Test Set:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Save model and encoders
joblib.dump(model, "model/knn_model.pkl")
joblib.dump(mlb, "model/symptom_binarizer.pkl")
joblib.dump(le, "model/label_encoder.pkl")

print("\n✅ Model and encoders saved.")