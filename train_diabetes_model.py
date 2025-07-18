import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

df = pd.read_csv("data/diabetes.csv")
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# ✅ Save with correct features
pickle.dump(model, open("models/diabetes_model.pkl", "wb"))

# Debug: Print feature info
print("🔍 Feature columns used for training:")
print(X.columns.tolist())
print(f"🔢 Number of features used for training: {X.shape[1]}")
print(f"🧠 Model expects {model.n_features_in_} features during prediction")
