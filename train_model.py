import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. LOAD DATA
DATA_PATH = 'data/holistic_data.csv'

try:
    df = pd.read_csv(DATA_PATH, header=None)
    print(f"✅ Data loaded: {df.shape[0]} samples found.")
except FileNotFoundError:
    print("❌ Error: holistic_data.csv not found. Did you record signs yet?")
    exit()

# 2. PREPARE FEATURES & LABELS
# First column is the Label (the word), the rest are landmarks
X = df.iloc[:, 1:].values  # Landmarks
y = df.iloc[:, 0].values   # Labels (Words)

# 3. SPLIT DATA
# We use 80% to teach the AI and 20% to test it
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, stratify=y
)

# 4. TRAIN MODEL
print("🧠 Training the Holistic Brain... please wait...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

# 5. EVALUATE
y_predict = model.predict(x_test)
score = accuracy_score(y_test, y_predict)
print(f"🎯 Training Complete! Accuracy: {score * 100:.2f}%")

# 6. SAVE MODEL AND LABELS
# We save the unique labels so the App knows which index belongs to which word
unique_labels = np.unique(y).tolist()

with open('model.p', 'wb') as f:
    pickle.dump({'model': model, 'labels': unique_labels}, f)

print("💾 Model saved as 'model.p'. Your App is ready to go!")