import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# 1. Load your hand coordinates
print("Loading data from hand_landmarks.csv...")
df = pd.read_csv('data/hand_landmarks.csv', header=None)

# X = the numbers (the 63 coordinates), y = the words (HELLO, LOVE)
X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values

# 2. Split data into "Study Material" and "Test Exam"
# We use 80% to train and 20% to see if the AI actually learned
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y)

# 3. Initialize the Brain (Random Forest)
print("Teaching the AI... this should only take a few seconds.")
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# 4. Check the "Grade"
y_predict = model.predict(X_test)
score = accuracy_score(y_test, y_predict)

print(f"✅ Training Complete! Accuracy: {score * 100:.2f}%")

# 5. Save the Brain to a file
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)

print("Brain saved as 'model.p'!")