import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. LOAD DATA
filename = 'lhon_features.csv'
try:
    df = pd.read_csv(filename)
except FileNotFoundError:
    print(f"Error: Could not find '{filename}'. Did you run Phase 2?")
    exit()

# Safety Check: Ensure we actually have both classes
counts = df['Class'].value_counts()
if len(counts) < 2:
    print("CRITICAL ERROR: Your dataset only has one class (either all Pathogenic or all Benign).")
    print("You must re-run the Phase 2 script to add the Benign data first.")
    exit()

# 2. PREPARE INPUTS
features = ['Delta_Hydro', 'Delta_MW']
X = df[features]
y = df['Class']

print(f"Dataset size: {len(df)} variants")
print(f"Class balance: {dict(counts)}")

# 3. SPLIT DATA (Stratified)
# stratify=y ensures the Test set gets a mix of Benign and Pathogenic, just like the full data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 4. TRAIN
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 5. EVALUATE
predictions = clf.predict(X_test)
acc = accuracy_score(y_test, predictions)

print("-" * 30)
print(f"Model Accuracy: {acc * 100:.2f}%")
print("-" * 30)

# Robust Report Printing
unique_classes = sorted(list(set(y_test) | set(predictions)))
target_names = ["Benign", "Pathogenic"]

# Only pass the names that actually exist in the test results
current_names = [target_names[i] for i in unique_classes]

print("\nDetailed Report:")
print(classification_report(y_test, predictions, labels=unique_classes, target_names=current_names))

# 6. LIVE PREDICTION TEST
# Let's test a hypothetical mutation: Arginine (R) -> Histidine (H)
# Hydro change: -3.2 - (-4.5) = +1.3
# MW change: 155.2 - 174.2 = -19.0
new_mutation_values = [[1.3, -19.0]]
# We use a dataframe to match the feature names and avoid warnings
new_mutation = pd.DataFrame(new_mutation_values, columns=features)
prediction = clf.predict(new_mutation)
result = "Pathogenic" if prediction[0] == 1 else "Benign"

print("-" * 30)
print(f"Live Test (Arg->His): Predicted as {result}")