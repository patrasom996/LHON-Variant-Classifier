import pandas as pd
import re
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut, cross_val_score

# 1. SETUP: Amino Acid Dictionary (Now with CHARGE!)
# Charge: Arg/Lys (+1), Asp/Glu (-1), His (+0.5)
aa_props = {
    'A': {'hydro': 1.8, 'mw': 89.1, 'c': 0},   'R': {'hydro': -4.5, 'mw': 174.2, 'c': 1},
    'N': {'hydro': -3.5, 'mw': 132.1, 'c': 0}, 'D': {'hydro': -3.5, 'mw': 133.1, 'c': -1},
    'C': {'hydro': 2.5, 'mw': 121.2, 'c': 0},  'Q': {'hydro': -3.5, 'mw': 146.2, 'c': 0},
    'E': {'hydro': -3.5, 'mw': 147.1, 'c': -1}, 'G': {'hydro': -0.4, 'mw': 75.1, 'c': 0},
    'H': {'hydro': -3.2, 'mw': 155.2, 'c': 0.5}, 'I': {'hydro': 4.5, 'mw': 131.2, 'c': 0},
    'L': {'hydro': 3.8, 'mw': 131.2, 'c': 0},  'K': {'hydro': -3.9, 'mw': 146.2, 'c': 1},
    'M': {'hydro': 1.9, 'mw': 149.2, 'c': 0},  'F': {'hydro': 2.8, 'mw': 165.2, 'c': 0},
    'P': {'hydro': -1.6, 'mw': 115.1, 'c': 0}, 'S': {'hydro': -0.8, 'mw': 105.1, 'c': 0},
    'T': {'hydro': -0.7, 'mw': 119.1, 'c': 0}, 'W': {'hydro': -0.9, 'mw': 204.2, 'c': 0},
    'Y': {'hydro': -1.3, 'mw': 181.2, 'c': 0}, 'V': {'hydro': 4.2, 'mw': 117.1, 'c': 0},
}

# 2. RELOAD DATA (We need to re-calculate features to include Charge)
# We assume lhon_data.csv exists (from Phase 1/Manual Step)
try:
    df_raw = pd.read_csv('lhon_data.csv')
except FileNotFoundError:
    print("Error: Could not find 'lhon_data.csv'.")
    exit()

# Add Benign Data (Augmentation) just like in Phase 2
benign_data = [
    {'Gene': 'MT-ND1', 'Protein_Change': 'A12T', 'Classification': 'Benign'},
    {'Gene': 'MT-ND4', 'Protein_Change': 'P25L', 'Classification': 'Benign'},
    {'Gene': 'MT-ND6', 'Protein_Change': 'I166V', 'Classification': 'Benign'},
    {'Gene': 'MT-ND1', 'Protein_Change': 'T12A', 'Classification': 'Benign'},
    {'Gene': 'MT-ND5', 'Protein_Change': 'L124F', 'Classification': 'Benign'},
    {'Gene': 'MT-ND4', 'Protein_Change': 'I165T', 'Classification': 'Benign'},
    {'Gene': 'MT-ND2', 'Protein_Change': 'V12I', 'Classification': 'Benign'},
    {'Gene': 'MT-ATP6', 'Protein_Change': 'T33A', 'Classification': 'Benign'},
]
df = pd.concat([df_raw, pd.DataFrame(benign_data)], ignore_index=True)

# 3. FEATURE ENGINEERING
data = []
pattern = re.compile(r"^([A-Z])(\d+)([A-Z])$")

print("Processing features...")
for _, row in df.iterrows():
    p_change = str(row['Protein_Change']).strip()
    classification = str(row['Classification']).lower()
    
    match = pattern.match(p_change)
    if not match: continue
    
    ref, pos, mut = match.groups()
    if ref not in aa_props or mut not in aa_props: continue
    
    label = 1 if 'pathogenic' in classification else 0
    
    # Calculate Features
    d_hydro = aa_props[mut]['hydro'] - aa_props[ref]['hydro']
    d_mw    = aa_props[mut]['mw'] - aa_props[ref]['mw']
    d_charge= aa_props[mut]['c'] - aa_props[ref]['c'] # NEW FEATURE
    
    data.append([d_hydro, d_mw, d_charge, label])

df_final = pd.DataFrame(data, columns=['Delta_Hydro', 'Delta_MW', 'Delta_Charge', 'Class'])

# 4. TRAIN & EVALUATE (SVM + LOOCV)
X = df_final[['Delta_Hydro', 'Delta_MW', 'Delta_Charge']]
y = df_final['Class']

# SVM works better for small datasets than Random Forest
model = SVC(kernel='rbf', class_weight='balanced', random_state=42)

# Leave-One-Out Cross Validation
# We train on N-1, test on 1, repeat for all rows.
loo = LeaveOneOut()
scores = cross_val_score(model, X, y, cv=loo)

print("-" * 30)
print(f"Total Variants: {len(df_final)}")
print(f"Benign: {len(df_final[df_final['Class']==0])}, Pathogenic: {len(df_final[df_final['Class']==1])}")
print("-" * 30)
print(f"New Accuracy (SVM + LOOCV): {scores.mean() * 100:.2f}%")
print("-" * 30)