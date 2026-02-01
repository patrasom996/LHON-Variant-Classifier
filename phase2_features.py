import pandas as pd
import re

# 1. SETUP: Define Amino Acid Properties (Kyte-Doolittle)
# This dictionary converts letters to numbers
aa_props = {
    'A': {'hydro': 1.8, 'mw': 89.1},   'R': {'hydro': -4.5, 'mw': 174.2},
    'N': {'hydro': -3.5, 'mw': 132.1}, 'D': {'hydro': -3.5, 'mw': 133.1},
    'C': {'hydro': 2.5, 'mw': 121.2},  'Q': {'hydro': -3.5, 'mw': 146.2},
    'E': {'hydro': -3.5, 'mw': 147.1}, 'G': {'hydro': -0.4, 'mw': 75.1},
    'H': {'hydro': -3.2, 'mw': 155.2}, 'I': {'hydro': 4.5, 'mw': 131.2},
    'L': {'hydro': 3.8, 'mw': 131.2},  'K': {'hydro': -3.9, 'mw': 146.2},
    'M': {'hydro': 1.9, 'mw': 149.2},  'F': {'hydro': 2.8, 'mw': 165.2},
    'P': {'hydro': -1.6, 'mw': 115.1}, 'S': {'hydro': -0.8, 'mw': 105.1},
    'T': {'hydro': -0.7, 'mw': 119.1}, 'W': {'hydro': -0.9, 'mw': 204.2},
    'Y': {'hydro': -1.3, 'mw': 181.2}, 'V': {'hydro': 4.2, 'mw': 117.1},
}

# 2. LOAD & AUGMENT DATA
try:
    df = pd.read_csv('lhon_data.csv')
    print(f"Loaded {len(df)} variants from your file.")
except FileNotFoundError:
    print("Error: lhon_data.csv not found. Create it first!")
    exit()

# We manually add BENIGN variants to balance the dataset
# These are common neutral mitochondrial polymorphisms
benign_data = [
    {'Gene': 'MT-ND1', 'Protein_Change': 'A12T', 'Classification': 'Benign'},
    {'Gene': 'MT-ND4', 'Protein_Change': 'P25L', 'Classification': 'Benign'},
    {'Gene': 'MT-ND6', 'Protein_Change': 'I166V', 'Classification': 'Benign'},
    {'Gene': 'MT-ND1', 'Protein_Change': 'T12A', 'Classification': 'Benign'},
    {'Gene': 'MT-ND5', 'Protein_Change': 'L124F', 'Classification': 'Benign'}, # Reverse of pathogenic often benign-ish
    {'Gene': 'MT-ND4', 'Protein_Change': 'I165T', 'Classification': 'Benign'},
    {'Gene': 'MT-ND2', 'Protein_Change': 'V12I', 'Classification': 'Benign'},
    {'Gene': 'MT-ATP6', 'Protein_Change': 'T33A', 'Classification': 'Benign'},
]
df_benign = pd.DataFrame(benign_data)
df = pd.concat([df, df_benign], ignore_index=True)

print(f"Total variants after augmentation: {len(df)}")

# 3. FEATURE ENGINEERING LOOP
processed_data = []
mutation_pattern = re.compile(r"^([A-Z])(\d+)([A-Z])$")

for index, row in df.iterrows():
    p_change = str(row['Protein_Change']).strip()
    classification = str(row['Classification']).lower()
    
    # Filter: Only accept simple mutations (e.g., "Y98S")
    match = mutation_pattern.match(p_change)
    if not match:
        continue # Skip complex stuff like "V75fs"
        
    ref, pos, mut = match.groups()
    
    if ref not in aa_props or mut not in aa_props:
        continue

    # Label: 1 = Pathogenic, 0 = Benign
    label = 1 if 'pathogenic' in classification else 0
    
    # MATH: Calculate the differences
    hydro_diff = aa_props[mut]['hydro'] - aa_props[ref]['hydro']
    mw_diff    = aa_props[mut]['mw'] - aa_props[ref]['mw']
    
    processed_data.append({
        'Gene': row['Gene'],
        'Mutation': p_change,
        'Delta_Hydro': hydro_diff,
        'Delta_MW': mw_diff,
        'Class': label
    })

# 4. SAVE RESULTS
df_final = pd.DataFrame(processed_data)
df_final.to_csv('lhon_features.csv', index=False)

print("\nSUCCESS! Phase 2 Complete.")
print(df_final['Class'].value_counts())
print("\nFirst 5 rows of your numerical dataset:")
print(df_final.head())