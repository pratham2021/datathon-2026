import pandas as pd

# ============================================================
# STEP 1: Load Consumer Data
# Update this path to wherever your CSV file is saved
# ============================================================
consumer = pd.read_csv('ConsumerData.csv')

print(f'Consumer: {consumer.shape[0]:,} rows x {consumer.shape[1]} columns')


# ============================================================
# STEP 2: Clean and Keep Only Selected Features
# Features selected based on correlation analysis
# ============================================================
c = consumer.copy()

# --- Encode OwnerRenter ---
# O = Owner (1), R = Renter (0), NaN = unknown (0)
c['OwnerRenter'] = c['OwnerRenter'].map({'O': 1, 'R': 0}).fillna(0).astype(int)

# --- Encode MaritalStatus ---
# We only need Marital_M and Marital_S based on correlation analysis
# M = Married, S = Single
marital_dummies = pd.get_dummies(c['MaritalStatus'], prefix='Marital', dummy_na=False)
c = pd.concat([c.drop(columns='MaritalStatus'), marital_dummies], axis=1)

# --- Encode Y/N flag columns -> 1/0 ---
# NaN = not known, treated as 0
yn_cols = ['Gardening', 'HomeImprovement', 'HomeImprovementDIY',
           'OutdoorsGrouping', 'SingleParent']
for col in yn_cols:
    if col in c.columns:
        c[col] = c[col].map({'Y': 1, 'N': 0}).fillna(0).astype(int)

# --- Fill nulls in numeric columns ---
c['HouseholdSize']           = c['HouseholdSize'].fillna(c['HouseholdSize'].median())
c['NumberOfChildren']        = c['NumberOfChildren'].fillna(0)
c['NetWorth']                = c['NetWorth'].fillna(c['NetWorth'].median())
c['VehicleKnownOwnedNumber'] = c['VehicleKnownOwnedNumber'].fillna(0)

print('Columns encoded and nulls filled.')


# ============================================================
# STEP 3: Keep Only Selected Columns
# MAK is kept as the join key for merging with property data
# ============================================================
selected_cols = [
    'MAK',
    'HouseholdSize',
    'VehicleKnownOwnedNumber',
    'NetWorth',
    'Marital_S',
    'Marital_M',
    'Gardening',
    'HomeImprovement',
    'OutdoorsGrouping',
    'NumberOfChildren',
    'OwnerRenter',
    'SingleParent',
    'HomeImprovementDIY',
]

# Only keep columns that exist (Marital_S / Marital_M may not exist if no one has that status)
selected_cols = [col for col in selected_cols if col in c.columns]
c = c[selected_cols]

print(f'\nFinal consumer data: {c.shape[0]:,} rows x {c.shape[1]} columns')
print('\nColumns kept:')
for col in c.columns:
    print(f'  {col}')

# --- Final null check ---
remaining = c.isnull().sum()
remaining = remaining[remaining > 0]
print('\nRemaining nulls:', remaining.to_dict() if len(remaining) > 0 else 'None!')


# ============================================================
# STEP 4: Save
# ============================================================
c.to_csv('consumer_clean.csv', index=False)
print('\nSaved: consumer_clean.csv')
print('\nNext step: run the model script once property_clean.csv is ready!')