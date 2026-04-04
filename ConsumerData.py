import pandas as pd

# ============================================================
# STEP 1: Load Consumer Data
# Update this path to wherever your CSV file is saved
# ============================================================
consumer = pd.read_csv('ConsumerData.csv')

print(f'Consumer: {consumer.shape[0]:,} rows x {consumer.shape[1]} columns')


# ============================================================
# STEP 2: Clean Consumer Data
# ============================================================
c = consumer.copy()

# --- Y/N flag columns -> encode to 1/0 ---
# NaN means 'not known', treated as 0
# Includes Veteran, SingleParent, GrandChildren which are Y/N flags
# (not numeric counts as they might appear)
yn_cols_consumer = [
    'Charitable', 'Health', 'Political', 'Religious',
    'Veteran', 'SingleParent', 'GrandChildren',
    'Gardening', 'HomeImprovement', 'HomeImprovementDIY',
    'CatOwner', 'DogOwner', 'OutdoorsGrouping',
    'SelfImprovement', 'MusicCollector', 'MovieCollector',
    'Photography', 'AutoWork', 'Fishing', 'CampingHiking',
    'HuntingShooting', 'EnvironmentalIssues', 'InvestmentsForeign',
    'BeautyCosmetics', 'TVCable', 'WirelessCellularPhoneOwner',
    'CreditCardUser', 'EducationOnline'
]
for col in yn_cols_consumer:
    if col in c.columns:
        c[col] = c[col].map({'Y': 1, 'N': 0}).fillna(0).astype(int)

print('Y/N columns encoded.')

# --- Encode OwnerRenter ---
# O = Owner (1), R = Renter (0), NaN = unknown (0)
c['OwnerRenter'] = c['OwnerRenter'].map({'O': 1, 'R': 0}).fillna(0).astype(int)

# --- Encode MaritalStatus ---
# M = Married, S = Single, A = Inferred Married, B = Inferred Single
# One-hot encoded so no false ordering is implied
# You'll get 4 columns: Marital_M, Marital_S, Marital_A, Marital_B
marital_dummies = pd.get_dummies(c['MaritalStatus'], prefix='Marital', dummy_na=False)
c = pd.concat([c.drop(columns='MaritalStatus'), marital_dummies], axis=1)

print('OwnerRenter + MaritalStatus encoded.')
print('  Marital_M = Married')
print('  Marital_S = Single')
print('  Marital_A = Inferred Married')
print('  Marital_B = Inferred Single')

# --- Fill nulls in numeric columns ---
# HouseholdSize = number of people in household
c['HouseholdSize']           = c['HouseholdSize'].fillna(c['HouseholdSize'].median())

# NumberOfChildren = count of children
c['NumberOfChildren']        = c['NumberOfChildren'].fillna(0)

# NetWorth = 1-9 tier (1=<=0, 9=$500k+)
c['NetWorth']                = c['NetWorth'].fillna(c['NetWorth'].median())

# VehicleKnownOwnedNumber = count of vehicles
c['VehicleKnownOwnedNumber'] = c['VehicleKnownOwnedNumber'].fillna(0)

# HomePurchaseDate = YYYYMM format, left as-is for now
# (decide later whether to use it)

print('Numeric nulls filled.')

# --- Final check ---
remaining = c.isnull().sum()
remaining = remaining[remaining > 0]
print('\nRemaining nulls:', remaining.to_dict() if len(remaining) > 0 else 'None!')
print(f'\nCleaned consumer data: {c.shape[0]:,} rows x {c.shape[1]} columns')
print('\nAll columns:')
for col in c.columns:
    print(f'  {col}')


# ============================================================
# STEP 3: Save
# ============================================================
c.to_csv('consumer_clean.csv', index=False)
print('\nSaved: consumer_clean.csv')
print('\nNext step: upload property_clean.csv and we will run')
print('the correlation analysis to find which consumer attributes')
print('matter most for predicting home type.')