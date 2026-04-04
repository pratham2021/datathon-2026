import pandas as pd
import numpy as np

# ============================================================
# STEP 1: Load Data
# Make sure these files are in the same folder as this script
# ============================================================
consumer = pd.read_csv('ConsumerData.csv')
prop     = pd.read_csv('PropertyAssessmentData_cleaned.csv')

print(f'Consumer: {consumer.shape[0]:,} rows x {consumer.shape[1]} columns')
print(f'Property: {prop.shape[0]:,} rows x {prop.shape[1]} columns')


# ============================================================
# STEP 2: Clean Consumer Data (same as 01_consumer_cleaning.py)
# ============================================================
c = consumer.copy()

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

c['OwnerRenter'] = c['OwnerRenter'].map({'O': 1, 'R': 0}).fillna(0).astype(int)
marital_dummies  = pd.get_dummies(c['MaritalStatus'], prefix='Marital', dummy_na=False)
c = pd.concat([c.drop(columns='MaritalStatus'), marital_dummies], axis=1)
c['HouseholdSize']           = c['HouseholdSize'].fillna(c['HouseholdSize'].median())
c['NumberOfChildren']        = c['NumberOfChildren'].fillna(0)
c['NetWorth']                = c['NetWorth'].fillna(c['NetWorth'].median())
c['VehicleKnownOwnedNumber'] = c['VehicleKnownOwnedNumber'].fillna(0)

print('Consumer data cleaned.')


# ============================================================
# STEP 3: Merge
# ============================================================
merged = pd.merge(c, prop, on='MAK', how='inner')
print(f'Merged: {merged.shape[0]:,} rows')


# ============================================================
# STEP 4: Define Features and Targets
# ============================================================

# Property characteristics we want to predict / match against
property_targets = [
    'NumberOfBedrooms',
    'NumberOfBaths',
    'LotSizeOrArea',
    'TotalAssessedValue',
    'YearBuilt',
    'GarageParkingNumberOfCars',
    'NumberOfPartialBaths',
    'SalesPriceFromAssessment',
]

# Consumer attributes to test
consumer_features = [
    'HouseholdSize', 'NumberOfChildren', 'NetWorth', 'OwnerRenter',
    'VehicleKnownOwnedNumber',
    'Charitable', 'Health', 'Political', 'Religious', 'Veteran',
    'SingleParent', 'GrandChildren', 'Gardening', 'HomeImprovement',
    'HomeImprovementDIY', 'CatOwner', 'DogOwner', 'OutdoorsGrouping',
    'SelfImprovement', 'MusicCollector', 'MovieCollector', 'Photography',
    'AutoWork', 'Fishing', 'CampingHiking', 'HuntingShooting',
    'EnvironmentalIssues', 'InvestmentsForeign', 'BeautyCosmetics',
    'TVCable', 'WirelessCellularPhoneOwner', 'CreditCardUser',
    'EducationOnline',
    'Marital_A', 'Marital_B', 'Marital_M', 'Marital_S'
]

# Only keep features that actually exist in the merged dataset
consumer_features = [f for f in consumer_features if f in merged.columns]
property_targets  = [t for t in property_targets  if t in merged.columns]


# ============================================================
# STEP 5: Correlation Analysis
# ============================================================
print('\n' + '='*60)
print('CORRELATION REPORT')
print('='*60)
print('Scale: -1.0 to +1.0')
print('  +value = consumer trait links to HIGHER property value')
print('  -value = consumer trait links to LOWER property value')
print('  Closer to 0 = weak/no relationship')
print('='*60)

# Build a full correlation matrix: consumer features vs property targets
results = {}
for feat in consumer_features:
    row = {}
    for target in property_targets:
        corr = merged[feat].corr(merged[target])
        row[target] = round(corr, 4) if not np.isnan(corr) else 0.0
    results[feat] = row

corr_df = pd.DataFrame(results).T  # rows = consumer features, cols = property targets

# Add a summary column: average absolute correlation across all targets
corr_df['AVG_ABS_CORRELATION'] = corr_df.abs().mean(axis=1).round(4)
corr_df = corr_df.sort_values('AVG_ABS_CORRELATION', ascending=False)

# Print full table
print('\n--- Full Correlation Table (sorted by avg strength) ---')
print(corr_df.to_string())

# Print top features per property target
print('\n\n--- Top 8 Consumer Attributes Per Property Characteristic ---')
for target in property_targets:
    print(f'\n  {target}:')
    top = corr_df[target].abs().sort_values(ascending=False).head(8)
    for feat in top.index:
        val = corr_df.loc[feat, target]
        direction = '+' if val > 0 else '-'
        print(f'    {direction} {feat}: {val}')

# Print recommended features to KEEP (avg abs correlation > 0.05)
print('\n\n--- RECOMMENDED: Keep These Consumer Attributes ---')
keep = corr_df[corr_df['AVG_ABS_CORRELATION'] >= 0.05].index.tolist()
for f in keep:
    print(f'  {f}  (avg corr: {corr_df.loc[f, "AVG_ABS_CORRELATION"]})')

# Print recommended features to DROP (avg abs correlation < 0.05)
print('\n--- CONSIDER DROPPING: Low Correlation Attributes ---')
drop = corr_df[corr_df['AVG_ABS_CORRELATION'] < 0.05].index.tolist()
for f in drop:
    print(f'  {f}  (avg corr: {corr_df.loc[f, "AVG_ABS_CORRELATION"]})')


# ============================================================
# STEP 6: Save Report
# ============================================================
corr_df.to_csv('correlation_report.csv')
print('\n\nSaved: correlation_report.csv')
print('Open this in Excel to review all correlations and decide which columns to keep.')
print('\nNext step: tell me which columns you want to keep and we will build the model!')