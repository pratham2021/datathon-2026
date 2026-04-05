import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid, NearestNeighbors
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics.pairwise import cosine_similarity

# Consumer Data (cleaned)
  # HouseholdSize
  # VehicleKnownOwnedNumber
  # NetWorth
  # Gardening
  # HomeImprovement
  # OutdoorsGrouping
  # OwnerRenter

# Property Assessment Data (cleaned)
  # NumberOfBedrooms
  # GarageParkingNumberOfCars
  # TotalAssessedValue
  # LotSizeOrArea
  # YearBuilt
  # SalesPriceFromAssessment

consumer_data_columns = ['HouseholdSize', 'VehicleKnownOwnedNumber', 'NetWorth', 'Gardening', 
                         'HomeImprovement', 'OutdoorsGrouping', 'OwnerRenter']

property_data_columns = ['NumberOfBedrooms', 'GarageParkingNumberOfCars', 'TotalAssessedValue', 
                         'LotSizeOrArea', 'YearBuilt', 'LotSizeOrArea', 'SalesPriceFromAssessment']

consumer_data = pd.read_csv('consumer_clean.csv')
property_data = pd.read_csv('PropertyAssessmentData_cleaned.csv')

consumer = consumer_data[consumer_data_columns].copy()
property = property_data[property_data_columns].copy()

# Rename property columns to match consumer columns
property.columns = consumer_data_columns

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

property_scaled = scaler.fit_transform(property)
consumer_scaled = scaler.fit_transform(consumer)

nn = NearestNeighbors(n_neighbors=3, metric='cosine')
nn.fit(property_scaled)

distances, indices = nn.kneighbors(consumer_scaled)

top_k_houses = property.iloc[indices[0]].copy()
top_k_houses["Similarity"] = 1 - distances[0]
print(top_k_houses)

consumer_data = pd.read_csv("ConsumerData.csv")
property_assessment_data = pd.read_csv("PropertyAssessmentData.csv")