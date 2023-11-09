import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd
import lightgbm as lgb
import sys
import sys

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

from scipy.stats import zscore

import pandas as pd
from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder
# from catboost import CatBoostRegressor

# Variance Threshold:
# from sklearn.feature_selection import VarianceThreshold

# Read the dataset
path = 'Train.csv'
dataset = pd.read_csv(path)
train_labels = dataset['Yield'].copy()
dataset = dataset.drop(columns=['ID','Yield'], axis=1)

test_path = 'Test.csv'
dataset_test = pd.read_csv(test_path)
dataset_upload = dataset_test.copy()
dataset_test = dataset_test.drop(columns=['ID'], axis=1)

# check the unique value of the "TransplantingIrrigationSource" column
print(dataset['TransplantingIrrigationSource'].unique())

print(dataset['TransplantingIrrigationPowerSource'].unique())

print(dataset['CropEstMethod'].unique())
dataset.info()
sys.exit()
# If no OrganicFertilizer, then fill the OrgFertilizers with None, and fill PCropSolidOrgFertAppMethod with None
dataset['Harv_hand_rent'] = dataset['Harv_hand_rent'].fillna(0)
dataset['OrgFertilizers'] = dataset['OrgFertilizers'].fillna('None')
dataset['CropbasalFerts'] = dataset['CropbasalFerts'].fillna('None')
dataset['FirstTopDressFert'] = dataset['FirstTopDressFert'].fillna('None')
dataset['NursDetFactor'] = dataset['NursDetFactor'].fillna('None')
dataset['LandPreparationMethod'] = dataset['LandPreparationMethod'].fillna('None')
dataset['TransDetFactor'] = dataset['TransDetFactor'].fillna(dataset['TransDetFactor'].mode()[0])
dataset['PCropSolidOrgFertAppMethod'] = dataset['PCropSolidOrgFertAppMethod'].fillna('None')

dataset_test['Harv_hand_rent'] = dataset_test['Harv_hand_rent'].fillna(0)
dataset_test['OrgFertilizers'] = dataset_test['OrgFertilizers'].fillna('None')
dataset_test['CropbasalFerts'] = dataset_test['CropbasalFerts'].fillna('None')
dataset_test['FirstTopDressFert'] = dataset_test['FirstTopDressFert'].fillna('None')
dataset_test['NursDetFactor'] = dataset_test['NursDetFactor'].fillna('None')
dataset_test['LandPreparationMethod'] = dataset_test['LandPreparationMethod'].fillna('None')
dataset_test['TransDetFactor'] = dataset_test['TransDetFactor'].fillna(dataset_test['TransDetFactor'].mode()[0])
dataset_test['PCropSolidOrgFertAppMethod'] = dataset_test['PCropSolidOrgFertAppMethod'].fillna('None')


"""
# For each column you want to fill with a distribution of its own values
for column in ['OrgFertilizers', 'CropbasalFerts', 'FirstTopDressFert', 
               'NursDetFactor', 'LandPreparationMethod', 'TransDetFactor', 
               'PCropSolidOrgFertAppMethod']:
    
    # Get the distribution of non-null values
    values_count = dataset[column].value_counts(normalize=True)

    # Generate random values following the distribution
    fill_values = np.random.choice(values_count.index, size=dataset[column].isna().sum(), p=values_count.values)
    
    # Fill the NaNs with the generated values
    dataset.loc[dataset[column].isna(), column] = fill_values

"""

# Checking the unique values for categorical variables to identify sparse classes
categorical_columns = dataset.select_dtypes(include=['object']).columns
sparse_classes = {col: dataset[col].nunique() for col in categorical_columns if dataset[col].nunique() > 30} 

# Define a threshold for grouping
threshold_percentage = 1
threshold = len(dataset) * (threshold_percentage / 100)

# Function to group sparse classes
def group_sparse_classes(df, column, threshold):
    # Find the categories that are below the threshold
    value_counts = df[column].value_counts()
    to_replace = value_counts[value_counts <= threshold].index.tolist()
    
    # Replace the sparse classes with 'other'
    df[column] = df[column].replace(to_replace, 'other')
    return df

# useless_columns = ['TransDetFactor', 'NursDetFactor'], this will be only useful later

# Apply grouping for the identified categorical variables with many unique values
for col in ['LandPreparationMethod', 'OrgFertilizers', 'CropbasalFerts']:
    dataset = group_sparse_classes(dataset, col, threshold)

# Check the new number of unique values for the grouped columns
grouped_classes = {col: dataset[col].nunique() for col in ['LandPreparationMethod', 'OrgFertilizers', 'CropbasalFerts']}

# Adjust the function to use the categories from the training dataset
def adjust_test_categories(train_df, test_df, column):
    # Get the categories in the training data
    train_categories = set(train_df[column].unique())
    # Define a function to apply the adjustment
    def adjust_category(cat):
        return cat if cat in train_categories else 'other'
    # Adjust the test data
    test_df[column] = test_df[column].apply(adjust_category)
    return test_df

# Apply the category adjustment to the test dataset based on the training dataset categories
for col in ['LandPreparationMethod', 'OrgFertilizers', 'CropbasalFerts']:
    test_dataset = adjust_test_categories(dataset, dataset_test, col)

high_skewed_features = ['Residue_perc', 'StandingWater']
# TOOOOOODOOOO, the log2, log10 will be tested 
low_skewed_features = ['Residue_length', 'CropOrgFYM', 'SeedlingsPerPit'] # No missing value here
moderate_skewed_features = ['BasalDAP', 'Acre', 
                            'TransIrriCost', 'Ganaura',
                            'BasalUrea', 'TransplantingIrrigationHours']

### The TransIrriCost should be connected with TransplantingIrrigationSource, if there is no method used, 
### then it should be 0, otherwise maybe try to use the mean value to fill the missing value for each method


dataset['StandingWater'] = dataset['StandingWater'].fillna(0)
dataset_test['StandingWater'] = dataset_test['StandingWater'].fillna(0)

dataset['Ganaura'] = dataset['Ganaura'].fillna(0)
dataset_test['Ganaura'] = dataset_test['Ganaura'].fillna(0)
# Fill the low_skewed_features with median
dataset[low_skewed_features] = dataset[low_skewed_features].fillna(0)
dataset_test[low_skewed_features] = dataset_test[low_skewed_features].fillna(0)

print(dataset.info())
# Fill the moderate_skewed_features with median 
dataset[moderate_skewed_features] = dataset[moderate_skewed_features].fillna(dataset[moderate_skewed_features].median())
dataset_test[moderate_skewed_features] = dataset_test[moderate_skewed_features].fillna(dataset_test[moderate_skewed_features].median()) 

"""
# Outlier remove preprocessing
# Cap at 99th percentile
for feature in moderate_skewed_features:
    percentile_value = dataset[feature].quantile(0.99)
    percentile_value_test = dataset_test[feature].quantile(0.99)
    dataset.loc[dataset[feature] > percentile_value, feature] = percentile_value
    dataset_test.loc[dataset_test[feature] > percentile_value_test, feature] = percentile_value_test

# debug
# print(set(dataset.columns) == set(dataset_test.columns))

for feature in low_skewed_features:
    median_value = dataset[feature].median()
    median_value_test = dataset_test[feature].median()
    z_scores = zscore(dataset[feature].dropna())
    z_scores_test = zscore(dataset_test[feature].dropna())
    abs_z_scores = abs(z_scores)
    abs_z_scores_test = abs(z_scores_test)
    outlier_indices = dataset[feature].dropna().index[abs_z_scores > 3]
    outlier_indices_test = dataset_test[feature].dropna().index[abs_z_scores_test > 3]
    dataset.loc[outlier_indices, feature] = median_value
    dataset_test.loc[outlier_indices_test, feature] = median_value_test
"""
# debug
# print(set(dataset.columns) == set(dataset_test.columns))

# ALERT, not so sure whether the right-skewed numerical features needed to be processed!
# Apply log transformation to right-skewed numerical features
# log_transform_features = ['TransIrriCost', 'StandingWater', 'Ganaura']
# dataset[log_transform_features] = dataset[log_transform_features].apply(np.log1p)
# dataset_test[log_transform_features] = dataset_test[log_transform_features].apply(np.log1p)

# debug
# print(set(dataset.columns) == set(dataset_test.columns))

# check the unique value of the SeedlingsPerPit
# print(dataset['SeedlingsPerPit'].unique())
# Find the best binning strategy for SeedlingsPerPit
# dataset['SeedlingsPerPit'].value_counts().sort_index()
# Binning the SeedlingsPerPit
dataset['SeedlingsPerPit'] = dataset['SeedlingsPerPit'].fillna(0)
dataset_test['SeedlingsPerPit'] = dataset_test['SeedlingsPerPit'].fillna(0)


bins_SeedlingsPerPit = [0, 2, 4, np.inf]
labels_SeedlingsPerPit = ['Low', 'Medium', 'High']
dataset['SeedlingsPerPit_Binned'] = pd.cut(dataset['SeedlingsPerPit'], bins=bins_SeedlingsPerPit, labels=labels_SeedlingsPerPit)
dataset_test['SeedlingsPerPit_Binned'] = pd.cut(dataset_test['SeedlingsPerPit'], bins=bins_SeedlingsPerPit, labels=labels_SeedlingsPerPit)

# debug
# print(set(dataset.columns) == set(dataset_test.columns))

# print(dataset['NoFertilizerAppln'].unique())
# dataset['NoFertilizerAppln'].value_counts().sort_index()


bins_NoFertilizerAppln = [0, 1, 2, np.inf]
labels_NoFertilizerAppln = ['Low', 'Medium', 'High']
dataset['NoFertilizerAppln_Binned'] = pd.cut(dataset['NoFertilizerAppln'], bins=bins_NoFertilizerAppln, labels=labels_NoFertilizerAppln)
dataset_test['NoFertilizerAppln_Binned'] = pd.cut(dataset_test['NoFertilizerAppln'], bins=bins_NoFertilizerAppln, labels=labels_NoFertilizerAppln)


# debug
# print(set(dataset.columns) == set(dataset_test.columns))
# print(dataset['RcNursEstDate'].unique())
# if RcNursEstDate is NaN, then fill NursDetFactor with None
# if RcNursEstDate is not NaN, then fill NursDetFactor with mode
dataset.loc[dataset['RcNursEstDate'].isnull(), 'NursDetFactor'] = 'None'
dataset.loc[dataset['RcNursEstDate'].notnull(), 'NursDetFactor'] = dataset['NursDetFactor'].mode()[0]

dataset_test.loc[dataset_test['RcNursEstDate'].isnull(), 'NursDetFactor'] = 'None'
dataset_test.loc[dataset_test['RcNursEstDate'].notnull(), 'NursDetFactor'] = dataset_test['NursDetFactor'].mode()[0]

# Fill the missing value with MineralFertAppMethod.1 with mode
dataset['MineralFertAppMethod.1'] = dataset['MineralFertAppMethod.1'].fillna(dataset['MineralFertAppMethod.1'].mode()[0])
dataset_test['MineralFertAppMethod.1'] = dataset_test['MineralFertAppMethod.1'].fillna(dataset_test['MineralFertAppMethod.1'].mode()[0])

# Datetime features generator
date_features = ['CropTillageDate', 'RcNursEstDate', 'SeedingSowingTransplanting', 'Harv_date', 'Threshing_date']
# generate new features from the date features
# One feature would be Harvest_date - SeedingSowingTransplanting
# Another feature would be Harvest_date - RcNursEstDate
# Another feature would be Harvest_date - CropTillageDate
# Another feature would be Threshing_date - Harvest_date
# We start to generate those features

# Harvest_date - SeedingSowingTransplanting
# 'Harv_date', 'SeedingSowingTransplanting', 'Harv_date_SeedingSowingTransplanting' and 'RcNursEstDate' are all datetime features
# Fill the 'Harv_date', 'SeedingSowingTransplanting', 'Harv_date_SeedingSowingTransplanting' and 'RcNursEstDate' with median
# Convert the datetime to numeric form (Unix timestamp)
# Forward-fill missing datetime values
dataset['Harv_date'] = pd.to_datetime(dataset['Harv_date']).ffill()
dataset['SeedingSowingTransplanting'] = pd.to_datetime(dataset['SeedingSowingTransplanting']).ffill()
dataset['RcNursEstDate'] = pd.to_datetime(dataset['RcNursEstDate']).ffill()
dataset['Threshing_date'] = pd.to_datetime(dataset['Threshing_date']).ffill()
dataset['CropTillageDate'] = pd.to_datetime(dataset['CropTillageDate']).ffill()

# If you need to forward-fill on the test dataset, you do it similarly:
dataset_test['Harv_date'] = pd.to_datetime(dataset_test['Harv_date']).ffill()
dataset_test['SeedingSowingTransplanting'] = pd.to_datetime(dataset_test['SeedingSowingTransplanting']).ffill()
dataset_test['RcNursEstDate'] = pd.to_datetime(dataset_test['RcNursEstDate']).ffill()
dataset_test['Threshing_date'] = pd.to_datetime(dataset_test['Threshing_date']).ffill()
dataset_test['CropTillageDate'] = pd.to_datetime(dataset_test['CropTillageDate']).ffill()

dataset['Harv_date'] = pd.to_datetime(dataset['Harv_date'], errors='coerce')
dataset['SeedingSowingTransplanting'] = pd.to_datetime(dataset['SeedingSowingTransplanting'], errors='coerce')
dataset['Harv_date_SeedingSowingTransplanting'] = (dataset['Harv_date'] - dataset['SeedingSowingTransplanting']).dt.days
dataset_test['Harv_date'] = pd.to_datetime(dataset_test['Harv_date'], errors='coerce')
dataset_test['SeedingSowingTransplanting'] = pd.to_datetime(dataset_test['SeedingSowingTransplanting'], errors='coerce')
dataset_test['Harv_date_SeedingSowingTransplanting'] = (dataset_test['Harv_date'] - dataset_test['SeedingSowingTransplanting']).dt.days

# Harvest_date - RcNursEstDate
dataset['RcNursEstDate'] = pd.to_datetime(dataset['RcNursEstDate'], errors='coerce')
dataset_test['RcNursEstDate'] = pd.to_datetime(dataset_test['RcNursEstDate'], errors='coerce')
# print(dataset['RcNursEstDate'].unique())
# print(dataset['RcNursEstDate'].dtype)
# If the Harv_date_RcNursEstDate is smaller than 0, add 365 to it
dataset['Harv_date_RcNursEstDate'] = (dataset['Harv_date'] - dataset['RcNursEstDate']).dt.days
dataset_test['Harv_date_RcNursEstDate'] = (dataset_test['Harv_date'] - dataset_test['RcNursEstDate']).dt.days
dataset.loc[dataset['Harv_date_RcNursEstDate'] < 0, 'Harv_date_RcNursEstDate'] = dataset.loc[dataset['Harv_date_RcNursEstDate'] < 0, 'Harv_date_RcNursEstDate'] + 365
dataset_test.loc[dataset_test['Harv_date_RcNursEstDate'] < 0, 'Harv_date_RcNursEstDate'] = dataset_test.loc[dataset_test['Harv_date_RcNursEstDate'] < 0, 'Harv_date_RcNursEstDate'] + 365

# Harvest_date - CropTillageDate
dataset['CropTillageDate'] = pd.to_datetime(dataset['CropTillageDate'], errors='coerce')
dataset_test['CropTillageDate'] = pd.to_datetime(dataset_test['CropTillageDate'], errors='coerce')
dataset['Harv_date_CropTillageDate'] = (dataset['Harv_date'] - dataset['CropTillageDate']).dt.days
dataset_test['Harv_date_CropTillageDate'] = (dataset_test['Harv_date'] - dataset_test['CropTillageDate']).dt.days

# Threshing_date - Harvest_date
dataset['Threshing_date'] = pd.to_datetime(dataset['Threshing_date'], errors='coerce')
dataset['Threshing_date_Harv_date'] = (dataset['Threshing_date'] - dataset['Harv_date']).dt.days
dataset_test['Threshing_date'] = pd.to_datetime(dataset_test['Threshing_date'], errors='coerce')
dataset_test['Threshing_date_Harv_date'] = (dataset_test['Threshing_date'] - dataset_test['Harv_date']).dt.days


# Generate the month features, this will be transfered into one-hot encoding later
dataset['Harv_date_Month'] = dataset['Harv_date'].dt.month
dataset_test['Harv_date_Month'] = dataset_test['Harv_date'].dt.month

# Check the unique values of the categorical features
# print(dataset['Harv_date_Month'].unique())
# print(dataset_test['Harv_date_Month'].unique())

# Transfer the Month values into seasonn feature
# 1, 2, 3 -> 1
# 4, 5, 6 -> 2
# 7, 8, 9 -> 3
# 10, 11, 12 -> 4

# Create a dictionary to map the month values to the season values
month_to_season = {1: 1, 2: 1, 3: 1, 4: 2, 5: 2,
                     6: 2, 7: 3, 8: 3, 9: 3, 10: 4,
                     11: 4, 12: 4}

# Map the month values to the season values
dataset['Harv_date_Season'] = dataset['Harv_date_Month'].map(month_to_season)
dataset_test['Harv_date_Season'] = dataset_test['Harv_date_Month'].map(month_to_season)

# debug
"""
print(set(dataset.columns) == set(dataset_test.columns))

print(dataset['Harv_date_SeedingSowingTransplanting'].isnull().sum())
print(dataset['Harv_date_RcNursEstDate'].isnull().sum())
print(dataset['Harv_date_CropTillageDate'].isnull().sum())
print(dataset['Threshing_date_Harv_date'].isnull().sum())

"""

# Drop the original date features
dataset = dataset.drop(columns=date_features)
dataset_test = dataset_test.drop(columns=date_features)

# debug
# print(set(dataset.columns) == set(dataset_test.columns))

# Fill missing values in 1tdUrea and 2tdUrea with 0, as they will be summed
dataset['BasalUrea'] = dataset['BasalUrea'].fillna(0)
dataset['1tdUrea'] = dataset['1tdUrea'].fillna(0)
dataset['2tdUrea'] = dataset['2tdUrea'].fillna(0)
dataset['BasalDAP'] = dataset['BasalDAP'].fillna(0)
dataset_test['BasalUrea'] = dataset_test['BasalUrea'].fillna(0)
dataset_test['1tdUrea'] = dataset_test['1tdUrea'].fillna(0)
dataset_test['2tdUrea'] = dataset_test['2tdUrea'].fillna(0)

dataset_test['BasalDAP'] = dataset_test['BasalDAP'].fillna(0)

# Fill the Ganaura and CropOrgFYM with 0
dataset['Ganaura'] = dataset['Ganaura'].fillna(0)
dataset['CropOrgFYM'] = dataset['CropOrgFYM'].fillna(0)
dataset_test['Ganaura'] = dataset_test['Ganaura'].fillna(0)
dataset_test['CropOrgFYM'] = dataset_test['CropOrgFYM'].fillna(0)

""""
dataset['TotalOrganicFertilizer'] = dataset['Ganaura'] + dataset['CropOrgFYM']
dataset['TotalChemicalFertilizer'] = dataset['BasalDAP'] + dataset['BasalUrea'] + dataset['1tdUrea'] + dataset['2tdUrea']
dataset['TotalFertilizerUsed'] = dataset['TotalOrganicFertilizer'] + dataset['TotalChemicalFertilizer']
dataset_test['TotalOrganicFertilizer'] = dataset_test['Ganaura'] + dataset_test['CropOrgFYM']
dataset_test['TotalChemicalFertilizer'] = dataset_test['BasalDAP'] + dataset_test['BasalUrea'] + dataset_test['1tdUrea'] + dataset_test['2tdUrea']
dataset_test['TotalFertilizerUsed'] = dataset_test['TotalOrganicFertilizer'] + dataset_test['TotalChemicalFertilizer']

# Create the ratio feature again
dataset['Organic_Chemical_Fertilizer_Ratio'] = dataset['TotalOrganicFertilizer'] / dataset['TotalChemicalFertilizer']
dataset['Organic_Total_Fertilizer_Ratio'] = dataset['TotalOrganicFertilizer'] / dataset['TotalFertilizerUsed']
dataset_test['Organic_Chemical_Fertilizer_Ratio'] = dataset_test['TotalOrganicFertilizer'] / dataset_test['TotalChemicalFertilizer']
dataset_test['Organic_Total_Fertilizer_Ratio'] = dataset_test['TotalOrganicFertilizer'] / dataset_test['TotalFertilizerUsed']
dataset['Organic_Chemical_Fertilizer_Ratio'].replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace inf with NaN for division by zero cases, actually no in our case.
dataset_test['Organic_Chemical_Fertilizer_Ratio'].replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace inf with NaN for division by zero cases, actually no in our case.
"""
"""
print(dataset['Organic_Chemical_Fertilizer_Ratio'].isnull().sum(), "NaNs in Organic_Chemical_Fertilizer_Ratio")
print(dataset_test['Organic_Chemical_Fertilizer_Ratio'].isnull().sum(), "NaNs in Organic_Chemical_Fertilizer_Ratio")
sys.exit()
"""

"""
# Create Polynomial Features: Squared terms for significant numerical features like CultLand and CropCultLand again
dataset['CultLand_Squared'] = dataset['CultLand'] ** 2
dataset['CropCultLand_Squared'] = dataset['CropCultLand'] ** 2
dataset_test['CultLand_Squared'] = dataset_test['CultLand'] ** 2
dataset_test['CropCultLand_Squared'] = dataset_test['CropCultLand'] ** 2
"""

"""
# Create new columns TotalUrea and TotalAppDaysUrea by summing 1tdUrea and 2tdUrea, and 1appDaysUrea and 2appDaysUrea
dataset['TotalUrea'] = dataset['1tdUrea'] + dataset['2tdUrea'] + dataset['BasalUrea']
dataset_test['TotalUrea'] = dataset_test['1tdUrea'] + dataset_test['2tdUrea'] + dataset_test['BasalUrea']
"""

# Do the same for 1appDaysUrea and 2appDaysUrea
dataset['1appDaysUrea'] = dataset['1appDaysUrea'].fillna(dataset['1appDaysUrea'].median())
dataset['2appDaysUrea'] = dataset['2appDaysUrea'].fillna(dataset['2appDaysUrea'].median())
dataset_test['1appDaysUrea'] = dataset_test['1appDaysUrea'].fillna(dataset_test['1appDaysUrea'].median())
dataset_test['2appDaysUrea'] = dataset_test['2appDaysUrea'].fillna(dataset_test['2appDaysUrea'].median())

"""
dataset['TotalAppDaysUrea'] = dataset['1appDaysUrea'] + dataset['2appDaysUrea']
dataset_test['TotalAppDaysUrea'] = dataset_test['1appDaysUrea'] + dataset_test['2appDaysUrea']

"""
dataset['Org_Crop'] = dataset['OrgFertilizers'].astype(str) + " + " + dataset['CropbasalFerts'].astype(str)
dataset_test['Org_Crop'] = dataset_test['OrgFertilizers'].astype(str) + "+" + dataset_test['CropbasalFerts'].astype(str)
dataset['Crop_FirstTop'] = dataset['CropbasalFerts'].astype(str) + "+" + dataset['FirstTopDressFert'].astype(str)
dataset_test['Crop_FirstTop'] = dataset_test['CropbasalFerts'].astype(str) + "+" + dataset_test['FirstTopDressFert'].astype(str)
dataset['Org_FirstTop'] = dataset['OrgFertilizers'].astype(str) + "+" + dataset['FirstTopDressFert'].astype(str)
dataset_test['Org_FirstTop'] = dataset_test['OrgFertilizers'].astype(str) + "+" + dataset_test['FirstTopDressFert'].astype(str)
"""
"""
dataset['PCropSolidOrgFertAppMethod'] = dataset['PCropSolidOrgFertAppMethod'].fillna('other')
dataset_test['PCropSolidOrgFertAppMethod'] = dataset_test['PCropSolidOrgFertAppMethod'].fillna('other')

# If CropEstMethod is None, then fill the TransIrriCost with 0
# if CropEstMethod is not none, then fill the StandingWater with median
dataset['TransIrriCost'] = dataset['TransIrriCost'].fillna(dataset['TransIrriCost'].median()) 
dataset_test['TransIrriCost'] = dataset_test['TransIrriCost'].fillna(dataset_test['TransIrriCost'].median())

"""
dataset['TransplantingIrrigationPowerSource'] = dataset['TransplantingIrrigationPowerSource'].fillna('other')
dataset_test['TransplantingIrrigationPowerSource'] = dataset_test['TransplantingIrrigationPowerSource'].fillna('other')

dataset['SeedlingsPerPit'] = dataset['SeedlingsPerPit'].fillna(0)    
dataset_test['SeedlingsPerPit'] = dataset_test['SeedlingsPerPit'].fillna(0)

print(dataset['TransIrriCost'].unique())

print(dataset.shape, dataset_test.shape)

"""
# print(dataset.columns)
# Fill the missing values in those columns with 0 and mode
dataset['TransplantingIrrigationHours'] = dataset['TransplantingIrrigationHours'].fillna(0)
dataset_test['TransplantingIrrigationHours'] = dataset_test['TransplantingIrrigationHours'].fillna(0)
dataset['TransplantingIrrigationSource'] = dataset['TransplantingIrrigationSource'].fillna(dataset['TransplantingIrrigationSource'].mode()[0])
dataset_test['TransplantingIrrigationSource'] = dataset_test['TransplantingIrrigationSource'].fillna(dataset_test['TransplantingIrrigationSource'].mode()[0])
dataset['TransplantingIrrigationPowerSource'] = dataset['TransplantingIrrigationPowerSource'].fillna(dataset['TransplantingIrrigationPowerSource'].mode()[0])
dataset_test['TransplantingIrrigationPowerSource'] = dataset_test['TransplantingIrrigationPowerSource'].fillna(dataset_test['TransplantingIrrigationPowerSource'].mode()[0])

# check the unique value of the SeedlingsPerPit column，fill it with 0
# fill harv_hand_rent with 0

# fill seeHarv_date_RcNursEstDate
#print(dataset['Harv_date_RcNursEstDate'].unique())

# print(dataset.info())
# check which columns have missing values
# print(dataset.isnull().sum())
# check the unique value of every columns  in the dataset
# print(dataset.nunique())

# The following columns have too many unique values right now, we will cut them into bins
"""['CultLand', 'CropCultLand', 'TransIrriCost', 'Harv_hand_rent', 
 'Harv_date_SeedingSowingTransplanting', 'Harv_date_RcNursEstDate', 'Harv_date_CropTillageDate', 'Threshing_date_Harv_date',
 'TotalChemicalFertilizer', 'TotalFertilizerUsed', 'Organic_Chemical_Fertilizer_Ratio', 'Organic_Total_Fertilizer_Ratio']"""
# Drop the columns used to generate the new features, not the new features themselves
"""
dataset = dataset.drop(columns = ['BasalUrea', '1tdUrea', '2tdUrea', 'BasalDAP', 
                                  'Ganaura', 'CropOrgFYM',  '1appDaysUrea', '2appDaysUrea', 'Harv_date_Month', 'District', 'TransDetFactor'], axis=1)
dataset_test = dataset_test.drop(columns = ['BasalUrea', '1tdUrea', '2tdUrea', 'BasalDAP',
                                            'Ganaura', 'CropOrgFYM',  '1appDaysUrea', '2appDaysUrea', 'Harv_date_Month', 'District', 'TransDetFactor'], axis=1)

"""
dataset = dataset.drop(columns=['Harv_date_Month'], axis=1)
dataset_test = dataset_test.drop(columns=['Harv_date_Month'], axis=1)


for col in dataset.columns:
    if dataset[col].dtype == 'object':
        dataset[col] = dataset[col].astype('category').cat.codes
    elif dataset[col].dtype == 'datetime64[ns]':
        dataset = dataset.drop(columns=[col])

for col in dataset_test.columns:
    if dataset_test[col].dtype == 'object':
        dataset_test[col] = dataset_test[col].astype('category').cat.codes
    elif dataset_test[col].dtype == 'datetime64[ns]':
        dataset_test = dataset_test.drop(columns=[col])

# A simple test with the lightgbm
# Prepare the data
X = dataset

y = train_labels    
# print(X.shape, y.shape)

# Train Test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=300)

# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Select the object columns and categorical columns
object_cols = X.select_dtypes(include=['object'])
categorical_cols = X.select_dtypes(include=['category'])
whole_categorical_cols = pd.concat([object_cols, categorical_cols], axis=1) 
# generate a list with the name of the categorical columns
whole_categorical_cols = whole_categorical_cols.columns
# Transfer into a list
whole_categorical_cols = list(whole_categorical_cols)

# make sure the ID column has been dropped from the dataset

# datetime_columns = dataset.select_dtypes(include=[np.datetime64]).columns.tolist()
# datetime_columns_test = dataset_test.select_dtypes(include=[np.datetime64]).columns.tolist()

# debug
# print(set(dataset.columns) == set(dataset_test.columns))
# print(dataset.shape, dataset_test.shape, "\n This is the shape after feature generation")

"""
from sklearn.ensemble import StackingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

lgbm_model = lgb.LGBMRegressor(objective='regression', num_leaves=31, learning_rate=0.01, n_estimators=200)

# Add other base models here
base_estimators = [
    ('lgbm', lgbm_model),
    # Add here more base models
]

stacking_model = StackingRegressor(estimators=base_estimators, final_estimator=lgb.LGBMRegressor())

# Train the stacking model
stacking_model.fit(X_train, y_train)

# Make predictions
y_pred = stacking_model.predict(X_test)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(rmse)

# Test with dataset_test
y_pred_test = stacking_model.predict(dataset_test)

# Create submission DataFrame
submission_df = pd.DataFrame({'ID': dataset_upload['ID'], 'Yield': y_pred_test})
submission_df.to_csv('submission_07_11.csv', index=False)

sys.exit()

"""
"""
# Create a LightGBM model
lgbm_model = lgb.LGBMRegressor(objective = 'regression', num_leaves = 31, learning_rate = 0.01, n_estimators = 120)


estimators = [('lgbm', lgbm_model)]
ensemble = VotingRegressor(estimators, weights=[1]) 
ensemble.fit(X_train, y_train)
y_pred = ensemble.predict(X_test)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(rmse)

# Test with dataset_test
y_pred_test = ensemble.predict(dataset_test)

# Create a submission file
dataset_upload = dataset_test_original

# Create a submission DataFrame
submission_df = pd.DataFrame({'ID': dataset_upload['ID'], 'Yield': y_pred_test})
submission_df.to_csv('submission_07_11_3.csv', index=False)
"""

###-----------###   Training beta test  ###-----------###
import lightgbm as lgb
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

print(dataset.columns, dataset_test.columns)

# Parameters
n_splits = 20
kf = KFold(n_splits=n_splits, shuffle=True, random_state=1024)

# Prepare an array to store the RMSE for each fold
rmse_scores = []
models = []

# Initialize an empty array to hold feature importances
feature_importances = np.zeros(X.shape[1])

# Start the K-Fold cross-validation loop
for fold, (train_index, val_index) in enumerate(kf.split(X)):
    # Split the data
    X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
    y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]

    # Create a LGBMRegressor object
    lgbm_model = lgb.LGBMRegressor(objective='regression', num_leaves=15, learning_rate=0.05, n_estimators=38)
    
    # Train the model
    lgbm_model.fit(
        X_train_fold, y_train_fold,  
        eval_set=[(X_val_fold, y_val_fold)],
        eval_metric='mae', 
        categorical_feature=whole_categorical_cols
    )
    
    # Predict on the validation set
    y_pred_val = lgbm_model.predict(X_val_fold, num_iteration=lgbm_model.best_iteration_)

    # Calculate and print RMSE for the current fold
    fold_rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred_val))
    rmse_scores.append(fold_rmse)
    print(f"Fold {fold}: RMSE: {fold_rmse}")

    # Accumulate feature importances
    feature_importances += lgbm_model.feature_importances_

    models.append(lgbm_model)

# After cross-validation, print the mean RMSE
print(f"Mean RMSE: {np.mean(rmse_scores)}, STD RMSE: {np.std(rmse_scores)}")

# Feature importances from all folds
feature_importances = feature_importances / n_splits

test_predictions = []

for model in models:
    # Make predictions
    test_df = dataset_test[X.columns]
    # print(test_df.shape)
    fold_preds = model.predict(test_df, num_iteration=model.best_iteration_)
    test_predictions.append(fold_preds)

# Now average these predictions
test_predictions = np.column_stack(test_predictions)
y_pred_test = np.mean(test_predictions, axis=1)

# Now you have `y_pred_test` which is the averaged predictions from all folds

# Create a submission file
# Make predictions on the Zindi test set
# test_df = dataset_test[X.columns]
# y_pred_test = lgbm_model.predict(test_df, num_iteration=lgbm_model.best_iteration_)

submission_df = pd.DataFrame({'ID': dataset_upload['ID'], 'Yield': y_pred_test})
submission_df.to_csv('10_mae_real.csv', index=False)

sys.exit()

lgbm_model = lgb.LGBMRegressor(objective='regression', num_leaves=31, learning_rate=0.01, n_estimators=100)

# Train the model#########
lgbm_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='mae',
                categorical_feature=whole_categorical_cols)

#########
# print(lgbm_model.best_iteration_)
# Make predictions
y_pred = lgbm_model.predict(X_test)
#y_pred = lgbm_model.predict(X_test, num_iteration=lgbm_model.best_iteration_)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(rmse)

# Test with dataset_test
y_pred_test = lgbm_model.predict(dataset_test, num_iteration=lgbm_model.best_iteration_)

# Create a submission file
dataset_upload = pd.read_csv(r'C:\Users\ra78lof\Not-so-auto-ml\Test.csv')
#print(dataset_upload.shape)
# Create a submission file
submission_df = pd.DataFrame({'ID': dataset_upload['ID'], 'Yield': y_pred_test})
#print(submission_df.head())
#print(submission_df.shape)
submission_df.to_csv('submission_07_11.csv', index=False)

sys.exit()
###-----------###   Training beta test  ###-----------###

# debug
# print(set(dataset.columns) == set(dataset_test.columns))
# Perform one-hot encoding
# Step 1: Combine the datasets
combined = pd.concat([dataset, dataset_test], keys=['train', 'test'])

# Step 2: Perform one-hot encoding
combined = pd.get_dummies(combined, columns=['District', 'Block', 'LandPreparationMethod'], drop_first=True)

# Step 3: Split the datasets back apart
dataset, dataset_test = combined.xs('train'), combined.xs('test')

# debug
print(set(dataset.columns) == set(dataset_test.columns))
print(dataset.shape, dataset_test.shape)

# Apply label encoding to ordinal categorical features
label_encode_features = ['Harv_method', 'Threshing_method']
label_encoder = LabelEncoder()
for feature in label_encode_features:
    dataset.loc[:, feature] = dataset[feature].astype(str)
    dataset.loc[:, feature] = label_encoder.fit_transform(dataset[feature])
    dataset_test.loc[:, feature] = dataset_test[feature].astype(str)
    dataset_test.loc[:, feature] = label_encoder.fit_transform(dataset_test[feature])

# debug
print(set(dataset.columns) == set(dataset_test.columns))

# Feature Cross
dataset_original['District_Block'] = dataset_original['District'].astype(str) + "_" + dataset_original['Block'].astype(str)
dataset_test_original['District_Block'] = dataset_test_original['District'].astype(str) + "_" + dataset_test_original['Block'].astype(str)

# debug
print(set(dataset.columns) == set(dataset_test.columns))

# K-means Clustering
kmeans = KMeans(n_clusters=3, random_state=0)
subset_for_clustering = dataset_original[['CultLand', 'CropCultLand']].dropna()
subset_for_clustering_test = dataset_test_original[['CultLand', 'CropCultLand']].dropna()
kmeans_labels = kmeans.fit_predict(subset_for_clustering)
kmeans_labels_test = kmeans.predict(subset_for_clustering_test)
dataset_original.loc[subset_for_clustering.index, 'Land_Cluster'] = kmeans_labels
dataset_test_original.loc[subset_for_clustering_test.index, 'Land_Cluster'] = kmeans_labels_test

# debug
print(set(dataset.columns) == set(dataset_test.columns))

# PCA 
numerical_features_original = dataset_original.select_dtypes(include=['float64', 'int64']).columns
numerical_features_original_test = dataset_test_original.select_dtypes(include=['float64', 'int64']).columns
pca = PCA(n_components=2)  # ALERT! Needed to be adjusted in the final phase
principal_components = pca.fit_transform(dataset_original[numerical_features_original].fillna(0))
principal_components_test = pca.transform(dataset_test_original[numerical_features_original_test].fillna(0))
dataset_original['PCA1'], dataset_original['PCA2'] = principal_components[:, 0], principal_components[:, 1]
dataset_test_original['PCA1'], dataset_test_original['PCA2'] = principal_components_test[:, 0], principal_components_test[:, 1]

# debug
print(set(dataset.columns) == set(dataset_test.columns))
print(dataset.shape, dataset_test.shape)

# Add to the dataset, ALERT, not the dataset_original
dataset.loc[:, 'District_Block'] = dataset_original['District_Block']
dataset_test.loc[:, 'District_Block'] = dataset_test_original['District_Block']
dataset.loc[:, 'Land_Cluster'] = dataset_original['Land_Cluster']
dataset_test.loc[:, 'Land_Cluster'] = dataset_test_original['Land_Cluster']
dataset.loc[:, 'PCA1'] = dataset_original['PCA1']
dataset.loc[:, 'PCA2'] = dataset_original['PCA2']
dataset_test.loc[:, 'PCA1'] = dataset_test_original['PCA1']
dataset_test.loc[:, 'PCA2'] = dataset_test_original['PCA2']

# debug
print(set(dataset.columns) == set(dataset_test.columns))
print(dataset.shape, dataset_test.shape)

# ALERT, this code snippt should not be used during the xgboost training!
one_hot_columns = [col for col in dataset.columns if 'District_' in col or 'Block_' in col]
one_hot_columns_test = [col for col in dataset_test.columns if 'District_' in col or 'Block_' in col]
dataset = dataset.drop(columns=one_hot_columns)
dataset_test = dataset_test.drop(columns=one_hot_columns_test)
# ALERT, NOT for xgboost
dataset['District'] = dataset_original['District']
dataset['Block'] = dataset_original['Block']
dataset_test['District'] = dataset_test_original['District']
dataset_test['Block'] = dataset_test_original['Block']

# debug
print(set(dataset.columns) == set(dataset_test.columns))

"""
Got a bug , here is the error message:
ValueError: pandas dtypes must be int, float or bool.
Fields with bad pandas dtypes: ID: object, CropTillageDate: datetime64[ns], CropEstMethod: object, RcNursEstDate: datetime64[ns], 
SeedingSowingTransplanting: datetime64[ns], NursDetFactor: object, TransDetFactor: object, TransplantingIrrigationSource: object, 
TransplantingIrrigationPowerSource: object, OrgFertilizers: object, PCropSolidOrgFertAppMethod: object, CropbasalFerts: object, 
MineralFertAppMethod: object, FirstTopDressFert: object, MineralFertAppMethod.1: object, Harv_date: datetime64[ns], Threshing_date: datetime64[ns], Stubble_use: object
"""

datetime_columns = dataset.select_dtypes(include=[np.datetime64]).columns.tolist()
datetime_columns_test = dataset_test.select_dtypes(include=[np.datetime64]).columns.tolist()
for col in dataset.columns:
    if dataset[col].dtype == 'object':
        dataset[col] = dataset[col].astype('category').cat.codes
    elif dataset[col].dtype == 'datetime64[ns]':
        dataset = dataset.drop(columns=[col])

for col in dataset_test.columns:
    if dataset_test[col].dtype == 'object':
        dataset_test[col] = dataset_test[col].astype('category').cat.codes
    elif dataset_test[col].dtype == 'datetime64[ns]':
        dataset_test = dataset_test.drop(columns=[col])

# debug
print(set(dataset.columns) == set(dataset_test.columns))
print(dataset.shape, dataset_test.shape, "\n This is the shape after feature generation")

# Re-attach the 'Yield' column to the training dataset
dataset['Yield'] = train_labels
dataset['Yield'] = dataset['Yield'].fillna(dataset['Yield'].median()) 

# debug
print(set(dataset.columns) == set(dataset_test.columns))
print("Columns in dataset but not in dataset_test: ", set(dataset.columns) - set(dataset_test.columns))

# Get the list of numerical column names
# numerical_cols_names = dataset.select_dtypes(include=['number']).columns.tolist()

# Display the data types and non-null counts of each column in your dataset
dataset.info()

# Assuming dataset and dataset_test are your data frames
print(dataset.select_dtypes(include=['object']).shape[1], "Before handling categorical features")
# How many numerical features are there now
print(dataset.select_dtypes(include=['number']).shape[1], "Before handling numerical features")

bool_cols = dataset.select_dtypes(include=['bool']).columns
dataset[bool_cols] = dataset[bool_cols].astype(int)
dataset_test[bool_cols] = dataset_test[bool_cols].astype(int)

# Identify numerical and categorical columns
numerical_cols = dataset.select_dtypes(include=['number']).columns
numerical_cols.drop('Yield')  # Remove the target variable itself
categorical_cols = dataset.select_dtypes(include=['object', 'category']).columns

# Fill missing values for numerical columns with median
# Fill missing values for numerical columns with the median
for col in numerical_cols:
    median_value = dataset[col].median()
    dataset[col] = dataset[col].fillna(median_value)
    if col in dataset_test.columns:
        median_value_test = dataset_test[col].median()
        dataset_test[col] = dataset_test[col].fillna(median_value_test)  # Apply the same median value to the test set

# Fill missing values for categorical columns with mode
# Fill missing values for categorical columns with the mode
for col in categorical_cols:
    mode_value = dataset[col].mode()[0]
    dataset[col] = dataset[col].fillna(mode_value)
    if col in dataset_test.columns:
        mode_value_test = dataset_test[col].mode()[0]
        dataset_test[col] = dataset_test[col].fillna(mode_value_test)  # Apply the same mode value to the test set

# Get the list of numerical column names
numerical_cols_names = dataset.select_dtypes(include=['number']).columns.tolist()

# Correlation Analysis for numerical columns
correlation_matrix = dataset[numerical_cols_names].corr()
correlated_features = correlation_matrix["Yield"].apply(lambda x: abs(x) >= 0.5).index.tolist()
correlated_features.remove('Yield')  # Remove the target variable itself

# Handling categorical columns
categorical_cols = dataset.select_dtypes(include=['category']).columns.tolist()

# Label encode your categorical columns for both training and testing datasets
le = LabelEncoder()
for col in categorical_cols:
    dataset[col] = le.fit_transform(dataset[col])
    dataset_test[col] = le.transform(dataset_test[col])  # Use transform to ensure consistent encoding

# Now perform Chi-Squared Test for categorical columns
chi2_values, p_values = chi2(dataset[categorical_cols], dataset['Yield'])

# Selecting significant categorical features (let's use p-value < 0.05 as a threshold)
significant_categorical_features = [col for col, p_value in zip(categorical_cols, p_values) if p_value < 0.05]

# Now filter your datasets using both correlated numerical and significant categorical features
selected_features = correlated_features + significant_categorical_features + ['Yield']
dataset = dataset[selected_features]
dataset_test = dataset_test[selected_features[:-1]]  # Exclude 'Yield' from dataset_test as it's not present there

print(dataset.shape, dataset_test.shape)
# How many categorical features are there now
print(dataset.select_dtypes(include=['category']).shape[1], "After handling categorical features")
# How many numerical features are there now
print(dataset.select_dtypes(include=['number']).shape[1],  "After handling numerical features")


# A simple test with the lightgbm
# Prepare the data
X = dataset.drop(columns=['Yield']) 
y = dataset['Yield']

# Test if x still has NaNs
print(X.isnull().sum().sum(), "Before handling missing values")

# Basic Handing, probably won't be the best solution
# This should only be used for the numerical features
# Select only numerical columns
numerical_cols = X.select_dtypes(include=['number'])

# Fill missing values in numerical columns with the median of each column
filled_numerical_cols = numerical_cols.apply(lambda x: x.fillna(x.median()), axis=0)

# Update the original DataFrame X with the filled numerical columns
X.loc[:, numerical_cols.columns] = filled_numerical_cols

# Test if X still has NaNs
print(X.isnull().sum().sum(), "After handling missing values")

# Select only categorical columns
categorical_cols = X.select_dtypes(include=['category'])

# Fill missing values in categorical columns with the mode of each column
filled_categorical_cols = categorical_cols.apply(lambda x: x.fillna(x.mode()[0]), axis=0)

# Update the original DataFrame X with the filled categorical columns
X.loc[:, categorical_cols.columns] = filled_categorical_cols

# Test if X still has NaNs
print(X.isnull().sum().sum(), "After handling missing values for categorical features")

# Figure out which column has a dtype of 'object'
# print(X.dtypes[X.dtypes == 'object'])

# Convert Object dtype to category dtype
# categorical_cols = ['District', 'Block', 'SeedlingsPerPit_Binned', 'NoFertilizerAppln_Binned']
# X[categorical_cols] = X[categorical_cols].astype('category')

# Train Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=9)
# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# print(dataset_test.shape)
print(X_train.columns)
categorical_feature_indices = [X_train.columns.get_loc(col) for col in categorical_cols.columns]


# Create a LightGBM model
lgbm_model = lgb.LGBMRegressor(objective='regression', num_leaves=31, learning_rate=0.05, n_estimators=100)
# lgbm_model = lgb.LGBMRegressor(objective='regression', num_leaves=31, learning_rate=0.01, n_estimators=100, feature_fraction=0.9, bagging_freq = 10, bagging_fraction = 0.9)

# Train the model
lgbm_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='rmse',
                categorical_feature=categorical_feature_indices)
print(lgbm_model.best_iteration_)
# Make predictions
y_pred = lgbm_model.predict(X_test)
#y_pred = lgbm_model.predict(X_test, num_iteration=lgbm_model.best_iteration_)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(rmse)

# Test with dataset_test
y_pred_test = lgbm_model.predict(dataset_test, num_iteration=lgbm_model.best_iteration_)

# Create a submission file
dataset_upload = pd.read_csv(r'C:\Users\ra78lof\Not-so-auto-ml\Test.csv')
print(dataset_upload.shape)
# Create a submission file
submission_df = pd.DataFrame({'ID': dataset_upload['ID'], 'Yield': y_pred_test})
print(submission_df.head())
print(submission_df.shape)
submission_df.to_csv('submission.csv', index=False)

# Use XGBoost

# Create a XGBoost model
import xgboost as xgb
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=0.01,
                max_depth=50, alpha=10, n_estimators=100, enable_categorical=True)

# Train the model
xgb_model.fit(X_train, y_train)

# Make predictions
y_pred = xgb_model.predict(X_test)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(rmse)

# Test with dataset_test
y_pred_test = xgb_model.predict(dataset_test)

# Create a submission file
dataset_upload = pd.read_csv(r'C:\Users\ra78lof\Not-so-auto-ml\Test.csv')
print(dataset_upload.shape)
# Create a submission file
submission_df = pd.DataFrame({'ID': dataset_upload['ID'], 'Yield': y_pred_test})
print(submission_df.head())
print(submission_df.shape)
submission_df.to_csv('submission_cat.csv', index=False)
