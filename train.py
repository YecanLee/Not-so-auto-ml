import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder


# Load the dataset
file_path = r'C:\Users\ra78lof\Not-so-auto-ml\Train.csv'
df = pd.read_csv(file_path)

# Display the first few rows of the dataset
df.head()

# Get summary statistics and data types
summary = df.describe(include='all')
data_types = df.dtypes

print(summary, data_types)

# Check for missing values
df.isnull().sum()

# Those columns with null values are datetime columns
# RcNursEstDate: 83 missing values

# Those columns with null values are categorical columns
# NursDetFactor: 289 missing values, TransDetFactor: 289 missing values, TransplantingIrrigationSource: 115 missing values, 
# OrgFertilizers: 1335 missing values, PCropSolidOrgFertAppMethod: 1337 missing values, CropbasalFerts: 188 missing values, 
# FirstTopDressFert: 485 missing values, MineralFertAppMethod.1: 481 missing values, 

# Those columns with no null values are numerical columns
# SeedlingsPerPit: 289 missing values, TransplantingIrrigationHours: 193 missing values, TransIrriCost: 882 missing values, TransplantingIrrigationPowerSource: 503 missing values
# StandingWater: 238 missing values, Ganaura: 2417 missing values, CropOrgFYM: 2674 missing values, BasalDAP: 543 missing values, BasalUrea: 1704 missing values,
# 1tdUrea: 556 missing values, 1appDaysUrea: 556 missing values, 2tdUrea: 2694 missing values, 2appDaysUrea: 2700 missing values, Harv_hand_rent: 252 missing value

# Drop columns with more than 50% missing values
# df = df.drop(['Ganaura', 'CropOrgFYM', 'BasalUrea', '1tdUrea', '1appDaysUrea', '2tdUrea', '2appDaysUrea'], axis=1)
# Create interaction terms for numerical features
df['CultLand_CropCultLand'] = df['CultLand'] * df['CropCultLand']
df['BasalDAP_BasalUrea'] = df['BasalDAP'] * df['BasalUrea']

# Display first few rows to verify the new columns
df[['CultLand', 'CropCultLand', 'CultLand_CropCultLand', 'BasalDAP', 'BasalUrea', 'BasalDAP_BasalUrea']].head()

# Generate aggregated statistics for categorical features
agg_features = df.groupby('District')['Yield'].agg(['mean', 'std']).reset_index()
agg_features.columns = ['District', 'District_Yield_mean', 'District_Yield_std']
df = pd.merge(df, agg_features, on='District', how='left')

agg_features = df.groupby('Block')['Yield'].agg(['mean', 'std']).reset_index()
agg_features.columns = ['Block', 'Block_Yield_mean', 'Block_Yield_std']
df = pd.merge(df, agg_features, on='Block', how='left')

# Display first few rows to verify the new columns
df[['District', 'District_Yield_mean', 'District_Yield_std', 'Block', 'Block_Yield_mean', 'Block_Yield_std']].head()

# Debug test
print(df[['Harv_date', 'SeedingSowingTransplanting']].dtypes)
df['Harv_date'] = pd.to_datetime(df['Harv_date'], errors='coerce')
df['SeedingSowingTransplanting'] = pd.to_datetime(df['SeedingSowingTransplanting'], errors='coerce')
df['CropTillageDate'] = pd.to_datetime(df['CropTillageDate'], errors='coerce')
df['Threshing_date'] = pd.to_datetime(df['Threshing_date'], errors='coerce')


# Create a new column 'Yield' by adding up the yield from the 3 harvests
df['sow_to_harv'] = (df['Harv_date'] - df['SeedingSowingTransplanting']).dt.days
df['til_to_harv'] = (df['Harv_date'] - df['CropTillageDate']).dt.days
df['thresh_to_harv'] = (df['Threshing_date'] - df['Harv_date']).dt.days
df['Harv_month'] = df['Harv_date'].dt.month
df['Harv_quarter'] = df['Harv_date'].dt.to_period("Q")

# Debug test
df['TillageBeforeSowing'] = (df['CropTillageDate'] < df['SeedingSowingTransplanting']).astype(int)

# Calculate basic statistical measures for the target variable 'Yield'
yield_stats = df['Yield'].describe()
yield_stats


# Set the style for the plot
sns.set(style="whitegrid")

# Plot the distribution of 'Yield'
plt.figure(figsize=(12, 6))
sns.histplot(df['Yield'], bins=50, kde=True)
plt.title('Distribution of Yield')
plt.xlabel('Yield')
plt.ylabel('Frequency')
plt.show()

# Plot the boxplot for 'Yield'
plt.figure(figsize=(12, 6))
sns.boxplot(x=df['Yield'])
plt.title('Boxplot of Yield')
plt.xlabel('Yield')
plt.show()

# Super severe outliers are present in the target variable 'Yield'
# Simple test if log transformation can reduce the skewness of the target variable
df['Yield_log'] = np.log1p(df['Yield'])

# Plot the distribution of 'Yield' from raw data and log transformed data
_, axes = plt.subplots(1, 2, figsize=(18, 6))

# Raw data for the target variable
sns.histplot(df['Yield'], bins=50, kde=True, ax=axes[0])
axes[0].set_title('Distribution of Yield (Before Transformation)')
axes[0].set_xlabel('Yield')
axes[0].set_ylabel('Frequency')

# Log Transformed data for the target variable
sns.histplot(df['Yield_log'], bins=50, kde=True, ax=axes[1])
axes[1].set_title('Distribution of Yield (After Log Transformation)')
axes[1].set_xlabel('Log(Yield)')
axes[1].set_ylabel('Frequency')

plt.tight_layout()
plt.show()

# Try to do some basic feature engineering
# Create a new column 'TotalFertilizer' by adding up all the fertilizer columns
df['TotalFertilizer'] = df['BasalDAP'] + df['BasalUrea'] + df['1tdUrea'] + df['2tdUrea']

# Plot the distribution of 'TotalFertilizer'
plt.figure(figsize=(12, 6))
sns.histplot(df['TotalFertilizer'], bins=50, kde=True)
plt.title('Distribution of TotalFertilizer')
plt.xlabel('TotalFertilizer')
plt.ylabel('Frequency')
plt.show()
# Looks much smoother than every single fertilizer column

# Build a data column, a possible season feature column would be added in the future
date_columns = df.select_dtypes(include=[object]).columns[df.select_dtypes(include=[object]).apply(lambda x: x.str.contains(r'\d{4}-\d{2}-\d{2}', na=False).any())]
print(date_columns)

# Extract year, month, and day from each date column
for col in date_columns:
    df[col + '_year'] = pd.to_datetime(df[col]).dt.year
    df[col + '_month'] = pd.to_datetime(df[col]).dt.month
    df[col + '_day'] = pd.to_datetime(df[col]).dt.day

# Display first few rows to verify the new columns
df[list(date_columns) + [col + '_year' for col in date_columns] + [col + '_month' for col in date_columns] + [col + '_day' for col in date_columns]].head()

# Visualize the distribution of the new columns
_, axes = plt.subplots(1, 3, figsize=(18, 6))

sns.histplot(df['RcNursEstDate_year'], bins=50, kde=True, ax=axes[0])
axes[0].set_title('Distribution of RcNursEstDate_year')
axes[0].set_xlabel('RcNursEstDate_year')
axes[0].set_ylabel('Frequency')

sns.histplot(df['RcNursEstDate_month'], bins=50, kde=True, ax=axes[1])
axes[1].set_title('Distribution of RcNursEstDate_month')
axes[1].set_xlabel('RcNursEstDate_month')
axes[1].set_ylabel('Frequency')

sns.histplot(df['RcNursEstDate_day'], bins=50, kde=True, ax=axes[2])
axes[2].set_title('Distribution of RcNursEstDate_day')
axes[2].set_xlabel('RcNursEstDate_day')
axes[2].set_ylabel('Frequency')

plt.tight_layout()
plt.show()

# Check the unique values of year, month, and day columns
df['RcNursEstDate_year'].unique()
df['RcNursEstDate_month'].unique()
df['RcNursEstDate_day'].unique()

# It seems that the year column is not useful, so we can drop it
df = df.drop(['RcNursEstDate_year'], axis=1)
df = df.drop('date_column', axis=1)

# Calculate basic statistical measures for numerical features (excluding the target variable 'Yield' and engineered features)
numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
numerical_columns.remove('Yield')  # Remove the target variable
numerical_columns = [col for col in numerical_columns if '_log' not in col]  # Remove log-transformed columns
numerical_columns = [col for col in numerical_columns if '_mean' not in col and '_std' not in col]  # Remove aggregated features

# Calculate basic statistics
basic_stats = df[numerical_columns].describe().transpose()
basic_stats

# Plot boxplots for numerical features in batches to visually inspect for outliers
batch_size = 10  # Number of features to plot in each batch
num_batches = len(numerical_columns) // batch_size + 1  # Calculate the number of batches

for i in range(num_batches):
    start_idx = i * batch_size
    end_idx = start_idx + batch_size
    batch_features = numerical_columns[start_idx:end_idx]
    
    plt.figure(figsize=(15, 10))
    df[batch_features].boxplot()
    plt.title(f'Boxplot of Features (Batch {i+1})')
    plt.xticks(rotation=90)
    plt.show()

# Calculate the Z-score for each data point in the numerical features
df_zscore = df[numerical_columns].apply(zscore)

# Identify outliers where Z-score > 3 or Z-score < -3
outliers = (df_zscore > 3) | (df_zscore < -3)

# Replace outliers with median of the respective feature
for col in numerical_columns:
    median_value = df[col].median()
    df[col] = df[col].mask(outliers[col], median_value)

# Check if the operation was successful by recalculating the Z-scores and identifying any outliers
df_zscore_new = df[numerical_columns].apply(zscore)
outliers_new = (df_zscore_new > 3) | (df_zscore_new < -3)
outliers_new.sum()

# Perform additional rounds of Z-score-based outlier replacement until no more outliers are found
max_iterations = 5  # Limit the number of iterations to avoid infinite loops
iteration = 0

while iteration < max_iterations:
    # Calculate the Z-score for each data point in the numerical features
    df_zscore = df[numerical_columns].apply(zscore)
    
    # Identify outliers where Z-score > 3 or Z-score < -3
    outliers = (df_zscore > 3) | (df_zscore < -3)
    
    # Count the number of outliers in each feature
    num_outliers = outliers.sum()
    
    # If no more outliers, break the loop
    if num_outliers.sum() == 0:
        break
    
    # Otherwise, replace outliers with median of the respective feature
    for col in numerical_columns:
        median_value = df[col].median()
        df[col] = df[col].mask(outliers[col], median_value)
    
    iteration += 1

# Final check: Calculate the Z-scores again and count any remaining outliers
df_zscore_final = df[numerical_columns].apply(zscore)
outliers_final = (df_zscore_final > 3) | (df_zscore_final < -3)
outliers_final_count = outliers_final.sum()
outliers_final_count

from sklearn.model_selection import train_test_split

# Features and target variable
X = df.drop(columns=['Yield', 'Yield_log'])
y = df['Yield_log']  # Using the log-transformed target variable

categorical_columns = df.select_dtypes(include=['object']).columns

for column in categorical_columns:
    if df[column].nunique() > 2:  # Assuming a threshold for one-hot encoding
        # One-Hot Encoding
        ohe = OneHotEncoder()
        encoded_matrix = ohe.fit_transform(df[[column]])
        encoded_df = pd.DataFrame(encoded_matrix.toarray(), columns=ohe.get_feature_names_out([column]))
        df = df.drop(column, axis=1).join(encoded_df)
    else:
        # Label Encoding
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check the shape of the training and test sets
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Check the distribution of the target variable in the training and test sets
_, axes = plt.subplots(1, 2, figsize=(18, 6))

sns.histplot(y_train, bins=50, kde=True, ax=axes[0])
axes[0].set_title('Distribution of Yield (Training Set)')
axes[0].set_xlabel('Yield')
axes[0].set_ylabel('Frequency')

sns.histplot(y_test, bins=50, kde=True, ax=axes[1])
axes[1].set_title('Distribution of Yield (Test Set)')
axes[1].set_xlabel('Yield')
axes[1].set_ylabel('Frequency')

plt.tight_layout()
plt.show()

import lightgbm as lgb

# Initialize and train the LightGBM model
lgb_model = lgb.LGBMRegressor()
lgb_model.fit(X_train, y_train)






