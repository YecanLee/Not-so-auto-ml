import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# In the second part of the notebook
from sklearn.preprocessing import LabelEncoder

# This part of importing would be moved to main.py or train.py later
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from catboost import CatBoostRegressor

# Variance Threshold:
from sklearn.feature_selection import VarianceThreshold


# Read the dataset
path = r'C:\Users\ra78lof\Not-so-auto-ml\Train.csv'
dataset = pd.read_csv(path)


# debug
print(dataset.head())

# Basic EDA
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Plot distribution for 'District'
plt.subplot(1, 2, 1)
sns.countplot(data=dataset, x='District')
plt.title('Distribution of Districts')

# Plot distribution for 'Block'
plt.subplot(1, 2, 2)
sns.countplot(data=dataset, x='Block')
plt.title('Distribution of Blocks')
plt.xticks(rotation=45)
plt.yticks(rotation=45)

plt.tight_layout()
plt.show()


# Continue the EDA
fig, ax = plt.subplots(1, 2, figsize=(14, 14))

# Plot distribution for 'CultLand'
plt.subplot(1, 2, 1)
sns.histplot(dataset['CultLand'], bins=30, kde=True)
plt.title('Distribution of CultLand')

# Plot distribution for 'CropCultLand'
plt.subplot(1, 2, 2)
sns.histplot(dataset['CropCultLand'], bins=30, kde=True)
plt.title('Distribution of CropCultLand')

plt.xticks(rotation=45)
plt.yticks(rotation=45)

plt.tight_layout()
plt.show()

"""
Super skewed data. Log transformation is needed. Or remove the outliers. 
"""

# Plot distribution for 'LandPreparationMethod'
# Continue the EDA
fig, ax = plt.subplots(1, 2, figsize=(14, 14))
plt.subplot(1, 2, 1)
sns.countplot(data=dataset, y='LandPreparationMethod')
plt.title('Distribution of Land Preparation Methods')

# Plot distribution for 'CropTillageDepth'
plt.subplot(1, 2, 2)
sns.histplot(dataset['CropTillageDepth'], bins=8, kde=True)
plt.title('Distribution of Crop Tillage Depth')

plt.xticks(rotation=45)
plt.yticks(rotation=45)

plt.tight_layout()
plt.show()

"""
Super imbalanced data for 'LandPreparationMethod'. Not so sure should we keep it or not.
Quite balanced data for 'CropTillageDepth'. Bin size as 8 is good enough.
"""

# Plot distribution for 'CropEstMethod'
sns.countplot(data=dataset, y='CropEstMethod')
plt.title('Distribution of Crop Establishment Methods')

plt.xticks(rotation=45)
plt.yticks(rotation=45)

plt.show()

"""
Last Crop related columns
Manual_PuddledRandom' is the most frequently used method for crop establishment.
"""

###-------------------###

"""
Following section for nursery and transplantation details
"""
#  RcNursEstDate is a date column. The missin values should be filled with foward fill or backward fill.
# NursDetFactor, TransDetFactor should also got plotted

# fig, ax method looks too ugly
plt.figure(figsize=(20, 18))

# Plot distribution for 'SeedlingsPerPit'
plt.subplot(3, 2, 1)
sns.histplot(dataset['SeedlingsPerPit'].dropna(), bins=20, kde=True)
plt.title('Distribution of Seedlings Per Pit')

# Plot distribution for 'TransplantingIrrigationHours'
plt.subplot(3, 2, 2)
sns.histplot(dataset['TransplantingIrrigationHours'].dropna(), bins=20, kde=True)
plt.title('Distribution of Transplanting Irrigation Hours')

# Plot distribution for 'TransplantingIrrigationSource'
plt.subplot(3, 2, 3)
sns.countplot(data=dataset, y='TransplantingIrrigationSource')
plt.title('Distribution of Transplanting Irrigation Source')

# Plot distribution for 'TransplantingIrrigationPowerSource'
plt.subplot(3, 2, 4)
sns.countplot(data=dataset, y='TransplantingIrrigationPowerSource')
plt.title('Distribution of Transplanting Irrigation Power Source')

# Plot distribution for 'TransIrriCost'
plt.subplot(3, 2, 5)
sns.histplot(dataset['TransIrriCost'].dropna(), bins=20, kde=True)
plt.title('Distribution of Transplanting Irrigation Cost')

plt.xticks(rotation=45)
plt.yticks(rotation=45)

plt.tight_layout()
plt.show()

###-------------------###

"""
This section focuse on fertilizers and water management
"""

# Start the visualization eda again
plt.figure(figsize=(20, 18))

# Plot distribution for 'StandingWater'
plt.subplot(3, 3, 1)
sns.histplot(dataset['StandingWater'].dropna(), bins=20, kde=True)
plt.title('Distribution of Standing Water Days')

# Plot distribution for 'Ganaura'
plt.subplot(3, 3, 2)
sns.histplot(dataset['Ganaura'].dropna(), bins=20, kde=True)
plt.title('Distribution of Ganaura (Organic Fertilizer Amount in Quintals)')

# Plot distribution for 'CropOrgFYM'
plt.subplot(3, 3, 3)
sns.histplot(dataset['CropOrgFYM'].dropna(), bins=20, kde=True)
plt.title('Distribution of CropOrgFYM (FYM Fertilizer Amount in Quintals)')

# Plot distribution for 'NoFertilizerAppln'
plt.subplot(3, 3, 4)
sns.histplot(dataset['NoFertilizerAppln'].dropna(), bins=20, kde=True)
plt.title('Distribution of No. of Fertilizer Applications')

# Plot distribution for 'BasalDAP'
plt.subplot(3, 3, 5)
sns.histplot(dataset['BasalDAP'].dropna(), bins=20, kde=True)
plt.title('Distribution of Basal DAP Amount (in kgs)')

# Plot distribution for 'BasalUrea'
plt.subplot(3, 3, 6)
sns.histplot(dataset['BasalUrea'].dropna(), bins=20, kde=True)
plt.title('Distribution of Basal Urea Amount (in kgs)')

plt.tight_layout()
plt.show()

# Standing water missing values should be filled with median.
# OrgFertilizers missing values should be filled with mode.
# Ganuana imputing withe median (skewness)
# CropOrgFYM should also be filled with median
# PCropSolidOrgFertAppMethod should be filled with mode
# CropbasalFerts should be filled with mode
# BasalUrea, BaselDAP skewed, median value may be filled to make things look better

###-----------------###

"""
The last section would be harvesting and post-harvesting details
"""
# fig, ax method looks too ugly
plt.figure(figsize=(20, 18))

# Plot distribution for 'Harv_hand_rent'
plt.subplot(3, 3, 1)
sns.histplot(dataset['Harv_hand_rent'].dropna(), bins=20, kde=True)
plt.title('Distribution of Harvesting Hand Rent (in rupees)')

# Plot distribution for 'Residue_length'
plt.subplot(3, 3, 2)
sns.histplot(dataset['Residue_length'].dropna(), bins=20, kde=True)
plt.title('Distribution of Residue Length')

# Plot distribution for 'Residue_perc'
plt.subplot(3, 3, 3)
sns.histplot(dataset['Residue_perc'].dropna(), bins=20, kde=True)
plt.title('Distribution of Residue Percentage')

# Plot distribution for 'Acre'
plt.subplot(3, 3, 4)
sns.histplot(dataset['Acre'].dropna(), bins=20, kde=True)
plt.title('Distribution of Area (in acres)')

# Plot distribution for 'Yield'
plt.subplot(3, 3, 5)
sns.histplot(dataset['Yield'].dropna(), bins=20, kde=True)
plt.title('Distribution of Yield')

plt.tight_layout()
plt.show()

# Harv_method not plotted, median filling may be used since it is right skewed
# Threshing_method not plotted, categorical column, median filling may be used since it is right skewed
# skewed distribution may be filled with median, categorical may be filled with mode

# Outlier detection should be done before we deal with the skewness
from scipy.stats import zscore

# Outlier detection based on skewed features.
skewed_features = ['SeedlingsPerPit', 'TransplantingIrrigationHours', 'TransIrriCost', 
                   'StandingWater', 'Ganaura', 'CropOrgFYM', 'NoFertilizerAppln',
                   'BasalDAP', 'BasalUrea', 'Harv_hand_rent', 'Residue_length',
                   'Residue_perc', 'Acre']

# Calculate Z-scores for these features
z_scores = dataset[skewed_features].apply(zscore, nan_policy='omit')

# Create a DataFrame to store features and the count of outliers based on Z-score > 3
outliers_count = {}
for feature in skewed_features:
    outliers_count[feature] = len(z_scores.loc[abs(z_scores[feature]) > 3])

outliers_count_df = pd.DataFrame(list(outliers_count.items()), columns=['Feature', 'Outliers_Count'])

outliers_count_df.sort_values(by='Outliers_Count', ascending=False)

# Residual Perc and Standing water columns are super bad based on z-score
# Interquartile Range method may be used to retest the correctl

# SeedlingsPerpit&NoFertilizerAppln may just be keeped since the ourlier numbers are small
# A test preprocessing method, truncate the data to a upper bound
# ALERT! FUTURE EXPLORARION MAY BE NEEDED!
high_skewed_features = ['Residual Perc', 'StandingWater']
low_skewed_features = ['Residue_length', 'CropOrgFYM', 'Harv_hand_rent']
moderate_skewed_features = ['BasalDAP', 'Acre', 
                            'TransIrriCost', 'Ganaura',
                            'BasalUrea', 'TransplantingIrrigationHours']

# Outlier remove preprocessing
# Cap at 99th percentile
for feature in moderate_skewed_features:
    percentile_value = dataset[feature].quantile(0.99)
    dataset.loc[dataset[feature] > percentile_value, feature] = percentile_value

# Remove outliers
for feature in low_skewed_features:
    z_scores = zscore(dataset[feature].dropna())
    abs_z_scores = abs(z_scores)
    filtered_entries = (abs_z_scores <= 3)
    dataset = dataset.loc[dataset[feature].index.isin(dataset[feature].dropna().index[filtered_entries])]

# Debug, print the dataframe shape after this preprocessing
print(dataset.shape)
dataset.head()

# ALERT, not so sure whether the right-skewed numerical features needed to be processed!
# Apply log transformation to right-skewed numerical features
log_transform_features = ['TransIrriCost', 'StandingWater', 'Ganaura']
dataset[log_transform_features] = dataset[log_transform_features].apply(np.log1p)

# Binning the SeedlingsPerPit
bins_SeedlingsPerPit = [0, 2, 4, np.inf]
labels_SeedlingsPerPit = ['Low', 'Medium', 'High']
dataset['SeedlingsPerPit_Binned'] = pd.cut(dataset['SeedlingsPerPit'], bins=bins_SeedlingsPerPit, labels=labels_SeedlingsPerPit)

bins_NoFertilizerAppln = [0, 1, 2, np.inf]
labels_NoFertilizerAppln = ['Low', 'Medium', 'High']
dataset['NoFertilizerAppln_Binned'] = pd.cut(dataset['NoFertilizerAppln'], bins=bins_NoFertilizerAppln, labels=labels_NoFertilizerAppln)

# Datetime features generator
date_features = ['CropTillageDate', 'RcNursEstDate', 'SeedingSowingTransplanting', 'Harv_date', 'Threshing_date']
for feature in date_features:
    dataset[feature] = pd.to_datetime(dataset[feature], errors='coerce')
    dataset[f"{feature}_Month"] = dataset[feature].dt.month

# Apply one-hot encoding to non-ordinal categorical features
one_hot_features = ['District', 'Block', 'LandPreparationMethod']
dataset = pd.get_dummies(dataset, columns=one_hot_features, drop_first=True)

# Apply label encoding to ordinal categorical features
label_encode_features = ['Harv_method', 'Threshing_method']
label_encoder = LabelEncoder()
for feature in label_encode_features:
    dataset[feature] = dataset[feature].astype(str)
    dataset[feature] = label_encoder.fit_transform(dataset[feature])

# Debug,for dataset shape after modification
print(dataset.shape)
print(dataset.head())
### DEBUG
# Visualize the distributions after transformations

plt.figure(figsize=(20, 12))

plt.subplot(2, 3, 1)
sns.histplot(dataset['TransIrriCost'].dropna(), bins=20, kde=True)
plt.title('Log Transformed TransIrriCost')

plt.subplot(2, 3, 2)
sns.histplot(dataset['StandingWater'].dropna(), bins=20, kde=True)
plt.title('Log Transformed StandingWater')

plt.subplot(2, 3, 3)
sns.histplot(dataset['Ganaura'].dropna(), bins=20, kde=True)
plt.title('Log Transformed Ganaura')

# Binned features
plt.subplot(2, 3, 4)
sns.countplot(data=dataset, x='SeedlingsPerPit_Binned')
plt.title('Binned SeedlingsPerPit')

plt.subplot(2, 3, 5)
sns.countplot(data=dataset, x='NoFertilizerAppln_Binned')
plt.title('Binned NoFertilizerAppln')

plt.xticks(rotation = 45)
plt.yticks(rotation = 45)

plt.tight_layout()
plt.show()

### lightgbm, catboost and xgboost would be tested first ###
### xgboost is the only method that needs those transformations ###
### Consider the early stopping

# Original dataset is needed again for a while
dataset_original = pd.read_csv(r'C:\Users\ra78lof\Not-so-auto-ml\Train.csv')

# Feature Cross
dataset_original['District_Block'] = dataset_original['District'].astype(str) + "_" + dataset_original['Block'].astype(str)

# K-means Clustering
kmeans = KMeans(n_clusters=3, random_state=0)
subset_for_clustering = dataset_original[['CultLand', 'CropCultLand']].dropna()
kmeans_labels = kmeans.fit_predict(subset_for_clustering)
dataset_original.loc[subset_for_clustering.index, 'Land_Cluster'] = kmeans_labels

# PCA 
numerical_features_original = dataset_original.select_dtypes(include=['float64', 'int64']).columns
pca = PCA(n_components=2)  # ALERT! Needed to be adjusted in the final phase
principal_components = pca.fit_transform(dataset_original[numerical_features_original].fillna(0))
print(principal_components[:, 0])
print(principal_components[:, 1])



dataset_original['PCA1'], dataset_original['PCA2'] = principal_components[:, 0], principal_components[:, 1]
print(dataset_original['PCA1'].head())
print(dataset_original['PCA2'].head())



# Add to the dataset, ALERT, not the dataset_original
dataset['District_Block'] = dataset_original['District_Block']
dataset['Land_Cluster'] = dataset_original['Land_Cluster']
dataset['PCA1'] = dataset_original['PCA1']
dataset['PCA2'] = dataset_original['PCA2']

# ALERT, this code snippt should not be used during the xgboost training!
one_hot_columns = [col for col in dataset.columns if 'District_' in col or 'Block_' in col]
dataset = dataset.drop(columns=one_hot_columns)
# ALERT, NOT for xgboost
dataset['District'] = dataset_original['District']
dataset['Block'] = dataset_original['Block']

# Debug with data shape
print(dataset.shape)
print(dataset.head())

"""
Got a bug , here is the error message:
ValueError: pandas dtypes must be int, float or bool.
Fields with bad pandas dtypes: ID: object, CropTillageDate: datetime64[ns], CropEstMethod: object, RcNursEstDate: datetime64[ns], 
SeedingSowingTransplanting: datetime64[ns], NursDetFactor: object, TransDetFactor: object, TransplantingIrrigationSource: object, 
TransplantingIrrigationPowerSource: object, OrgFertilizers: object, PCropSolidOrgFertAppMethod: object, CropbasalFerts: object, 
MineralFertAppMethod: object, FirstTopDressFert: object, MineralFertAppMethod.1: object, Harv_date: datetime64[ns], Threshing_date: datetime64[ns], Stubble_use: object
"""
#DEBUG
datetime_columns = dataset.select_dtypes(include=[np.datetime64]).columns.tolist()
print(datetime_columns)
# Assuming df is your DataFrame
for col in dataset.columns:
    if dataset[col].dtype == 'object':
        dataset[col] = dataset[col].astype('category').cat.codes
    elif dataset[col].dtype == 'datetime64[ns]':
        dataset = dataset.drop(columns=[col])
dataset.head()
dataset.info()

# A simple test with the lightgbm
# Prepare the data
X = dataset.drop(columns=['Yield']) 
y = dataset['Yield']


# Basic Handing, probably won't be the best solution
# X = X.apply(lambda x: x.fillna(x.median()),axis=0)

# Convert Object dtype to category dtype
categorical_cols = ['District', 'Block', 'SeedlingsPerPit_Binned', 'NoFertilizerAppln_Binned']
X[categorical_cols] = X[categorical_cols].astype('category')

# Train Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=200000)

# Create a LightGBM model
lgbm_model = lgb.LGBMRegressor(objective='regression', num_leaves=31, learning_rate=0.05, n_estimators=100)

# Train the model
lgbm_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='rmse',
                categorical_feature=categorical_cols)

# Make predictions
y_pred = lgbm_model.predict(X_test, num_iteration=lgbm_model.best_iteration_)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(rmse)

# test set prediction
import pandas as pd
test_path = r'C:\Users\ra78lof\Not-so-auto-ml\Test.csv'
dataset_test = pd.read_csv(test_path)

# debug 
print(dataset_test.head())

# Feature Cross
dataset_test['District_Block'] = dataset_test['District'].astype(str) + "_" + dataset_test['Block'].astype(str)

# K-means Clustering
kmeans = KMeans(n_clusters=3, random_state=0)
subset_for_clustering = dataset_test[['CultLand', 'CropCultLand']].dropna()
kmeans_labels = kmeans.fit_predict(subset_for_clustering)
dataset_test.loc[subset_for_clustering.index, 'Land_Cluster'] = kmeans_labels

# PCA
numerical_features_original = dataset_test.select_dtypes(include=['float64', 'int64']).columns
pca = PCA(n_components=2)  # ALERT! Needed to be adjusted in the final phase
principal_components = pca.fit_transform(dataset_test[numerical_features_original].fillna(0))
dataset_test['PCA1'], dataset_test['PCA2'] = principal_components[:, 0], principal_components[:, 1]

# Add to the dataset, ALERT, not the dataset_original
dataset_test['District_Block'] = dataset_test['District_Block']
dataset_test['Land_Cluster'] = dataset_test['Land_Cluster']
dataset_test['PCA1'] = dataset_test['PCA1']
dataset_test['PCA2'] = dataset_test['PCA2']

# ALERT, this code snippt should not be used during the xgboost training!
one_hot_columns = [col for col in dataset_test.columns if 'District_' in col or 'Block_' in col]
dataset_test = dataset_test.drop(columns=one_hot_columns)
# ALERT, NOT for xgboost
dataset_test['District'] = dataset_test['District']
dataset_test['Block'] = dataset_test['Block']

# Debug with data shape
print(dataset_test.shape)
print(dataset_test.head())

# Assuming df is your DataFrame
for col in dataset_test.columns:
    if dataset_test[col].dtype == 'object':
        dataset_test[col] = dataset_test[col].astype('category').cat.codes
    elif dataset_test[col].dtype == 'datetime64[ns]':
        dataset_test = dataset_test.drop(columns=[col])



dataset_test.head()
dataset_test.info()
# Make predictions
y_pred = lgbm_model.predict(dataset_test, num_iteration=lgbm_model.best_iteration_)
y_pred

dataset_upload = pd.read_csv(r'C:\Users\ra78lof\Not-so-auto-ml\Test.csv')
# Create a submission file
submission = pd.DataFrame({'ID': dataset_upload['ID'], 'Yield': y_pred})
print(submission.head())
sys.exit()

submission.to_csv('submission.csv', index=False)

