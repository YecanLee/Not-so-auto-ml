import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the dataset
path = r'C:\Users\LMMISTA-WAP803\Not-so-auto-ml\Train.csv'
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
Last Crop related 
"""