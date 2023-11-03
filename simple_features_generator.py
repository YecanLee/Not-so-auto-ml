import pandas as pd
path = r'C:\Users\LMMISTA-WAP803\Not-so-auto-ml\Train.csv'
dataset = pd.read_csv(path)

df = dataset.copy()

# Temporal Features
# ALERT! NEED TO BE REMOVED LATER!
df['CropTillageDate'] = pd.to_datetime(df['CropTillageDate'])
df['Harv_date'] = pd.to_datetime(df['Harv_date'])

# Time to Harvest
df['Time_to_Harvest'] = (df['Harv_date'] - df['CropTillageDate']).dt.days

# Season 
df['Season'] = df['CropTillageDate'].dt.month % 12 // 3
season_mapping = {0: 'Winter', 1: 'Spring', 2: 'Summer', 3: 'Autumn'}
df['Season'] = df['Season'].map(season_mapping)

# Interaction Features
df['Fertilizer_per_Acre'] = df['BasalDAP'] / df['Acre']
df['Water_per_Acre'] = df['TransplantingIrrigationHours'] / df['Acre']
df['Seedlings_per_Acre'] = df['SeedlingsPerPit'] / df['Acre']

# Aggregation Features
df['Total_Fertilizer'] = df['BasalDAP'] + df['BasalUrea']
df['Total_Urea_Applied'] = df['BasalUrea'] + df['1tdUrea'] + df['2tdUrea']

# Polynomial Features
df['Square_of_CultLand'] = df['CultLand'] ** 2
df['Cube_of_CultLand'] = df['CultLand'] ** 3

df['Square_of_CropCultLand'] = df['CropCultLand'] ** 2
df['Cube_of_CropCultLand'] = df['CropCultLand'] ** 3

# Ratio Features
df['Crop_Land_to_Cultivated_Land'] = df['CropCultLand'] / df['CultLand']
df['Organic_to_Chemical_Fertilizer'] = df['Ganaura'] / (df['BasalDAP'] + 1e-5)  # Added a small value to avoid division by zero

# Continuing to add more features to the existing 'df' dataframe.

# Efficiency Metrics
# Note: These features would ideally be created post-modeling, as they include the target variable 'Yield'
# df['Fertilizer_Efficiency'] = df['Yield'] / (df['Total_Fertilizer'] + 1e-5)
# df['Water_Efficiency'] = df['Yield'] / (df['TransplantingIrrigationHours'] + 1e-5)

# Advanced Temporal Features
# Convert additional date columns to datetime format for calculations
df['RcNursEstDate'] = pd.to_datetime(df['RcNursEstDate'])
df['SeedingSowingTransplanting'] = pd.to_datetime(df['SeedingSowingTransplanting'])
df['Threshing_date'] = pd.to_datetime(df['Threshing_date'])

# Days Between Nursery and Transplant
df['Days_Between_Nursery_Transplant'] = (df['SeedingSowingTransplanting'] - df['RcNursEstDate']).dt.days

# Days of Residue
df['Days_of_Residue'] = (df['Threshing_date'] - df['Harv_date']).dt.days

# Cost Features
# I guess this need to be done with VJH lmao
# fertilizer_cost_per_kg = 10  
# irrigation_cost_per_hour = 5  

# df['Cost_per_Acre'] = (df['Total_Fertilizer'] * fertilizer_cost_per_kg + df['TransplantingIrrigationHours'] * irrigation_cost_per_hour) / df['Acre']
df['Rent_per_Acre'] = df['Harv_hand_rent'] / df['Acre']

# Logistic Features
# Complexity
df['Irrigation_Complexity'] = df['TransplantingIrrigationSource'] + '_' + df['TransplantingIrrigationPowerSource']

# Efficiency!
df['Tillage_Method_Efficiency'] = df['LandPreparationMethod'].astype(str) + '_' + df['CropTillageDepth'].astype(str)

# Location features
df['Location'] = df['Block'] + '_' + df['District']

# Z-Scores features for 'CultLand' and 'CropCultLand', high chance to be some bullshit
df['Z_Score_CultLand'] = (df['CultLand'] - df['CultLand'].mean()) / df['CultLand'].std()
df['Z_Score_CropCultLand'] = (df['CropCultLand'] - df['CropCultLand'].mean()) / df['CropCultLand'].std()

# DEBUG, based on the new dataframe
df.sample(5)

# In case I have 1000 features at the end LMAO
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# X_new_scaled = scaler.fit_transform(X_new)

# from sklearn.feature_selection import SelectKBest, f_classif
# selector = SelectKBest(score_func=f_classif, k=35)
# X_new_selected = selector.fit_transform(X_new_scaled, y_new)