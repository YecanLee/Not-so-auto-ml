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