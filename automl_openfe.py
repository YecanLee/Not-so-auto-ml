# This is a simple showcase of how to use OpenFE in AutoML
# A complicated dataset is used to show the power of OpenFE

from openfe import OpenFE, transform
import pandas as pd
from helper import get_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

ofe = OpenFE()

# Load the dataset
file_path = r'C:\Users\ra78lof\Not-so-auto-ml\Train.csv'
df = pd.read_csv(file_path)

# Transfer the dtype Object columns 
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

# Debugging, to see if we have the right dataset
print(df.head())

# Split the dataset into X and y
X = df.drop(['Yield'], axis=1)
y = df['Yield']

X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Baseline score
score = get_score(X_train, x_test, y_train, y_test)
print("The MSE before feature generation is", score)

# Generate features by using OpenFE
features = ofe.fit(data = X_train, label = y_train, n_jobs=12, feature_boosting = True, metric='rmse')
print(ofe.new_features_list[:10])
print('The top 10 generated features are', ofe.new_features_list[:10])

# Test the newly generated features
train_x, test_x = transform(X_train, x_test, ofe.new_features_list[:10], n_jobs=12)


score = get_score(train_x, test_x, y_train, y_test)
print("The MSE after feature generation is", score)