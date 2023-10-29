# This is a simple showcase of how to use OpenFE in AutoML
# A complicated dataset is used to show the power of OpenFE

from openfe import OpenFE, transform
import pandas as pd
from sklearn.model_selection import train_test_split

ofe = OpenFE()

# Load the dataset
file_path = r'C:\Users\ra78lof\Not-so-auto-ml\Train.csv'
df = pd.read_csv(file_path)

# Debugging, to see if we have the right dataset
print(df.head())

# Split the dataset into X and y
X = df.drop(['yield'], axis=1)
y = df['yield']

X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.2, random_state=42)

# Generate features by using OpenFE
features = ofe.fit(data = X_train, label = y_train, n_jobs=1)

