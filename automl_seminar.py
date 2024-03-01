import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import numpy as np
import random


import sys
# Load the training data
train = pd.read_csv('Train.csv')

# Debugging: Display the first few rows of the training data
print(train.head())


# Handle the missing values, we will first test 
for i in train.columns:
    if train[i].dtype == 'object':
        train[i].fillna(train[i].mode()[0], inplace=True)
    else:
        train[i].fillna(train[i].mean(), inplace=True)

# Encoding categorical variables
label_encoders = {}
for i in train.select_dtypes(include=['object']).columns:
    # Skip the ID column
    if i != 'ID': 
        le = LabelEncoder()
        train[i] = le.fit_transform(train[i])
        label_encoders[i] = le

# Convert date features to numerical by calculating the number of days from a reference date
date_columns = train.columns[train.columns.str.contains('Date')]
# Convert all date columns to datetime
reference_date = pd.to_datetime('2022-01-01')  
for column in date_columns:
    train[column] = pd.to_datetime(train[column])
    train[column] = (train[column] - reference_date).dt.days

# Normalizing numerical features
scaler = MinMaxScaler()
numeric_columns = train.select_dtypes(include=[np.number]).columns.drop('Yield')
train[numeric_columns] = scaler.fit_transform(train[numeric_columns])

# Preparing the data for modeling
# Training Features
X = train.drop(['ID', 'Yield'], axis=1) 
# Target Feature
y = train['Yield']  

# Splitting the dataset into training and validation sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Debugging: Display the first few rows of the training data
X_train.head()


# First we will use the H2o AutoML library to train the model, 
# We will use three AutoML models in total, and then we will compare the results of the three models to select the best model.
import h2o
from h2o.automl import H2OAutoML

# Start H2O cluster
h2o.init()

# Convert to H2O Frame, We will use the whole dataset for training
h2o_train = h2o.H2OFrame(pd.concat([X_train, y_train], axis=1))
h2o_test = h2o.H2OFrame(pd.concat([X_test, y_test], axis=1))

# Specify target and features
x = h2o_train.columns
y = "Yield"
x.remove(y)

# Run H2O AutoML
aml = H2OAutoML(max_runtime_secs=300)
aml.train(x=x, y=y, training_frame=h2o_train)

# Evaluate model
pred = aml.leader.predict(h2o_test)

# Get leaderboard with all possible columns
lb = h2o.automl.get_leaderboard(aml, extra_columns = "ALL")
lb

# Get the best XGBoost model, ranked by logloss, this is just an example about how to use the method
# You can also use other algorithms and criterions
xgb = aml.get_best_model(algorithm="xgboost", criterion="logloss")

perf = aml.leader.model_performance(h2o_test)
print(perf)

# Use the newly developed method from H2o to explain the result
exa = aml.explain(h2o_test)

# Explain a single model
exm = aml.leader.explain(h2o_test)

# clean up the memory
import gc
gc.collect()
sys.exit()
"""

# Next we will use the AutoSklearn library to train the model
# We will only use the train dataset to train the model
"""
from autosklearn.regression import AutoSklearnRegressor

# Instantiate and fit the model
automl = AutoSklearnRegressor(time_left_for_this_task=3600, per_run_time_limit=300)
automl.fit(X_train, y_train)

# Evaluate the model
predictions = automl.predict(X_test)
 
sys.exit()

from tpot import TPOTRegressor

# Instantiate and fit the model
tpot = TPOTRegressor(generations=5, population_size=50, verbosity=2, random_state=42)
tpot.fit(X_train, y_train)

# Evaluate the model
print(tpot.score(X_test, y_test))
