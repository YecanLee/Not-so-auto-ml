# Original dataset is needed again for a while
dataset_original = pd.read_csv('')

# Prepare the data again LMAO
dataset_original = dataset_original.drop(columns=['Yield'])  # Features

# Handling missing values: filling NaNs with median of the column
# dataset_original = dataset_original.apply(lambda x: x.fillna(x.median()),axis=0)
# Upper line would cause bug
numerical_features = X.select_dtypes(include=['float64', 'int64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# Fill missing values based on their dtype
dataset_original[numerical_features] = X[numerical_features].apply(lambda x: x.fillna(x.median()), axis=0)
dataset_original[categorical_features] = X[categorical_features].apply(lambda x: x.fillna(x.mode()[0]), axis=0)

# Correlation Analysis
correlation_matrix = X.corr().abs()

# Select upper triangle of correlation matrix
upper = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(np.bool))

# Identify columns to drop based on a threshold correlation value
threshold = 0.8
to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

# Drop highly correlated columns
dataset_original_selected = dataset_original.drop(columns=to_drop)

# numerical features transformed
selector = VarianceThreshold(threshold=0.1)
dataset_original_selected = pd.DataFrame(selector.fit_transform(dataset_original[numerical_features]), columns=numerical_features[selector.get_support()])

# Merge the numerical columns with the categorical one
data_fine_tuned = pd.concat([dataset_original_selected, X[categorical_features]], axis=1)

# Debug by printing out the shape
print(data_fine_tuned.shape)

# Prepare the target variable and handle its missing values
y = dataset_original['Yield']
y = y.fillna(y.median())

# Split the reduced data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data_fine_tuned, y, test_size=0.2, random_state=42)

# Super shitty baseline
from sklearn.linear_model import LinearRegression

# Initialize and train the model
linear_model = LinearRegression()

# This stupid baseline does not work with categorical features
linear_model.fit(X_train.select_dtypes(include=['float64', 'int64']), y_train) 

y_pred = linear_model.predict(X_test.select_dtypes(include=['float64', 'int64']))

# A super shitty baseline result
rmse_linear = np.sqrt(mean_squared_error(y_test, y_pred))
rmse_linear

###-----------###
### lightgbm one
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
lgb_model = lgb.LGBMRegressor(objective='regression', num_leaves=30, learning_rate=0.01, n_estimators=1000)
lgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='rmse', early_stopping_rounds=100)

y_pred_1 = lgb_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_1))

# Show the most important features
lgb.plot_importance(lgb_model)

###--------------###
# Catboost
from catboost import CatBoostRegressor
cat_model = CatBoostRegressor(iterations=150, learning_rate=0.1, depth=10)
cat_model.fit(X_train, y_train, eval_set=(X_test, y_test), cat_features=categorical_features)

y_pred_2 = cat_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_2))
feature_importances = cat_model.feature_importances_

###---------------###
import xgboost as xgb
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
xgb_model.fit(X_train, y_train)

y_pred_3 = xgb_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_3))

xgb.plot_importance(xgb_model)

###----------------###
### Basic Ensemble
weights = [0.2, 0.2, 0.6]
final_predictions = np.average(np.array([y_pred_1, y_pred_2, y_pred_3]))

