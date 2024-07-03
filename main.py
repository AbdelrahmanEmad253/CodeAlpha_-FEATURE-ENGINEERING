#                                                                                   project details
# FEATURE ENGINEERING
# Create additional features that might be useful
# for predicting equipment failure.
# Consider time-based features, rolling statistics,
# and any other relevant transformations.
#
#                                                                           Steps to Perform Feature Engineering:
# Understanding the Data
# Data Cleaning
# Time-Based Features       xxxx
# Rolling Statistics
# Lag Features
# Cumulative Features
# Statistical Transformations
# Frequency-Based Features  xxxx
# Interaction Features
# Domain-Specific Features  xxxx
# Feature Selection
# Evaluation and Iteration

#                                                                                           code

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

###################################################################################
# Load the dataset
data = pd.read_csv('predictive-maintenance-dataset.csv')

# Display the first few rows of the dataset
print(data.head())

# Get the structure of the dataset
print(data.info())

# Get basic statistics of the dataset
print(data.describe())

# Check for missing values
print(data.isnull().sum())
###################################################################################
# Histogram of numerical features
data.hist(bins=30, figsize=(15, 10))
plt.show()

# Pairplot to visualize relationships between features
sns.pairplot(data)
plt.show()

# Heatmap to visualize correlation between features
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.show()

# Handle missing values in 'vibration' by filling with the median
data['vibration'] = data['vibration'].fillna(data['vibration'].median())

# Verify there are no missing values
print(data.isnull().sum())
#############################################################################
#                                                                           to find the best k value
# import pandas as pd
# import numpy as np
# from sklearn.impute import KNNImputer
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error
#
# # Split the data into training and validation sets
# train_data, valid_data = train_test_split(data, test_size=0.2, random_state=42)
#
# # Introduce missing values in the training set for evaluation purposes
# train_data_missing = train_data.copy()
# missing_mask = np.random.rand(*train_data_missing.shape) < 0.1
# train_data_missing = train_data_missing.mask(missing_mask)
#
# # Define a range of k values
# k_values = [3, 5, 7, 10]
# errors = []
#
# # Evaluate different k values
# for k in k_values:
#     imputer = KNNImputer(n_neighbors=k)
#     train_data_imputed = imputer.fit_transform(train_data_missing)
#
#     # Calculate the mean squared error for the artificially introduced missing values only
#     mask_flat = missing_mask.flatten()
#     mse = mean_squared_error(train_data.values.flatten()[mask_flat], train_data_imputed.flatten()[mask_flat])
#     errors.append(mse)
#     print(f"k: {k}, MSE: {mse}")
#
# # Select the best k value
# best_k = k_values[np.argmin(errors)]
# print(f"The best k value is: {best_k}")
#
# # Impute missing values in the entire dataset using the best k
# imputer = KNNImputer(n_neighbors=best_k)
# data_imputed = imputer.fit_transform(data)
#
# # Convert the imputed data back to a DataFrame
# data_imputed = pd.DataFrame(data_imputed, columns=data.columns)
#
# # Save the imputed dataset if necessary
# data_imputed.to_csv('predictive-maintenance-dataset-imputed.csv', index=False)
#
# print("KNN imputation completed successfully.")
###################################################################################
# Impute missing values using KNN
k = 5  # You can choose the optimal k based on previous steps or domain knowledge
imputer = KNNImputer(n_neighbors=k)
data_imputed = imputer.fit_transform(data)

# Convert the imputed data back to a DataFrame
data_imputed = pd.DataFrame(data_imputed, columns=data.columns)

# Evaluate the imputation performance using cross-validation with a regression model
X = data_imputed.drop(columns='vibration')  # Replace 'vibration' with the actual target column name
y = data_imputed['vibration']  # Replace 'vibration' with the actual target column name

# Define a pipeline with imputation and a regressor
pipeline = Pipeline([
    ('imputer', KNNImputer(n_neighbors=k)),
    ('regressor', RandomForestRegressor())
])

# Perform cross-validation to evaluate the performance
scores = cross_val_score(pipeline, X, y, cv=5, scoring='neg_mean_squared_error')
mean_score = np.mean(scores)
print(f"Mean Cross-Validation MSE: {-mean_score}")

# Save the imputed dataset if necessary
data_imputed.to_csv('predictive-maintenance-dataset-imputed.csv', index=False)

print("KNN imputation and evaluation completed successfully.")

# Check for missing values
data2 = pd.read_csv('predictive-maintenance-dataset-imputed.csv')
print(data2.isnull().sum())


#
# Function to identify and handle outliers using IQR
def handle_outliers(df):
    for COLUMN in df.select_dtypes(include=['float64', 'int64']).columns:
        q1 = df[COLUMN].quantile(0.25)
        q3 = df[COLUMN].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        # Cap the outliers
        df[COLUMN] = df[COLUMN].apply(
            lambda x: lower_bound if x < lower_bound else (upper_bound if x > upper_bound else x))
    return df


# Apply the function to handle outliers
data2 = handle_outliers(data2)

# Save the corrected dataset to the same file
data2.to_csv('predictive-maintenance-dataset-imputed.csv', index=False)

# Ensure correct data types
data2['ID'] = data2['ID'].astype(int)
data2['revolutions'] = data2['revolutions'].astype(float)
data2['humidity'] = data2['humidity'].astype(float)
data2['vibration'] = data2['vibration'].astype(float)
data2['x1'] = data2['x1'].astype(float)
data2['x2'] = data2['x2'].astype(float)
data2['x3'] = data2['x3'].astype(float)
data2['x4'] = data2['x4'].astype(float)
data2['x5'] = data2['x5'].astype(float)

# Check data types
print(data2.dtypes)

# Save the corrected dataset to the same file
data2.to_csv('predictive-maintenance-dataset-imputed.csv', index=False)

# Print summary statistics to identify any inconsistencies
print(data2.describe())

# Save the corrected dataset to the same file
data2.to_csv('predictive-maintenance-dataset-imputed.csv', index=False)

# If specific corrections are needed, perform them here
# Example: Correct negative values that shouldn't exist
data2['vibration'] = data2['vibration'].apply(lambda x: abs(x) if x < 0 else x)

# Assuming 'ID' column represents the order of data points
data2 = data2.sort_values(by='ID')

# Create lag features for the specified columns
lag_columns = ['revolutions', 'humidity', 'vibration', 'x1', 'x2', 'x3', 'x4', 'x5']
lag_periods = [1, 2, 3]

for column in lag_columns:
    for lag in lag_periods:
        data2[f'{column}_lag{lag}'] = data2[column].shift(lag)

# Check the dataset with new lag features
print(data2.head())

# Save the corrected dataset to the same file
data2.to_csv('predictive-maintenance-dataset-imputed.csv', index=False)

# Create rolling window statistics for the specified columns
window_size = 3

for column in lag_columns:
    data2[f'{column}_rolling_mean'] = data2[column].rolling(window=window_size).mean()
    data2[f'{column}_rolling_std'] = data2[column].rolling(window=window_size).std()

# Check the dataset with new rolling window statistics
print(data2.head())

# Save the corrected dataset to the same file
data2.to_csv('predictive-maintenance-dataset-imputed.csv', index=False)

# Assuming there is a 'timestamp' column
if 'timestamp' in data2.columns:
    data2['timestamp'] = pd.to_datetime(data2['timestamp'])
    data2['year'] = data2['timestamp'].dt.year
    data2['month'] = data2['timestamp'].dt.month
    data2['day'] = data2['timestamp'].dt.day
    data2['hour'] = data2['timestamp'].dt.hour
    data2['day_of_week'] = data2['timestamp'].dt.dayofweek

# Check the dataset with new datetime features
print(data2.head())

# Save the corrected dataset to the same file
data2.to_csv('predictive-maintenance-dataset-imputed.csv', index=False)

# List of original columns for creating additional features
lag_columns = ['revolutions', 'humidity', 'vibration', 'x1', 'x2', 'x3', 'x4', 'x5']

# Step 4.1: Create Exponential Moving Average (EMA) Features
span = 3

for column in lag_columns:
    data2[f'{column}_ema'] = data2[column].ewm(span=span, adjust=False).mean()

# Step 4.2: Create Difference Features
for column in lag_columns:
    data2[f'{column}_diff'] = data2[column].diff()

# Step 4.3: Create Cumulative Sum Features
for column in lag_columns:
    data2[f'{column}_cumsum'] = data2[column].cumsum()

# Step 4.4: Create Rate of Change Features
for column in lag_columns:
    data2[f'{column}_pct_change'] = data2[column].pct_change()

# Save the dataset with new features to the same file
data2.to_csv('predictive-maintenance-dataset-imputed.csv', index=False)

# Check the dataset with new features
print(data2.head())
###################################################################################
# Load the dataset
data2 = pd.read_csv('predictive-maintenance-dataset-imputed.csv')

# Define the target variable and features
target = 'equipment_failure'  # Replace with your actual target column name
features = data2.columns.difference([target, 'ID'])

# Separate numeric and categorical columns
numeric_cols = data2[features].select_dtypes(include=[np.number]).columns
categorical_cols = data2[features].select_dtypes(exclude=[np.number]).columns

# Check if there are any numeric columns to impute
if not numeric_cols.empty:
    # Impute missing values for numeric columns
    numeric_imputer = SimpleImputer(strategy='mean')
    data2[numeric_cols] = numeric_imputer.fit_transform(data2[numeric_cols])

# Check if there are any categorical columns to impute
if not categorical_cols.empty:
    # Impute missing values for categorical columns
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    data2[categorical_cols] = categorical_imputer.fit_transform(data2[categorical_cols])

# Check for missing values
missing_values_after_imputation = data2.isnull().sum()
print(f"Missing values after imputation:\n{missing_values_after_imputation[missing_values_after_imputation > 0]}")

# Calculate the correlation matrix
corr_matrix = data2[numeric_cols].corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Find features with correlation greater than 0.9
high_corr_features = [column for column in upper.columns if any(upper[column] > 0.9)]

# Drop highly correlated features
data2 = data2.drop(columns=high_corr_features)

# Save the dataset after correlation analysis
data2.to_csv('predictive-maintenance-dataset-imputed-corr.csv', index=False)

print("Correlation analysis done. Highly correlated features removed.")
print(f"Removed features: {high_corr_features}")

# Load the dataset after correlation analysis
data2 = pd.read_csv('predictive-maintenance-dataset-imputed-corr.csv')

# Define the target variable and features
features = data2.columns.difference([target, 'ID'])

# Separate features and target
X = data2[features]
y = data2[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = RandomForestClassifier(random_state=42)

# Step 8: Feature Scaling and Model Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', model)
])

# Fit the pipeline
pipeline.fit(X_train, y_train)

# Get feature importances
importances = model.feature_importances_

# Create a DataFrame for feature importances
feature_importances = pd.DataFrame({'feature': features, 'importance': importances})
feature_importances = feature_importances.sort_values(by='importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(10, 8))
sns.barplot(x='importance', y='feature', data=feature_importances.head(20))
plt.title('Top 20 Feature Importances')
plt.show()

# Select features based on importance
sfm = SelectFromModel(model, threshold=0.01)
sfm.fit(X_train, y_train)
selected_features = X.columns[(sfm.get_support())]

print("Selected features based on importance: ", selected_features)

# Reduce dataset to selected features
data2_selected = data2[selected_features.tolist() + [target]]

# Save the dataset after feature selection
data2_selected.to_csv('predictive-maintenance-dataset-imputed-selected.csv', index=False)

print("Feature importance analysis done. Dataset reduced to selected features.")

# Evaluate the model
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

# Cross-validation for more robust evaluation
cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
print(f"Cross-validation accuracy scores: {cv_scores}")
print(f"Mean cross-validation accuracy: {cv_scores.mean()}")

print("Model training and evaluation complete.")
###################################################################################
