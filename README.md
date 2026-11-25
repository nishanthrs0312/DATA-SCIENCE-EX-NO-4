# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.

STEP 2:Clean the Data Set using Data Cleaning Process.

STEP 3:Apply Feature Scaling for the feature in the data set.

STEP 4:Apply Feature Selection for the feature in the data set.

STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1

2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.

3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.

4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.

The feature selection techniques used are:

1.Filter Method

2.Wrapper Method

3.Embedded Method

# CODING AND OUTPUT:
```

# EXNO:4-DS
# Feature Scaling and Feature Selection

# Step 1: Import required libraries

import pandas as pd
import numpy as np
from google.colab import files

from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    MaxAbsScaler,
    RobustScaler
)

from sklearn.feature_selection import (
    SelectKBest,
    f_classif,
    RFE,
    SelectFromModel
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Step 2: Read the given dataset

uploaded = files.upload()

df = pd.read_csv('bmi.csv')

print("Original Dataset:")
print(df.head())

# Step 3: Data Cleaning

print("\nMissing Values:")
print(df.isnull().sum())

# Fill missing categorical values
df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])

# Fill missing numerical values
for col in df.select_dtypes(include=np.number).columns:
    df[col] = df[col].fillna(df[col].median())

# Remove duplicate records
df = df.drop_duplicates()

# Encode categorical data
df['Gender'] = df['Gender'].map({
    'Male': 1,
    'Female': 0
})

print("\nCleaned Dataset:")
print(df.head())

# Separate features and target

X = df.drop('Index', axis=1)
y = df['Index']

# Step 4: Feature Scaling

# 1. Standard Scaler
standard_scaler = StandardScaler()
standard_scaled = standard_scaler.fit_transform(X)

standard_df = pd.DataFrame(
    standard_scaled,
    columns=X.columns
)

# 2. MinMax Scaler
minmax_scaler = MinMaxScaler()
minmax_scaled = minmax_scaler.fit_transform(X)

minmax_df = pd.DataFrame(
    minmax_scaled,
    columns=X.columns
)

# 3. Maximum Absolute Scaler
maxabs_scaler = MaxAbsScaler()
maxabs_scaled = maxabs_scaler.fit_transform(X)

maxabs_df = pd.DataFrame(
    maxabs_scaled,
    columns=X.columns
)

# 4. Robust Scaler
robust_scaler = RobustScaler()
robust_scaled = robust_scaler.fit_transform(X)

robust_df = pd.DataFrame(
    robust_scaled,
    columns=X.columns
)

print("\nStandard Scaled Data:")
print(standard_df.head())

print("\nMinMax Scaled Data:")
print(minmax_df.head())

print("\nMaxAbs Scaled Data:")
print(maxabs_df.head())

print("\nRobust Scaled Data:")
print(robust_df.head())

# Step 5: Feature Selection

# 1. Filter Method - SelectKBest

filter_selector = SelectKBest(
    score_func=f_classif,
    k=2
)

filter_selector.fit(standard_df, y)

filter_features = X.columns[
    filter_selector.get_support()
]

print("\nFilter Method Selected Features:")
print(list(filter_features))

# 2. Wrapper Method - Recursive Feature Elimination (RFE)

model = LogisticRegression(
    max_iter=2000,
    random_state=42
)

rfe_selector = RFE(
    estimator=model,
    n_features_to_select=2
)

rfe_selector.fit(standard_df, y)

wrapper_features = X.columns[
    rfe_selector.get_support()
]

print("\nWrapper Method Selected Features:")
print(list(wrapper_features))

# 3. Embedded Method - Random Forest

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(standard_df, y)

embedded_selector = SelectFromModel(
    rf_model,
    threshold="median",
    prefit=True
)

embedded_features = X.columns[
    embedded_selector.get_support()
]

print("\nEmbedded Method Selected Features:")
print(list(embedded_features))

# Step 6: Create final dataset

final_df = pd.DataFrame()

final_df['Index'] = y.reset_index(drop=True)

# Add Standard Scaled Features
for col in filter_features:
    final_df['Standard_' + col] = standard_df[col].values

# Add selected feature information
final_df['Filter_Selected'] = [
    ', '.join(filter_features)
] * len(final_df)

final_df['Wrapper_Selected'] = [
    ', '.join(wrapper_features)
] * len(final_df)

final_df['Embedded_Selected'] = [
    ', '.join(embedded_features)
] * len(final_df)

print("\nFinal Dataset:")
print(final_df.head())

# Step 7: Save the processed dataset

final_df.to_csv(
    'bmi_feature_scaled_selected.csv',
    index=False
)

print("\nDataset saved successfully!")

# Download the output file
files.download('bmi_feature_scaled_selected.csv')
```

# Output:

<img width="551" height="710" alt="image" src="https://github.com/user-attachments/assets/7f3d8153-8413-49da-b0be-9cc305c9e34e" />
<img width="512" height="472" alt="image" src="https://github.com/user-attachments/assets/107d23ef-0ddb-438a-8f22-323333aac8be" />

                                                 
# RESULT:
The given BMI dataset was successfully cleaned, and Feature Scaling was performed using StandardScaler, MinMaxScaler, MaxAbsScaler, and RobustScaler.Feature Selection was successfully performed using Filter, Wrapper, and Embedded methods, and the final processed dataset was saved to a CSV file.

## Summary:
The program reads the BMI dataset and performs data cleaning by handling missing values, removing duplicates, and encoding categorical data. It applies four Feature Scaling techniques: StandardScaler, MinMaxScaler, MaxAbsScaler, and RobustScaler to transform the numerical features. Feature Selection is performed using three methods: Filter Method (SelectKBest), Wrapper Method (RFE), and Embedded Method (Random Forest). The selected features are identified based on their importance and statistical scores. The processed features are combined into a final dataset
