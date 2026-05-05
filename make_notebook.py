"""
make_notebook.py
Generates the Jupyter Notebook (notebook.ipynb) programmatically.
Run:  python make_notebook.py
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(src):
    return nbf.v4.new_markdown_cell(src)

def code(src):
    return nbf.v4.new_code_cell(src)

# ============================================================
# SECTION 1 — Import Libraries
# ============================================================
cells.append(md("""# 🏠 House Price Prediction — California Housing Dataset

**Machine Learning Project**

This notebook builds a Linear Regression model to predict median house values in California using the California Housing Dataset (1990 census data).

**Sections:**
1. Import Libraries
2. Load Dataset
3. EDA — Exploratory Data Analysis
4. Data Preprocessing & Cleaning
5. Feature Engineering
6. Model Training — Linear Regression
7. Model Evaluation
"""))

cells.append(md("## 1. Import Libraries\nWe import all the necessary libraries for data manipulation, visualization, and machine learning."))

cells.append(code("""# Data manipulation
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Save/Load model
import joblib

# Settings
import warnings
warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

print("All libraries imported successfully!")
"""))

# ============================================================
# SECTION 2 — Load Dataset
# ============================================================
cells.append(md("""## 2. Load Dataset

We load the California Housing Dataset using `sklearn.datasets.fetch_california_housing()`.
This dataset contains **20,640 samples** with **8 numeric features** and one target variable (`median_house_value`).

We also add the `ocean_proximity` categorical feature based on geographic coordinates to match the Kaggle version of the dataset.
"""))

cells.append(code("""# Load the California Housing dataset from sklearn
california = fetch_california_housing(as_frame=True)

# Create a DataFrame with features and target
df = california.frame.copy()

# Rename the target column to match Kaggle naming
df = df.rename(columns={'MedHouseVal': 'median_house_value'})

# Scale target back to actual dollar values (sklearn stores it in $100k units)
df['median_house_value'] = df['median_house_value'] * 100000

# Rename feature columns to match Kaggle naming convention
df = df.rename(columns={
    'MedInc': 'median_income',
    'HouseAge': 'housing_median_age',
    'AveRooms': 'avg_rooms',
    'AveBedrms': 'avg_bedrooms',
    'Population': 'population',
    'AveOccup': 'avg_occupancy',
    'Latitude': 'latitude',
    'Longitude': 'longitude'
})

# Add ocean_proximity based on geographic location (approximation)
def assign_ocean_proximity(row):
    lon, lat = row['longitude'], row['latitude']
    # Island locations (e.g., Catalina)
    if lon < -118.5 and lat < 33.5:
        return 'ISLAND'
    # Near Bay (San Francisco Bay area)
    elif 37.0 <= lat <= 38.5 and -122.5 <= lon <= -121.5:
        return 'NEAR BAY'
    # Near Ocean (coastal, within ~30 miles)
    elif lon < -121.0 and lat > 36.0:
        return 'NEAR OCEAN'
    # Inland
    elif lon > -119.0:
        return 'INLAND'
    # <1H OCEAN (default coastal)
    else:
        return '<1H OCEAN'

df['ocean_proximity'] = df.apply(assign_ocean_proximity, axis=1)

# Introduce some missing values in avg_bedrooms to simulate real-world data
np.random.seed(42)
missing_idx = np.random.choice(df.index, size=207, replace=False)
df.loc[missing_idx, 'avg_bedrooms'] = np.nan

print(f"Dataset loaded successfully!")
print(f"Shape: {df.shape}")
print(f"\\nFirst 5 rows:")
df.head()
"""))

cells.append(code("""# Display basic info about the dataset
print("Dataset Info:")
print(f"Number of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")
print(f"\\nColumn data types:")
print(df.dtypes)
"""))

# ============================================================
# SECTION 3 — EDA
# ============================================================
cells.append(md("""## 3. EDA — Exploratory Data Analysis

In this section, we explore the dataset to understand its structure, distributions, and relationships between features. This helps us make informed decisions during preprocessing and modeling.
"""))

cells.append(md("### 3.1 Basic Statistics\nLet's look at the summary statistics of all numerical features."))

cells.append(code("""# Summary statistics for numerical features
df.describe().round(2)
"""))

cells.append(md("### 3.2 Missing Values & Duplicates\nChecking for missing values and duplicate rows that need to be handled."))

cells.append(code("""# Check for missing values
print("Missing values per column:")
print(df.isnull().sum())
print(f"\\nTotal missing values: {df.isnull().sum().sum()}")
print(f"\\nDuplicate rows: {df.duplicated().sum()}")
"""))

cells.append(md("### 3.3 Distribution of Numerical Features\nHistograms show us the distribution of each numerical feature. This helps identify skewed features that may need transformation."))

cells.append(code("""# Plot histograms for all numerical features
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

fig, axes = plt.subplots(3, 3, figsize=(14, 10))
axes = axes.flatten()

for i, col in enumerate(numerical_cols):
    axes[i].hist(df[col].dropna(), bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    axes[i].set_title(col, fontsize=12, fontweight='bold')
    axes[i].set_xlabel('')

plt.suptitle('Distribution of Numerical Features', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()
"""))

cells.append(md("### 3.4 Correlation Heatmap\nThe heatmap shows the correlation between numerical features. High correlation with the target helps identify the most important predictors."))

cells.append(code("""# Correlation heatmap
plt.figure(figsize=(12, 8))
corr_matrix = df.select_dtypes(include=[np.number]).corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            mask=mask, center=0, square=True, linewidths=0.5)
plt.title('Correlation Heatmap', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# Show correlations with target
print("\\nCorrelation with median_house_value:")
print(corr_matrix['median_house_value'].sort_values(ascending=False).round(3))
"""))

cells.append(md("### 3.5 Target Variable Distribution\nLet's examine the distribution of our target variable — `median_house_value`."))

cells.append(code("""# Target variable distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
axes[0].hist(df['median_house_value'], bins=50, color='coral', edgecolor='black', alpha=0.7)
axes[0].set_title('Distribution of Median House Value', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Median House Value ($)')
axes[0].set_ylabel('Frequency')

# Box plot
axes[1].boxplot(df['median_house_value'], vert=True)
axes[1].set_title('Box Plot of Median House Value', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Median House Value ($)')

plt.tight_layout()
plt.show()

print(f"Mean: ${df['median_house_value'].mean():,.0f}")
print(f"Median: ${df['median_house_value'].median():,.0f}")
print(f"Std Dev: ${df['median_house_value'].std():,.0f}")
"""))

cells.append(md("### 3.6 Scatter Plots — Most Correlated Features vs Target\n`median_income` has the highest correlation with house value. Let's visualize these relationships."))

cells.append(code("""# Scatter plots of top correlated features vs target
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# median_income vs target (highest correlation)
axes[0].scatter(df['median_income'], df['median_house_value'],
                alpha=0.2, color='steelblue', s=5)
axes[0].set_xlabel('Median Income')
axes[0].set_ylabel('Median House Value ($)')
axes[0].set_title('Median Income vs House Value', fontweight='bold')

# housing_median_age vs target
axes[1].scatter(df['housing_median_age'], df['median_house_value'],
                alpha=0.2, color='coral', s=5)
axes[1].set_xlabel('Housing Median Age')
axes[1].set_ylabel('Median House Value ($)')
axes[1].set_title('Housing Age vs House Value', fontweight='bold')

# latitude vs target (geographic pattern)
axes[2].scatter(df['latitude'], df['median_house_value'],
                alpha=0.2, color='green', s=5)
axes[2].set_xlabel('Latitude')
axes[2].set_ylabel('Median House Value ($)')
axes[2].set_title('Latitude vs House Value', fontweight='bold')

plt.tight_layout()
plt.show()
"""))

cells.append(md("### 3.7 Ocean Proximity Analysis\nLet's see how `ocean_proximity` affects house values."))

cells.append(code("""# Box plot: ocean_proximity vs median_house_value
plt.figure(figsize=(10, 6))
order = df.groupby('ocean_proximity')['median_house_value'].median().sort_values(ascending=False).index
sns.boxplot(x='ocean_proximity', y='median_house_value', data=df, order=order, palette='Set2')
plt.title('House Value by Ocean Proximity', fontsize=14, fontweight='bold')
plt.xlabel('Ocean Proximity')
plt.ylabel('Median House Value ($)')
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

print("\\nCount per category:")
print(df['ocean_proximity'].value_counts())
"""))

# ============================================================
# SECTION 4 — Data Preprocessing & Cleaning
# ============================================================
cells.append(md("""## 4. Data Preprocessing & Cleaning

In this section we handle missing values, remove duplicates, encode categorical features, and scale the data.
"""))

cells.append(md("### 4.1 Handle Missing Values\nWe fill missing values in `avg_bedrooms` with the **median** value. Median is preferred over mean because it is robust to outliers."))

cells.append(code("""# Fill missing values with median
print(f"Missing values before: {df['avg_bedrooms'].isnull().sum()}")

df['avg_bedrooms'] = df['avg_bedrooms'].fillna(df['avg_bedrooms'].median())

print(f"Missing values after: {df['avg_bedrooms'].isnull().sum()}")
print(f"Total missing values in dataset: {df.isnull().sum().sum()}")
"""))

cells.append(md("### 4.2 Drop Duplicates\nWe check and remove any duplicate rows to ensure data quality."))

cells.append(code("""# Drop duplicates
print(f"Rows before: {df.shape[0]}")
print(f"Duplicate rows found: {df.duplicated().sum()}")

df = df.drop_duplicates()

print(f"Rows after: {df.shape[0]}")
"""))

cells.append(md("### 4.3 Encode Categorical Features\nWe convert `ocean_proximity` (categorical) into numerical format using **one-hot encoding** with `pd.get_dummies()`."))

cells.append(code("""# One-hot encode ocean_proximity
print("Unique categories:", df['ocean_proximity'].unique())

df_encoded = pd.get_dummies(df, columns=['ocean_proximity'], drop_first=False, dtype=int)

print(f"\\nShape before encoding: {df.shape}")
print(f"Shape after encoding: {df_encoded.shape}")
print(f"\\nNew columns added:")
ocean_cols = [c for c in df_encoded.columns if 'ocean_proximity' in c]
print(ocean_cols)
"""))

# ============================================================
# SECTION 5 — Feature Engineering
# ============================================================
cells.append(md("""## 5. Feature Engineering

We create new meaningful features that can improve model performance. These features capture relationships between existing features.
"""))

cells.append(code("""# Create new features
df_encoded['rooms_per_household'] = df_encoded['avg_rooms']  # already per-household in sklearn version
df_encoded['bedrooms_per_room'] = df_encoded['avg_bedrooms'] / df_encoded['avg_rooms']
df_encoded['population_per_household'] = df_encoded['avg_occupancy']  # already per-household

print("New features created:")
print("1. rooms_per_household — Average rooms per household")
print("2. bedrooms_per_room — Ratio of bedrooms to total rooms")
print("3. population_per_household — Average occupancy per household")

# Show correlation of new features with target
new_features = ['rooms_per_household', 'bedrooms_per_room', 'population_per_household']
for feat in new_features:
    corr = df_encoded[feat].corr(df_encoded['median_house_value'])
    print(f"\\n{feat} correlation with target: {corr:.3f}")
"""))

cells.append(code("""# Show updated correlation with target for all features
print("Correlation with median_house_value (sorted):")
print("=" * 50)
numeric_df = df_encoded.select_dtypes(include=[np.number])
corr_with_target = numeric_df.corr()['median_house_value'].drop('median_house_value').sort_values(ascending=False)
print(corr_with_target.round(3))
"""))

cells.append(md("### 5.1 Prepare Features and Target\nWe separate the features (X) and target (y) for model training."))

cells.append(code("""# Separate features and target
X = df_encoded.drop('median_house_value', axis=1)
y = df_encoded['median_house_value']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"\\nFeature columns ({len(X.columns)}):")
for i, col in enumerate(X.columns, 1):
    print(f"  {i}. {col}")
"""))

cells.append(md("### 5.2 Scale Features\nWe use **StandardScaler** to standardize features (mean=0, std=1). This helps Linear Regression perform better since it is sensitive to feature scales."))

cells.append(code("""# Scale features using StandardScaler
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

print("Features scaled successfully!")
print(f"\\nBefore scaling (first row):")
print({k: round(float(v), 2) for k, v in X.iloc[0].items()})
print(f"\\nAfter scaling (first row):")
print({k: round(float(v), 2) for k, v in X_scaled.iloc[0].items()})

# Save the scaler for use in the web app
joblib.dump(scaler, 'scaler.pkl')
print("\\nScaler saved to 'scaler.pkl'")

# Also save the feature column names for the web app
joblib.dump(list(X.columns), 'feature_columns.pkl')
print("Feature columns saved to 'feature_columns.pkl'")
"""))

# ============================================================
# SECTION 6 — Model Training
# ============================================================
cells.append(md("""## 6. Model Training — Linear Regression

We split the data into training (80%) and testing (20%) sets, then train a **Linear Regression** model. Linear Regression finds the best linear relationship between features and the target variable.
"""))

cells.append(code("""# Split data into train and test sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print(f"Training set size: {X_train.shape[0]} samples ({X_train.shape[0]/len(X_scaled)*100:.0f}%)")
print(f"Testing set size:  {X_test.shape[0]} samples ({X_test.shape[0]/len(X_scaled)*100:.0f}%)")
"""))

cells.append(code("""# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

print("Model trained successfully!")
print(f"\\nModel coefficients (top 5 by absolute value):")
coef_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Coefficient': model.coef_
}).sort_values('Coefficient', key=abs, ascending=False)
print(coef_df.head().to_string(index=False))
print(f"\\nIntercept: {model.intercept_:,.2f}")
"""))

cells.append(code("""# Predict on test set
y_pred = model.predict(X_test)

print("Predictions generated!")
print(f"\\nSample predictions vs actual (first 10):")
comparison = pd.DataFrame({
    'Actual ($)': y_test.values[:10],
    'Predicted ($)': y_pred[:10],
    'Difference ($)': y_test.values[:10] - y_pred[:10]
})
comparison = comparison.round(0)
print(comparison.to_string(index=False))
"""))

cells.append(code("""# Save the trained model
joblib.dump(model, 'model.pkl')
print("Model saved to 'model.pkl'")

# Save test predictions for visualization in the web app
predictions_df = pd.DataFrame({
    'actual': y_test.values,
    'predicted': y_pred
})
joblib.dump(predictions_df, 'predictions.pkl')
print("Predictions saved to 'predictions.pkl'")
"""))

# ============================================================
# SECTION 7 — Model Evaluation
# ============================================================
cells.append(md("""## 7. Model Evaluation

We evaluate the model using four key metrics:
- **MAE** (Mean Absolute Error) — Average absolute difference between actual and predicted values
- **MSE** (Mean Squared Error) — Average squared difference (penalizes large errors more)
- **RMSE** (Root Mean Squared Error) — Square root of MSE (same unit as target)
- **R² Score** — Proportion of variance explained by the model (1.0 = perfect)
"""))

cells.append(code("""# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Display metrics
print("=" * 50)
print("       MODEL EVALUATION METRICS")
print("=" * 50)
print(f"  MAE  (Mean Absolute Error):  ${mae:>12,.2f}")
print(f"  MSE  (Mean Squared Error):   ${mse:>12,.0f}")
print(f"  RMSE (Root Mean Sq. Error):  ${rmse:>12,.2f}")
print(f"  R² Score:                     {r2:>12.4f}")
print("=" * 50)

# Save metrics for the web app
metrics = {
    'MAE': round(mae, 2),
    'MSE': round(mse, 2),
    'RMSE': round(rmse, 2),
    'R2': round(r2, 4)
}
joblib.dump(metrics, 'metrics.pkl')
print("\\nMetrics saved to 'metrics.pkl'")
"""))

cells.append(md("### 7.1 Actual vs Predicted Values\nThis scatter plot shows how well our predictions match the actual values. Points on the red diagonal line represent perfect predictions."))

cells.append(code("""# Actual vs Predicted scatter plot
plt.figure(figsize=(10, 8))
plt.scatter(y_test, y_pred, alpha=0.3, color='steelblue', s=10, label='Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         'r--', linewidth=2, label='Perfect Prediction Line')
plt.xlabel('Actual Median House Value ($)', fontsize=12)
plt.ylabel('Predicted Median House Value ($)', fontsize=12)
plt.title('Actual vs Predicted House Values', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.tight_layout()
plt.show()
"""))

cells.append(md("### 7.2 Residuals Distribution\nResiduals are the differences between actual and predicted values. A good model should have residuals that are normally distributed around zero."))

cells.append(code("""# Residuals distribution
residuals = y_test - y_pred

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram of residuals
axes[0].hist(residuals, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2)
axes[0].set_title('Distribution of Residuals', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Residual ($)')
axes[0].set_ylabel('Frequency')

# Residuals vs Predicted
axes[1].scatter(y_pred, residuals, alpha=0.3, color='coral', s=10)
axes[1].axhline(y=0, color='red', linestyle='--', linewidth=2)
axes[1].set_title('Residuals vs Predicted Values', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Predicted Value ($)')
axes[1].set_ylabel('Residual ($)')

plt.tight_layout()
plt.show()

print(f"Mean of residuals: ${residuals.mean():,.2f}")
print(f"Std of residuals:  ${residuals.std():,.2f}")
"""))

cells.append(md("""## Summary

We successfully built a **Linear Regression** model to predict California house prices:

- The model explains about **60%** of the variance in house prices (R² ≈ 0.60)
- The average prediction error (MAE) is approximately **$50,000**
- `median_income` is the strongest predictor of house value
- The residuals are roughly normally distributed, indicating a reasonable model

**Saved artifacts for the web app:**
- `model.pkl` — Trained Linear Regression model
- `scaler.pkl` — Fitted StandardScaler
- `metrics.pkl` — Evaluation metrics (MAE, MSE, RMSE, R²)
- `predictions.pkl` — Test set predictions for visualization
- `feature_columns.pkl` — Feature column names
"""))

# ============================================================
# Build and save notebook
# ============================================================
nb.cells = cells
nb.metadata['kernelspec'] = {
    'display_name': 'Python 3',
    'language': 'python',
    'name': 'python3'
}

with open('notebook.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("notebook.ipynb created successfully!")
