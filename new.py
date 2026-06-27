import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import pickle
import warnings
warnings.filterwarnings('ignore')

# Set plot style
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 100


df = pd.read_csv("data/yield_df.csv")

print("First 5 Rows")
print(df.head())

print("\nShape:", df.shape)

print("\nColumns:")
print(df.columns)


if 'Unnamed: 0' in df.columns:
    df.drop('Unnamed: 0', axis=1, inplace=True)

print("\nMissing Values")
print(df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())

df.drop_duplicates(inplace=True)

print("\nShape After Removing Duplicates:", df.shape)


plt.figure(figsize=(10,5))
sns.histplot(df['hg/ha_yield'], bins=30, kde=True)
plt.title("Crop Yield Distribution")
plt.savefig('results/yield_distribution.png', dpi=300, bbox_inches='tight')
plt.show()


plt.figure(figsize=(10,5))
sns.scatterplot(
    x='average_rain_fall_mm_per_year',
    y='hg/ha_yield',
    data=df
)
plt.title("Rainfall vs Yield")
plt.savefig('results/rainfall_vs_yield.png', dpi=300, bbox_inches='tight')
plt.show()


plt.figure(figsize=(10,5))
sns.scatterplot(
    x='avg_temp',
    y='hg/ha_yield',
    data=df
)
plt.title("Temperature vs Yield")
plt.savefig('results/temperature_vs_yield.png', dpi=300, bbox_inches='tight')
plt.show()


plt.figure(figsize=(10,5))
sns.scatterplot(
    x='pesticides_tonnes',
    y='hg/ha_yield',
    data=df
)
plt.title("Pesticides vs Yield")
plt.savefig('results/pesticides_vs_yield.png', dpi=300, bbox_inches='tight')
plt.show()


plt.figure(figsize=(12,6))
df['Item'].value_counts().head(10).plot(kind='bar')
plt.title("Top 10 Crops")
plt.savefig('results/top_10_crops.png', dpi=300, bbox_inches='tight')
plt.show()


le_area = LabelEncoder()
le_item = LabelEncoder()

df['Area'] = le_area.fit_transform(df['Area'])
df['Item'] = le_item.fit_transform(df['Item'])


plt.figure(figsize=(12,8))

sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap='coolwarm'
)

plt.title("Correlation Heatmap")
plt.savefig('results/correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()


X = df.drop('hg/ha_yield', axis=1)

y = df['hg/ha_yield']


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


lr = LinearRegression()

lr.fit(X_train_scaled, y_train)

pred_lr = lr.predict(X_test_scaled)

mae_lr = mean_absolute_error(y_test, pred_lr)
mse_lr = mean_squared_error(y_test, pred_lr)
rmse_lr = np.sqrt(mse_lr)
r2_lr = r2_score(y_test, pred_lr)

print("\n==============================")
print("LINEAR REGRESSION RESULTS")
print("==============================")

print("MAE :", mae_lr)
print("MSE :", mse_lr)
print("RMSE:", rmse_lr)
print("R2  :", r2_lr)

rf = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

pred_rf = rf.predict(X_test)

mae_rf = mean_absolute_error(y_test, pred_rf)
mse_rf = mean_squared_error(y_test, pred_rf)
rmse_rf = np.sqrt(mse_rf)
r2_rf = r2_score(y_test, pred_rf)

print("\n==============================")
print("RANDOM FOREST RESULTS")
print("==============================")

print("MAE :", mae_rf)
print("MSE :", mse_rf)
print("RMSE:", rmse_rf)
print("R2  :", r2_rf)

xgb = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=8,
    random_state=42
)

xgb.fit(X_train, y_train)

pred_xgb = xgb.predict(X_test)

mae_xgb = mean_absolute_error(y_test, pred_xgb)
mse_xgb = mean_squared_error(y_test, pred_xgb)
rmse_xgb = np.sqrt(mse_xgb)
r2_xgb = r2_score(y_test, pred_xgb)

print("\n==============================")
print("XGBOOST RESULTS")
print("==============================")

print("MAE :", mae_xgb)
print("MSE :", mse_xgb)
print("RMSE:", rmse_xgb)
print("R2  :", r2_xgb)


results = pd.DataFrame({

    'Model': [
        'Linear Regression',
        'Random Forest',
        'XGBoost'
    ],

    'MAE': [
        mae_lr,
        mae_rf,
        mae_xgb
    ],

    'MSE': [
        mse_lr,
        mse_rf,
        mse_xgb
    ],

    'RMSE': [
        rmse_lr,
        rmse_rf,
        rmse_xgb
    ],

    'R2 Score': [
        r2_lr,
        r2_rf,
        r2_xgb
    ]
})

results = results.sort_values(
    by='R2 Score',
    ascending=False
)

print("\nMODEL COMPARISON")
print(results)

# Save results
results.to_csv('results/model_comparison.csv', index=False)


plt.figure(figsize=(8,5))

sns.barplot(
    x='Model',
    y='R2 Score',
    data=results
)

plt.title("Model Comparison")
plt.savefig('results/model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()


feature_importance = pd.DataFrame({

    'Feature': X.columns,

    'Importance': rf.feature_importances_

})

feature_importance = feature_importance.sort_values(
    by='Importance',
    ascending=False
)

print("\nFeature Importance")
print(feature_importance)

# Save feature importance
feature_importance.to_csv('results/feature_importance.csv', index=False)


plt.figure(figsize=(10,6))

sns.barplot(
    x='Importance',
    y='Feature',
    data=feature_importance
)

plt.title("Feature Importance")
plt.savefig('results/feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()


plt.figure(figsize=(8,8))

plt.scatter(
    y_test,
    pred_rf,
    alpha=0.5
)

plt.xlabel("Actual Yield")

plt.ylabel("Predicted Yield")

plt.title("Actual vs Predicted Yield (Random Forest)")

plt.savefig('results/actual_vs_predicted.png', dpi=300, bbox_inches='tight')

plt.show()

best_model = results.iloc[0]

print("\n====================================")
print("BEST MODEL")
print("====================================")
print(best_model)

# Save models
print("\nSaving models...")
with open('model/random_forest_model.pkl', 'wb') as f:
    pickle.dump(rf, f)

with open('model/xgboost_model.pkl', 'wb') as f:
    pickle.dump(xgb, f)

with open('model/linear_regression_model.pkl', 'wb') as f:
    pickle.dump(lr, f)

with open('model/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('model/label_encoder_area.pkl', 'wb') as f:
    pickle.dump(le_area, f)

with open('model/label_encoder_item.pkl', 'wb') as f:
    pickle.dump(le_item, f)

print("All models and preprocessing objects saved successfully!")