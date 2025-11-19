import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from function_preprocessing_motorbike import *

# load function
df = preprocess_motobike_data("data_motobikes.xlsx")

# Lọc outliers theo segment
def remove_outliers_by_segment(data, col='price', lower_q=0.05, upper_q=0.95):
    cleaned_df = []
    for seg, group in data.groupby('segment'):
        q_low = group[col].quantile(lower_q)
        q_high = group[col].quantile(upper_q)
        filtered = group[(group[col] >= q_low) & (group[col] <= q_high)]
        cleaned_df.append(filtered)
        # print(f"{seg:<25} | Before: {len(group):4} | After: {len(filtered):4} | Kept: {len(filtered)/len(group)*100:.1f}%")
    return pd.concat(cleaned_df, ignore_index=True)

df = remove_outliers_by_segment(df, 'price', lower_q=0.1, upper_q=0.9)


cat_cols = ['segment','bike_type','origin','engine_capacity']
num_cols = ['age','mileage_km','min_price','max_price','brand_meanprice']

# Prepare features and target
X = df[cat_cols + num_cols]
y = df['log_price']

df_final = pd.concat([X,y], axis=1)

# Preprocessing
num_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, num_cols),
    ('cat', cat_transformer, cat_cols)
])

model = Pipeline(steps=[
    ('preproc', preprocessor),
    ('reg', RandomForestRegressor(n_estimators=100,
    max_depth=12,
    random_state=42,
    n_jobs=-1))
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)

# đánh giá model
print("Check the train/test dataset:")
print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)

y_train_real = np.expm1(y_train)
y_pred_train_real = np.expm1(model.predict(X_train))

y_test_real = np.expm1(y_test)
y_pred_test_real = np.expm1(model.predict(X_test))

mae_train_real = mean_absolute_error(y_train_real, y_pred_train_real)
rmse_train_real = root_mean_squared_error(y_train_real, y_pred_train_real)
r2_train_real = r2_score(y_train_real, y_pred_train_real)

mae_test_real = mean_absolute_error(y_test_real, y_pred_test_real)
rmse_test_real = root_mean_squared_error(y_test_real, y_pred_test_real)
r2_test_real = r2_score(y_test_real, y_pred_test_real)

print(f"MAE train (VND): {mae_train_real:,.0f}, RMSE train (VND): {rmse_train_real:,.0f}, R2 train: {r2_train_real:.3f}")
print(f"MAE test (VND): {mae_test_real:,.0f}, RMSE test (VND): {rmse_test_real:,.0f}, R2 test: {r2_test_real:.3f}")

# Save model
import pickle
with open('motobike_price_prediction_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Model saved to motobike_price_prediction_model.pkl")