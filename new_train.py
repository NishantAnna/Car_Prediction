import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import StackingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import joblib
import warnings

warnings.filterwarnings("ignore")

# --- Load Data ---
df = pd.read_csv("Final_Used_Cars.csv")
df = df.dropna(subset=["price"])  # Ensure target exists

# --- Feature Engineering ---
current_year = datetime.now().year
df["car_age"] = current_year - df["year"]
df["car_age"] = df["car_age"].clip(lower=0)

# Interaction: miles per year
df["age_miles_ratio"] = df["miles"] / (df["car_age"] + 1)
df["age_miles_ratio"].replace([np.inf, -np.inf], np.nan, inplace=True)
df["age_miles_ratio"].fillna(0, inplace=True)

# Handle infinite/NaN values
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(0, inplace=True)

# Mileage buckets
df["mileage_bucket"] = pd.cut(
    df["miles"],
    bins=[-1, 20000, 60000, 100000, np.inf],
    labels=["Low", "Medium", "High", "Very_High"]
)

# Region clusters
coords = df[["latitude", "longitude"]].fillna(0).to_numpy()
kmeans = KMeans(n_clusters=10, n_init=10, random_state=42)
df["region_cluster"] = kmeans.fit_predict(coords).astype(str)

# --- Prepare X/y ---
X = df.drop(columns=["price"])
y = df["price"]

# Save raw training column names for inference alignment
joblib.dump(X.columns.tolist(), "raw_training_columns.pkl")
joblib.dump(kmeans, "kmeans_region.pkl")

cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = X.select_dtypes(exclude=["object", "category"]).columns.tolist()

# --- Preprocessor ---
preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

# --- Base Models ---
xgb_model = XGBRegressor(objective="reg:squarederror", random_state=42, n_estimators=500,
                         learning_rate=0.05, max_depth=6, subsample=0.85, colsample_bytree=0.85,
                         reg_alpha=0.1, reg_lambda=1.5, min_child_weight=3, gamma=0.1, verbosity=0, n_jobs=-1)

lgb_model = LGBMRegressor(objective="regression", random_state=42, n_estimators=500,
                          learning_rate=0.05, max_depth=6, subsample=0.85, colsample_bytree=0.85,
                          reg_alpha=0.1, reg_lambda=1.5, min_child_samples=20, verbose=-1)

# --- Final Meta Model ---
final_model = XGBRegressor(objective="reg:squarederror", random_state=42, n_estimators=300,
                           learning_rate=0.05, max_depth=4, subsample=0.85, colsample_bytree=0.85,
                           reg_alpha=0.1, reg_lambda=1.5, gamma=0.1, verbosity=0, n_jobs=-1)

# --- Stacking Regressor ---
stack_model = StackingRegressor(
    estimators=[("xgb", xgb_model), ("lgb", lgb_model)],
    final_estimator=final_model,
    n_jobs=-1,
    passthrough=False
)

# --- Pipeline ---
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("stack", stack_model)
])

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- Train the Pipeline ---
pipeline.fit(X_train, y_train)

# --- Evaluate ---
y_pred = pipeline.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MAE: {mae:.2f}")
print(f"R²: {r2:.4f}")

# --- Save Artifacts ---
joblib.dump(pipeline, "car_price_model.pkl")
joblib.dump(mae, "test_mae.pkl")

print("Saved: car_price_model.pkl, kmeans_region.pkl, raw_training_columns.pkl, test_mae.pkl")

