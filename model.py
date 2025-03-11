import pandas as pd
from itertools import combinations
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import pickle

# Load dataset
def load_data(file_path="processed_warehouse_data.xlsx"):
    xls = pd.ExcelFile(file_path)
    return pd.read_excel(xls)

# Preprocess dataset
def preprocess_data(df):
    min_entries = 5
    resource_counts = df["resource"].value_counts()
    valid_resources = resource_counts[resource_counts >= min_entries].index
    df_filtered = df[df["resource"].isin(valid_resources)].copy()
    df_filtered["productivity_rate"] = df_filtered["totalprocessedqty"] / (df_filtered["timediffseconds"] / 3600)
    df_filtered["temp_hum_interaction"] = df_filtered["temperature"] * df_filtered["humidity"]
    df_filtered.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_filtered.dropna(subset=["productivity_rate"], inplace=True)
    df_filtered["hour"] = pd.to_numeric(df_filtered["hour"], errors='coerce')
    return df_filtered

# Train and save model
def train_model(df_filtered, model_filename="model.pkl"):
    features = ["temperature", "humidity", "estimatedpickdistancemeters", "estimatedstepstaken",
                "estimateditemweightkg", "totalweightprocessedkg", "productivity_rate", "temp_hum_interaction", "hour"]
    scaler = MinMaxScaler()
    df_filtered[features] = scaler.fit_transform(df_filtered[features])
    X = df_filtered[features]
    y = df_filtered["totalprocessedqty"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    xgb_model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, random_state=42)
    xgb_model.fit(X_train, y_train)
    
    # Save model and scaler
    with open(model_filename, "wb") as f:
        pickle.dump({"model": xgb_model, "scaler": scaler, "features": features}, f)
    
    print("Model and scaler saved successfully!")

# Load model
def load_model(model_filename="model.pkl"):
    with open(model_filename, "rb") as f:
        return pickle.load(f)

# Recommend worker combinations
def recommend_worker_combinations_for_all_regions(df, temp, humidity, combination_size=2, top_n=5):
    model_data = load_model()
    model, scaler, features = model_data["model"], model_data["scaler"], model_data["features"]
    region_assignments = {}
    available_workers = set(df["resource"].unique())
    regions_sorted = sorted(df["region"].unique(), key=lambda r: len(df[df["region"] == r]["resource"].unique()))
    unique_hours = sorted(df["hour"].dropna().unique())
    for region in regions_sorted:
        df_region = df[df["region"] == region]
        region_workers = set(df_region["resource"].unique())
        eligible_workers = list(available_workers.intersection(region_workers))
        # Ensure eligible_workers is a list of unique worker names
        eligible_workers = list(set(eligible_workers))  

        # If not enough workers, expand to region workers
        if len(eligible_workers) < combination_size:
            eligible_workers = list(set(region_workers))

        # Final check: If still insufficient, skip the region
        if len(eligible_workers) < combination_size:
            print(f"⚠️ Skipping region {region} due to insufficient workers.")
            continue


        best_combinations = []
        for hour in unique_hours:
            input_data = pd.DataFrame([[temp, humidity, 0, 0, 0, 0, 0, temp * humidity, hour]], columns=features)
            input_data_scaled = pd.DataFrame(scaler.transform(input_data), columns=features)
            worker_combinations = list(combinations(eligible_workers, combination_size))
            region_scores = []
            for combo in worker_combinations:
                avg_score = 0
                for worker in combo:
                    worker_data = df_region[df_region["resource"] == worker]
                    if worker_data.empty:
                        continue
                    worker_features = worker_data[features].iloc[0].copy()
                    worker_features["temperature"] = input_data["temperature"].iloc[0]
                    worker_features["humidity"] = input_data["humidity"].iloc[0]
                    worker_features["hour"] = input_data["hour"].iloc[0]
                    worker_features["temp_hum_interaction"] = input_data["temp_hum_interaction"].iloc[0]
                    worker_features_df = pd.DataFrame([worker_features.values], columns=features)
                    worker_features_scaled = pd.DataFrame(scaler.transform(worker_features_df), columns=features)
                    avg_score += model.predict(worker_features_scaled)[0]
                avg_score /= combination_size
                region_scores.append((combo, avg_score))
            best_combinations.extend(sorted(region_scores, key=lambda x: x[1], reverse=True)[:top_n])
        if best_combinations:
            best_combinations.sort(key=lambda x: x[1], reverse=True)
            assigned_workers = set()
            final_combinations = []
            for combo, score in best_combinations:
                if not assigned_workers.intersection(set(combo)):
                    final_combinations.append((combo, score))
                    assigned_workers.update(combo)
            region_assignments[region] = final_combinations
            available_workers -= assigned_workers
    return region_assignments

if __name__ == "__main__":
    df = load_data()
    df_filtered = preprocess_data(df)
    train_model(df_filtered)
    temp = 22.0  # Example temperature
    humidity = 60  # Example humidity
    recommendations = recommend_worker_combinations_for_all_regions(df_filtered, temp, humidity)
    print("Generated Worker Recommendations:")
    print(recommendations)