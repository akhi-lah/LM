import pandas as pd
import numpy as np
import pickle
import math
import os
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

MODEL_FILE = 'warehouse_xgb_model.pkl'

def load_data(file_path="updated_warehouse_data_fixed.xlsx"):
    return pd.read_excel(file_path)
 
def preprocess_data(df):
    df["productivity_rate"] = df["totalprocessedqty"] / (df["timediffseconds"] / 3600)
    df["temp_hum_interaction"] = df["temperature"] * df["humidity"]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=["productivity_rate"], inplace=True)
    le = LabelEncoder()
    df["zone_encoded"] = le.fit_transform(df["Zone"])
    return df, le
 
def train_model(df, model_filename, zone_type):
    features = ["Room_Temp", "estimatedpickdistancemeters", "estimatedstepstaken",
                "estimateditemweightkg", "totalweightprocessedkg", "productivity_rate", "zone_encoded"]
    if zone_type == "ambient":
        features.extend(["humidity", "temp_hum_interaction"])
   
    scaler = MinMaxScaler()
    X = df[features]
    y = df[["totalprocessedqty", "timediffseconds"]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model_qty = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, random_state=42)
    model_time = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, random_state=42)

    model_qty.fit(X_train_scaled, y_train["totalprocessedqty"])
    model_time.fit(X_train_scaled, y_train["timediffseconds"])

    # Evaluate models on test set
    y_pred_qty = model_qty.predict(X_test_scaled)
    y_pred_time = model_time.predict(X_test_scaled)

    r2_qty = r2_score(y_test["totalprocessedqty"], y_pred_qty)
    rmse_qty = np.sqrt(mean_squared_error(y_test["totalprocessedqty"], y_pred_qty))

    r2_time = r2_score(y_test["timediffseconds"], y_pred_time)
    rmse_time = np.sqrt(mean_squared_error(y_test["timediffseconds"], y_pred_time))

    print("Model for Processed Quantity: R2 =", r2_qty, "RMSE =", rmse_qty)
    print("Model for Time: R2 =", r2_time, "RMSE =", rmse_time)

    with open(model_filename, "wb") as f:
        pickle.dump({"model_qty": model_qty, "model_time": model_time, "scaler": scaler, "features": features}, f)

def load_model(model_filename):
    with open(model_filename, "rb") as f:
        return pickle.load(f)
 
# Remove the cap: number of workers is now calculated as quantity divided by 300 (rounded up)
def get_worker_count(quantity):
    return math.ceil(quantity / 300)
 
def assign_workers_to_zones(zone_df, worker_df, le, user_inputs):
    assigned_workers_global = set()  # Track workers already assigned to a zone
    results = []
   
    # Process each zone independently while ensuring global uniqueness of workers.
    for _, row in zone_df.iterrows():
        zone_id = row["zone"]
        quantity = user_inputs.get(f"{zone_id}_qty", 50)
        model_data = load_model("ambient_model.pkl" if zone_id == "Ambient" else "cold_cooler_model.pkl")
        
        features = model_data["features"]
        scaler = model_data["scaler"]
        worker_df["zone_encoded"] = le.transform(worker_df["Zone"])
        
        # Only consider workers not already assigned to another zone.
        unique_workers = [w for w in worker_df["resource"].unique() if w not in assigned_workers_global]
        worker_perf = {}
       
        for worker in unique_workers:
            w_data = worker_df[worker_df["resource"] == worker]
            worker_features = w_data[features].mean().to_dict()
            input_df = pd.DataFrame([worker_features], columns=features)
            input_scaled = scaler.transform(input_df)
            predicted_qty = model_data["model_qty"].predict(input_scaled)[0]
            predicted_time = model_data["model_time"].predict(input_scaled)[0]
            productivity = predicted_qty / (predicted_time / 3600) if predicted_time != 0 else 0
            worker_perf[worker] = {"pred_qty": predicted_qty, "pred_time": predicted_time, "productivity": productivity}
       
        num_workers = get_worker_count(quantity)
        assigned_workers = sorted(worker_perf.keys(), key=lambda w: worker_perf[w]["productivity"], reverse=True)[:num_workers]
        # Mark these workers as assigned globally.
        assigned_workers_global.update(assigned_workers)
       
        total_team_productivity = sum(worker_perf[w]["productivity"] for w in assigned_workers)
        team_etc_minutes = (quantity / total_team_productivity) * 60 if total_team_productivity > 0 else 0
        # Ensure a minimum team time of 2 minutes
        if team_etc_minutes < 2:
            team_etc_minutes = 2
        
        worker_details = []
        for worker in assigned_workers:
            ind_etc_minutes = (quantity / worker_perf[worker]["productivity"]) * 60 if worker_perf[worker]["productivity"] > 0 else 0
            # Ensure a minimum individual time of 2 minutes
            if ind_etc_minutes < 2:
                ind_etc_minutes = 2
            worker_details.append({
                "worker_id": worker,
                "Individual ETC": f"{int(ind_etc_minutes // 60)}hrs {int(ind_etc_minutes % 60)}mins",
                "Individual Productivity": f"{round(worker_perf[worker]['productivity'], 2)} items/hr"
            })
       
        results.append({
            "Zone": zone_id,
            "Processed_Quantity": quantity,
            "Team Size": len(assigned_workers),
            "EstimatedTimeToPickTheQuantity": f"{int(team_etc_minutes // 60)}hrs {int(team_etc_minutes % 60)}mins",
            "Team Productivity": round(total_team_productivity, 2),
            "WorkerDetails": worker_details
        })
   
    return pd.DataFrame(results).sort_values("Zone")
 
if __name__ == "__main__":
    df = load_data()
    df, le = preprocess_data(df)
    train_model(df, "ambient_model.pkl", "ambient")
    train_model(df, "cold_cooler_model.pkl", "cold_cooler")
    user_inputs = {"Ambient_qty": 500, "Cold_qty": 800, "Cooler_qty": 1000}
    zone_input = pd.DataFrame({"zone": ["Ambient", "Cold", "Cooler"]})
    result_df = assign_workers_to_zones(zone_input, df, le, user_inputs)
    print(result_df)
