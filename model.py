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
 
def categorize_temperature(temp):
    """
    Categorize temperature values into 'Cold', 'Cool', or 'Ambient' zones.
   
    Args:
        temp (float): Temperature value in Celsius
       
    Returns:
        str: 'Cold', 'Cool', or 'Ambient' category
    """
    if pd.isna(temp):
        return "Unknown"
    elif temp < 0:
        return "Cold"
    elif temp <= 10:
        return "Cool"
    else:
        return "Ambient"
 
def preprocess_data(df):
    df["productivity_rate"] = df["totalprocessedqty"] / (df["timediffseconds"] / 3600)
    df["temp_hum_interaction"] = df["temperature"] * df["humidity"]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=["productivity_rate"], inplace=True)
    le = LabelEncoder()
    df["zone_encoded"] = le.fit_transform(df["Zone"])
    return df, le
 
def find_nearest_temperature(df, target_temp):
    """
    Find workers with the temperature closest to the target temperature.
   
    Args:
        df (pandas.DataFrame): Worker dataframe
        target_temp (float): Target temperature
       
    Returns:
        pandas.DataFrame: Filtered dataframe with nearest temperature
    """
    if "Room_Temp" not in df.columns:
        return df  # Return original dataframe if no temperature column
   
    # Get all available temperatures
    available_temps = df["Room_Temp"].dropna().unique()
    if len(available_temps) == 0:
        return df  # Return original if no temperatures available
   
    # Find closest temperature
    nearest_temp = available_temps[np.abs(available_temps - target_temp).argmin()]
   
    # Allow a small margin (0.5°C) to get more results
    margin = 0.5
    temp_filtered_df = df[(df["Room_Temp"] >= nearest_temp - margin) &
                          (df["Room_Temp"] <= nearest_temp + margin)]
   
    # If no results within margin, use exact nearest
    if len(temp_filtered_df) == 0:
        temp_filtered_df = df[df["Room_Temp"] == nearest_temp]
   
    # If still no results, return original dataframe
    if len(temp_filtered_df) == 0:
        return df
   
    return temp_filtered_df
 
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
 
def adjust_productivity_for_temperature(productivity, worker_temp, target_temp):
    """
    Adjust worker productivity based on temperature difference.
   
    Args:
        productivity (float): Original productivity rate
        worker_temp (float): Worker's historical temperature
        target_temp (float): Current target temperature
       
    Returns:
        float: Adjusted productivity
    """
    # Calculate temperature difference
    temp_diff = abs(worker_temp - target_temp)
   
    # Apply productivity penalty based on temperature difference
    # 1% reduction per degree difference
    adjustment_factor = max(0.7, 1 - (temp_diff * 0.01))  # Cap at 30% reduction
   
    return productivity * adjustment_factor
 
def assign_workers_to_zones(zone_df, worker_df, le, user_inputs):
    assigned_workers_global = set()  # Track workers already assigned to a zone
    results = []
   
    # Process each zone independently while ensuring global uniqueness of workers.
    for _, row in zone_df.iterrows():
        zone_id = row["zone"]
        quantity = user_inputs.get(f"{zone_id}_qty", 50)
        target_temp = user_inputs.get(f"{zone_id}_temp", None)  # Get target temperature if provided
        model_data = load_model("ambient_model.pkl" if zone_id == "Ambient" else "cold_cooler_model.pkl")
       
        features = model_data["features"]
        scaler = model_data["scaler"]
        worker_df["zone_encoded"] = le.transform(worker_df["Zone"])
        # Filter out workers with fewer than 5 records
        worker_df = worker_df.groupby("resource").filter(lambda x: len(x) >= 5)
 
        # Filter worker pool based on temperature if target_temp is provided
        filtered_worker_df = worker_df
        if target_temp is not None:
            filtered_worker_df = find_nearest_temperature(worker_df, target_temp)
       
        # Only consider workers not already assigned to another zone.
        unique_workers = [w for w in filtered_worker_df["resource"].unique() if w not in assigned_workers_global]
        worker_perf = {}
       
        for worker in unique_workers:
            w_data = filtered_worker_df[filtered_worker_df["resource"] == worker]
            worker_features = w_data[features].mean().to_dict()
           
            # Get the worker's historical temperature
            worker_temp = worker_features.get("Room_Temp")
           
            input_df = pd.DataFrame([worker_features], columns=features)
            input_scaled = scaler.transform(input_df)
            predicted_qty = model_data["model_qty"].predict(input_scaled)[0]
            predicted_time = model_data["model_time"].predict(input_scaled)[0]
            productivity = predicted_qty / (predicted_time / 3600) if predicted_time != 0 else 0
           
            # Adjust productivity if target temperature is different from worker's historical temperature
            if target_temp is not None and worker_temp is not None:
                productivity = adjust_productivity_for_temperature(productivity, worker_temp, target_temp)
           
            worker_perf[worker] = {
                "pred_qty": predicted_qty,
                "pred_time": predicted_time,
                "productivity": productivity,
                "temp": worker_temp
            }
       
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
                "Individual Productivity": f"{round(worker_perf[worker]['productivity'], 2)} items/hr",
                "Historical Temp": worker_perf[worker]["temp"],
                "Current Zone Temp": target_temp
            })
       
        results.append({
            "Zone": zone_id,
            "Processed_Quantity": quantity,
            "Zone Temperature": target_temp,
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
   
    # Updated user inputs to include temperature for each zone
    user_inputs = {
        "Ambient_qty": 500, "Ambient_temp": 23.0,  # Room temperature (~23°C)
        "Cold_qty": 800, "Cold_temp": 4.5,        # Cold storage (~4-5°C)
        "Cooler_qty": 1000, "Cooler_temp": 12.0   # Cooler (~12°C)
    }
   
    zone_input = pd.DataFrame({"zone": ["Ambient", "Cold", "Cooler"]})
    result_df = assign_workers_to_zones(zone_input, df, le, user_inputs)
    print(result_df)
 
