import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

def generate_logistics_data(samples=1000):
    np.random.seed(42)
    data = {
        "shipment_distance_km": np.random.uniform(50, 2500, samples),
        "warehouse_processing_hours": np.random.uniform(2, 48, samples),
        "carrier_reliability_score": np.random.uniform(0.5, 1.0, samples),
        "order_volume_units": np.random.randint(1, 500, samples),
        "weather_delay_factor": np.random.choice(
            [1.0, 1.2, 1.5, 2.0], samples, p=[0.6, 0.2, 0.1, 0.1]
        ),
    }
    df = pd.DataFrame(data)
    # Target: Delivery duration in hours
    df["delivery_duration_hours"] = (
        (df["shipment_distance_km"] / 60)
        + df["warehouse_processing_hours"]
        + (df["order_volume_units"] * 0.05)
    ) * df["weather_delay_factor"] + np.random.normal(0, 2, samples)
    return df

def train_delivery_predictor(df):
    X = df.drop(columns=["delivery_duration_hours"])
    y = df["delivery_duration_hours"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    print(f"Model Trained Successfully.")
    print(f"Mean Absolute Error (MAE): {mae:.2f} hours")
    print(f"R2 Score: {r2:.4f}")
    return model

if __name__ == "__main__":
    df = generate_logistics_data()
    model = train_delivery_predictor(df)
