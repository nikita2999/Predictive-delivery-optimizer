import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.title("🚚 Predictive Delivery Optimizer")

DATA_DIR = "data"

try:
    orders = pd.read_csv(os.path.join(DATA_DIR, "orders.csv"))
    delivery = pd.read_csv(os.path.join(DATA_DIR, "delivery_performance.csv"))
    routes = pd.read_csv(os.path.join(DATA_DIR, "routes_distance.csv"))
except:
    st.error("Place required CSV files in `/data` folder.")
    st.stop()

df = orders.merge(delivery, on="order_id", how="left").merge(routes, on="order_id", how="left")

df["delay_hours"] = (pd.to_datetime(df["actual_delivery_time"]) - pd.to_datetime(df["promised_delivery_time"])).dt.total_seconds() / 3600
df["delayed"] = (df["delay_hours"] > 0).astype(int)

st.subheader("Data Preview")
st.dataframe(df.head())

features = ["priority", "distance", "traffic_delays", "weather_impact"]
df = df.dropna(subset=features + ["delayed"])

X = df[features]
y = df["delayed"]

numeric = ["distance", "traffic_delays", "weather_impact"]
categorical = ["priority"]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric),
    ("cat", OneHotEncoder(), categorical)
])

model = Pipeline([("prep", preprocessor), ("model", RandomForestClassifier())])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

df["predicted_delay_risk"] = model.predict_proba(X)[:, 1]

st.subheader("Delay Risk Chart")
fig = px.histogram(df, x="predicted_delay_risk", title="Predicted Delay Risk Distribution")
st.plotly_chart(fig, use_container_width=True)

st.success("✅ Model Trained Successfully")
