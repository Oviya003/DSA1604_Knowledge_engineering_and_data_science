
import pandas as pd

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv("bike_sharing.csv")

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["station_id", "date"])

# Time-based features
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["weekday"] = df["date"].dt.dayofweek

# Lag feature
df["lag_rentals"] = df.groupby(
    "station_id"
)["rentals"].shift(1)

# Rolling mean
df["rolling_mean"] = df.groupby(
    "station_id"
)["rentals"].transform(
    lambda x: x.shift(1).rolling(7).mean()
)

df = df.dropna()

# Input and output
features = [
    "station_id", "holiday", "temperature",
    "humidity", "wind_speed", "month",
    "day", "weekday", "lag_rentals",
    "rolling_mean"
]

X = df[features]
y = df["rentals"]

# Time-ordered split
dates = sorted(df["date"].unique())
split_date = dates[int(len(dates) * 0.8)]

train = df[df["date"] < split_date]
test = df[df["date"] >= split_date]

X_train = train[features]
X_test = test[features]

y_train = train["rentals"]
y_test = test["rentals"]

# Decision Tree
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)

# Random Forest
rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
rf.fit(X_train, y_train)

# Gradient Boosting
gb = GradientBoostingRegressor(
    n_estimators=100,
    random_state=42
)
gb.fit(X_train, y_train)

# Compare models
models = {
    "Decision Tree": dt,
    "Random Forest": rf,
    "Gradient Boosting": gb
}

for name, model in models.items():
    pred = model.predict(X_test)

    print("\n", name)
    print("MAE:", mean_absolute_error(y_test, pred))
    print("R2 Score:", r2_score(y_test, pred))
