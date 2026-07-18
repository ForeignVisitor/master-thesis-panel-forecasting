from pathlib import Path
import pandas as pd
import numpy as np

input_path = Path("data/processed/panel_data.csv")
output_path = Path("results/forecasts/naive_forecasts.csv")

df = pd.read_csv(input_path)
df = df.sort_values(["unit_id", "time"]).copy()

unique_times = sorted(df["time"].unique())
test_times = unique_times[-10:]

test = df[df["time"].isin(test_times)].copy()

test["forecast_naive"] = test["lag_1_target"]
test["actual"] = test["target"]
test["error_naive"] = test["actual"] - test["forecast_naive"]
test["squared_error_naive"] = test["error_naive"] ** 2

result = test[
    [
        "unit_id",
        "country",
        "time",
        "actual",
        "forecast_naive",
        "error_naive",
        "squared_error_naive",
    ]
].copy()

output_path.parent.mkdir(parents=True, exist_ok=True)
result.to_csv(output_path, index=False)

rmse = np.sqrt(result["squared_error_naive"].mean())
mae = result["error_naive"].abs().mean()

print("Saved:", output_path)
print("Test years:", min(test_times), "to", max(test_times))
print("Forecast rows:", len(result))
print("RMSE:", round(rmse, 4))
print("MAE:", round(mae, 4))