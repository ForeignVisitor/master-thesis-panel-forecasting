import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

N = 50
T = 100
rho_true = 0.6

rows = []

alpha = np.random.normal(0, 1, N)

for i in range(N):
    y = np.zeros(T)
    eps = np.random.normal(0, 1, T)
    y[0] = alpha[i] + eps[0]

    for t in range(1, T):
        y[t] = alpha[i] + rho_true * y[t-1] + eps[t]

    for t in range(1, T):
        forecast_ar1 = alpha[i] + rho_true * y[t-1]
        forecast_naive = y[t-1]

        err_ar1 = y[t] - forecast_ar1
        err_naive = y[t] - forecast_naive

        rows.append({
            "unit": i + 1,
            "time": t + 1,
            "alpha_i": alpha[i],
            "y": y[t],
            "y_lag1": y[t-1],
            "forecast_ar1": forecast_ar1,
            "forecast_naive": forecast_naive,
            "error_ar1": err_ar1,
            "error_naive": err_naive,
            "sqerr_ar1": err_ar1 ** 2,
            "sqerr_naive": err_naive ** 2,
            "loss_diff_ar1_minus_naive": err_ar1 ** 2 - err_naive ** 2
        })

df = pd.DataFrame(rows)

Path("results").mkdir(exist_ok=True)

df.to_csv("results/simulated_panel.csv", index=False)

summary = pd.DataFrame({
    "model": ["ar1", "naive"],
    "mse": [df["sqerr_ar1"].mean(), df["sqerr_naive"].mean()],
    "mae": [df["error_ar1"].abs().mean(), df["error_naive"].abs().mean()]
})

summary.to_csv("results/simulation_summary.csv", index=False)

unit_summary = df.groupby("unit")[["sqerr_ar1", "sqerr_naive"]].mean().reset_index()
unit_summary.to_csv("results/simulation_unit_summary.csv", index=False)

time_summary = df.groupby("time")[["sqerr_ar1", "sqerr_naive", "loss_diff_ar1_minus_naive"]].mean().reset_index()
time_summary.to_csv("results/simulation_time_summary.csv", index=False)

print(summary)
print("Files saved in results/")