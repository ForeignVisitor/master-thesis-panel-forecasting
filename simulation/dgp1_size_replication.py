import numpy as np
import pandas as pd
from pathlib import Path

Path("results").mkdir(exist_ok=True)
Path("docs").mkdir(exist_ok=True)

np.random.seed(42)

MC = 1000
BURN = 500
phis = [0.5, 0.8]
Ns = [50, 100]
Ts = [50, 100, 500, 1000]

mu1 = 0.0
mu2 = 0.0
sigma_eps = 1.0
z_975 = 1.959963984540054

def bartlett_se(R):
    T = len(R)
    J = max(1, int(T ** (1/3)))
    x = R - R.mean()
    gamma0 = np.mean(x * x)
    lrv = gamma0

    for j in range(1, J + 1):
        weight = 1.0 - j / (J + 1)
        if weight <= 0:
            continue
        gammaj = np.mean(x[j:] * x[:-j])
        lrv += 2.0 * weight * gammaj

    if lrv <= 0 or np.isnan(lrv):
        return np.nan

    return np.sqrt(lrv)

rows = []

for phi in phis:
    for N in Ns:
        for T in Ts:
            for sim in range(1, MC + 1):
                TT = T + BURN

                e1 = np.zeros((TT, N))
                e2 = np.zeros((TT, N))

                eps1 = np.random.normal(0, sigma_eps, size=(TT, N))
                eps2 = np.random.normal(0, sigma_eps, size=(TT, N))

                for t in range(1, TT):
                    e1[t, :] = mu1 * (1 - phi) + phi * e1[t - 1, :] + eps1[t, :]
                    e2[t, :] = mu2 * (1 - phi) + phi * e2[t - 1, :] + eps2[t, :]

                e1 = e1[BURN:, :]
                e2 = e2[BURN:, :]

                loss_diff = e1**2 - e2**2
                delta_L_t = loss_diff.mean(axis=1)
                R_t = np.sqrt(N) * delta_L_t

                sigma_hat = bartlett_se(R_t)

                if np.isnan(sigma_hat) or sigma_hat == 0:
                    jdm = np.nan
                    reject_5pct = np.nan
                else:
                    jdm = np.sqrt(T) * R_t.mean() / sigma_hat
                    reject_5pct = int(abs(jdm) > z_975)

                rows.append({
                    "phi": phi,
                    "N": N,
                    "T": T,
                    "sim": sim,
                    "jdm": jdm,
                    "reject_5pct": reject_5pct,
                    "mean_loss_diff": float(loss_diff.mean()),
                    "sigma_hat": float(sigma_hat)
                })

run_df = pd.DataFrame(rows)
run_df.to_csv("results/dgp1_runlevel_v2.csv", index=False)

summary_df = (
    run_df.groupby(["phi", "N", "T"], as_index=False)
    .agg(
        rejection_rate=("reject_5pct", "mean"),
        mean_jdm=("jdm", "mean"),
        std_jdm=("jdm", "std"),
        mean_loss_diff=("mean_loss_diff", "mean"),
        mean_sigma_hat=("sigma_hat", "mean")
    )
    .sort_values(["phi", "N", "T"])
)

summary_df.to_csv("results/dgp1_summary_v2.csv", index=False)

print("Saved results/dgp1_runlevel_v2.csv")
print("Saved results/dgp1_summary_v2.csv")
print(summary_df)