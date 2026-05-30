import numpy as np
import pandas as pd
from pathlib import Path

Path("results").mkdir(exist_ok=True)
Path("docs").mkdir(exist_ok=True)

np.random.seed(42)

MC = 500
scenarios = [
    {"N": 50, "T": 50},
    {"N": 50, "T": 100},
    {"N": 100, "T": 50},
    {"N": 100, "T": 100},
]

rho1 = 0.6
rho2 = 0.6
sigma1 = 1.0
sigma2 = 1.0
y_rho = 0.5

scenario_results = []
all_runs = []

for sc in scenarios:
    N = sc["N"]
    T = sc["T"]

    for sim in range(MC):
        unit_effects = np.random.normal(0, 1, N)
        losses1 = []
        losses2 = []
        loss_diffs = []

        for i in range(N):
            e1 = np.zeros(T)
            e2 = np.zeros(T)
            y = np.zeros(T)

            shock1 = np.random.normal(0, sigma1, T)
            shock2 = np.random.normal(0, sigma2, T)
            yshock = np.random.normal(0, 1, T)

            y[0] = unit_effects[i] + yshock[0]

            for t in range(1, T):
                e1[t] = rho1 * e1[t-1] + shock1[t]
                e2[t] = rho2 * e2[t-1] + shock2[t]
                y[t] = unit_effects[i] + y_rho * y[t-1] + yshock[t]

            for t in range(1, T):
                forecast1 = y[t] - e1[t]
                forecast2 = y[t] - e2[t]

                loss1 = (y[t] - forecast1) ** 2
                loss2 = (y[t] - forecast2) ** 2
                ld = loss1 - loss2

                losses1.append(loss1)
                losses2.append(loss2)
                loss_diffs.append(ld)

        run_row = {
            "scenario": f"N{N}_T{T}",
            "N": N,
            "T": T,
            "sim": sim + 1,
            "avg_loss1": float(np.mean(losses1)),
            "avg_loss2": float(np.mean(losses2)),
            "avg_loss_diff": float(np.mean(loss_diffs)),
            "model1_better": int(np.mean(losses1) < np.mean(losses2)),
        }
        all_runs.append(run_row)

run_df = pd.DataFrame(all_runs)
run_df.to_csv("results/replication_run_summary.csv", index=False)

scenario_df = (
    run_df.groupby(["scenario", "N", "T"], as_index=False)
    .agg(
        mean_loss1=("avg_loss1", "mean"),
        mean_loss2=("avg_loss2", "mean"),
        mean_loss_diff=("avg_loss_diff", "mean"),
        share_model1_better=("model1_better", "mean"),
    )
)

scenario_df["distance_to_paper"] = (
    (scenario_df["N"] - 100).abs() + (scenario_df["T"] - 100).abs()
)

scenario_df = scenario_df.sort_values(["distance_to_paper", "N", "T"])
scenario_df.to_csv("results/replication_scenarios_summary.csv", index=False)

closest = scenario_df.iloc[0]

md = f"""# Closest scenario to the paper

The script was run for four scenarios:

- N=50, T=50
- N=50, T=100
- N=100, T=50
- N=100, T=100

Using a simple closeness rule based on sample sizes, the scenario closest to the paper among these four is:

- Scenario: {closest['scenario']}
- N: {int(closest['N'])}
- T: {int(closest['T'])}

This does not mean the full data-generating process is already identical to the paper.
It only means this scenario is the closest in terms of panel dimensions among the four tested cases.

Next steps:
- keep refining the dependence structure
- move the simulation design closer to the paper
- only consider Step 3 after Step 1 is judged strong enough
"""

with open("docs/closest_scenario.md", "w", encoding="utf-8") as f:
    f.write(md)

print("Saved: results/replication_run_summary.csv")
print("Saved: results/replication_scenarios_summary.csv")
print("Saved: docs/closest_scenario.md")
print(scenario_df)