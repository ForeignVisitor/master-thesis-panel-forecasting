# Step 1 target

Paper:
Qu, Timmermann, Zhu - Comparing Forecasting Performance with Panel Data

Immediate replication target:
- DGP 1 only
- Size experiment only
- MC = 1000
- phi in {0.5, 0.8}
- N in {50, 100}
- T in {50, 100, 500, 1000}
- mu1 = mu2 = 0
- squared-error loss
- pooled panel DM-style statistic
- Bartlett/Newey-West long-run variance with lag T^(1/3)

Current issue from first run:
- rejection rates are too high, especially when phi = 0.8 and T is small

Next goal:
- add burn-in
- correct Bartlett weighting
- rerun all 16 scenarios
- check whether rejection rates move closer to the paper