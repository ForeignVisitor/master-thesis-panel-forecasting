# Step 1 meeting note

What I replicated:
- DGP 1 only
- size experiment only
- phi = 0.5 and phi = 0.8
- N = 50, 100
- T = 50, 100, 500, 1000
- MC = 1000
- squared-error loss differential
- pooled panel DM-style statistic with Bartlett/Newey-West long-run variance

Files:
- simulation/dgp1_size_replication.py
- results/dgp1_runlevel.csv
- results/dgp1_summary.csv

What I want to ask:
- Are these results close enough numerically to the paper for Step 1?
- Should I next add the cluster-based tests from the paper, or first refine DGP 1 further?
- Do you want the conditional distribution function to remain a side diagnostic only?