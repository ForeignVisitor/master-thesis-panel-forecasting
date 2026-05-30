# DGP1 changes v2

Version purpose:
Improve numerical closeness to the paper for Step 1.

Changes made in v2:
- Added burn-in of 500 periods before keeping the final T observations.
- Kept the same DGP 1 scenario grid:
  - phi in {0.5, 0.8}
  - N in {50, 100}
  - T in {50, 100, 500, 1000}
  - MC = 1000
- Kept mu1 = mu2 = 0 for the size experiment.
- Kept squared-error loss differential.
- Replaced Bartlett weight with 1 - j / (J + 1).

Expected effect:
- reduce over-rejection
- improve small-sample behavior
- get rejection rates closer to the paper

New output files:
- results/dgp1_runlevel_v2.csv
- results/dgp1_summary_v2.csv