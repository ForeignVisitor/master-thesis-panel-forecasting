# DGP1 check v1

Checked file:
- results/dgp1_summary.csv

What is already correct:
- All 16 scenarios are present.
- mean_jdm is close to zero.
- mean_loss_diff is close to zero.

Main problem:
- rejection rates are too high relative to a 5% nominal level.
- the problem is strongest when phi = 0.8 and T is small.
- std_jdm is also too large in those scenarios.

Interpretation:
- the simulation is centered reasonably well
- the main remaining issue is over-rejection
- the next rerun should focus on better finite-sample behavior

Decision:
- keep DGP 1
- keep the same 16 scenarios
- rerun with burn-in and corrected Bartlett weighting