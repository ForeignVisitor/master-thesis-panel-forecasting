# Supervisor feedback - restart point

## Terminology correction

Use "prediction performance," not "forecasting performance."

Reasoning from supervisor:
- A prediction can apply to new cross-sectional units or to new (future) time periods.
- A forecast only applies to new time periods.
- Since the thesis studies both new units and new periods, "prediction" is the correct term.

## Scope correction

Qu et al. (2024) is the starting point only. The thesis does not need to include or relate to every part of that paper.

## New core focus

Focus on comparing the prediction error distribution across different methods.

Reference example: Figure 1 in Haupt, Schnurbus, and Tschernig (2010).

What Figure 1 does:
- runs many Monte Carlo replications with a random train/test split each time
- for each replication and each method, computes the Average Squared Error of Prediction (ASEP)
- plots the empirical cumulative distribution function of ASEP for each method
- checks whether one method's distribution lies entirely to the left of another (first-order stochastic dominance)
- reports the percentage of replications in which one method predicts better than another

## New thesis direction

Compare the prediction error distribution of a small set of methods on panel data, following the ASEP-distribution comparison style of Figure 1, rather than only reporting a single aggregate pooled test statistic.