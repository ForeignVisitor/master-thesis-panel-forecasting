# Working plan after restart

## Step 1: Data and prediction setup

- choose one panel dataset
- define cross-sectional units and time periods
- define target variable to predict
- decide whether "new units" or "new time periods" prediction is used first

## Step 2: Candidate methods

- one simple statistical benchmark
- one machine learning benchmark

## Step 3: Monte Carlo prediction comparison

- repeat many random train/test splits
- for each split and each method, compute ASEP
- store all ASEP values in one table

## Step 4: Prediction error distribution comparison

- plot empirical CDF of ASEP per method
- check whether one method's distribution lies to the left of another
- compute win-rate: percentage of replications where method A beats method B
- test whether mean ASEP differs significantly between methods

## Open questions for supervisor

1. Should the first version predict new time periods, new units, or both?
2. Which dataset is most appropriate for this narrower ASEP-distribution focus?
3. How many Monte Carlo replications are reasonable given the panel size?
4. Should Qu et al.'s pooled test still be used alongside the ASEP-distribution comparison, or is the distribution comparison now the main tool?