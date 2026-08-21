# Weekly progress notes

## What I did this week

Followed up on last week's AR(1) fix with the three things the prof asked for after reading Section 4.1 of Qu, Timmermann & Zhu (2024):

1. Walk-forward time-series CV for AR(1) and RF - train through year Y, predict Y+1, moving forward one year at a time over the last 5 years (2021-2025), instead of only random splits.
2. Reran with a strictly per-unit AR(1) (no pooled fallback at all) under the same walk-forward setup, to see how much the fallback from last week's fix actually matters once countries aren't artificially removed.
3. Built a small simulated dataset from a known AR(1) process and reran the pipeline on it as a sanity check that everything's actually working correctly.

## Key results

- Walk-forward: AR(1) is very stable across the 5 years (avg ASEP 38.3), naive is worse (79.2). RF's average looks bad (110.8) but that's almost entirely one year - 2021, where RF spiked to 451. Best guess: training data through 2020 includes the COVID crash, and a tree-based model can't extrapolate to the 2021 rebound the way a linear AR(1) can. From 2022 onward RF is competitive with AR(1) again, so the average alone is misleading - the year-by-year plot tells the real story.
- Per-unit-only AR(1) came out exactly identical to the hybrid version in every single fold - the pooled fallback never had to fire, since every country already has years of its own history by the time we're predicting recent years. Confirms the fallback really is a new_units-specific fix, not something that changes normal forecasting.
- Simulated check: built fake panel data from a known AR(1) process (true noise variance = 9.0). AR(1) came back at 9.39, basically nailing it. RF got 10.28, naive got 11.26 - correct order, so the pipeline is doing what it's supposed to.

## From today's meeting

Walked through last week's AR(1) fix (new_units, AR(1) now ties with Random Forest, p = 0.32) and got answers on the open questions:

1. Pooling AR(1) vs marking it N/A for new_units - prof wants to also try a model with no AR dynamics at all, and a fixed-effects panel data model, as extra baselines to compare against.
2. Which design to center the thesis on - focus on new_periods for now, combine with new_units later.
3. Clustering countries before pooling AR(1) - hold off on that for now, try suggestion 1 first (the no-AR-dynamics and fixed-effects baselines).
4. Whether the ASEP-distribution comparison is enough on its own, or Qu et al.'s pooled test should run alongside it - still open, need to bring this up again.

## For next week

- Build the no-AR-dynamics baseline and a fixed-effects panel model, add both to the comparison.
- Center the write-up around new_periods, keep new_units as a secondary result to bring back in later.
- Get a clear answer on the pooled-test question (point 4 above) at the next meeting.
