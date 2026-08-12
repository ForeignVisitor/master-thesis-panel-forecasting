# Weekly progress notes

## Where I left off

Last week's script was `timeseries_and_new_periods_diagnostics.py`, trying to answer why the `new_periods` ASEP distribution has a sharp vertical jump instead of a smooth curve. Quick recap since it still matters this week too:

- Plotted GDP growth over time for the 12 countries with the most data, with 2008, 2009 and 2020 marked as known crisis years.
- Plotted the global average per year with a +/- 1 std dev band - dips close to 0% in 2009 and around -5% in 2020, band widens a lot both times.
- Re-ran new_periods but tracked which years got held out each replication, and flagged whether a crisis year was among them.
- Result: whenever a crisis year lands in the holdout, all three methods jump to a clearly higher ASEP. Numbers: without a crisis year - naive 53.3, AR(1) 37.2, RF 39.7. With a crisis year - naive 80.2, AR(1) 50.2, RF 52.4. So the jump isn't a bug, it's real - none of the methods can see a global shock coming.

## This week: fixing the AR(1) bug on new_units

Found last week that the "AR(1)" results for new_units were fake. Reason: a held-out country has 0 training rows by definition, and the AR(1) function had a fallback that copies the naive prediction whenever a unit has fewer than 5 training rows. 0 is always below 5, so every single new_units test row was quietly falling back to naive. Confirmed by the significance test - AR(1) and naive had exactly the same mean and the t-test couldn't even compute a stat because there was zero variation between them.

Two options were on the table for fixing this:
1. Mark AR(1) as not applicable for new_units instead of silently faking it.
2. Fit one AR(1) pooled across all training countries and use that as the fallback, so it can still predict a country it's never seen.

Went with option 2. Instead of falling back to naive, the fallback now fits one intercept + slope across the whole training set (not per-country), which doesn't need any rows from the specific country being predicted. Same rule kicks in for any country with under 5 training rows, not just the fully unseen ones.

Reran everything after the fix (full 200 replications, all three scripts). Results:

- new_units: AR(1) is no longer identical to naive anywhere (0/200 matching replications, used to be 200/200). Mean ASEP dropped from 54.5 (fake, = naive) to 35.3.
- The interesting part: on new_units, AR(1) and Random Forest are now statistically indistinguishable (35.31 vs 35.53, t-test p = 0.32). A pooled AR(1) predicts an unseen country about as well as RF does. Wasn't expecting that - worth bringing up.
- random_rows and new_periods barely moved, since almost every unit has enough of its own history there anyway (RF still wins on both).
- Crisis-year numbers came back basically the same as last week (53.3/37.2/39.7 and 80.2/50.2/52.4) - makes sense, that part of the pipeline wasn't really touched by the fix.

## Also cleaned up the code

The three scripts (`predict_performance_pipeline.py`, `predict_performance_by_split_type.py`, `timeseries_and_new_periods_diagnostics.py`) were each defining their own copy of `naive_predict`, `ar1_predict`, `rf_predict`, `asep`, and the split functions. Moved all of that into one shared `src/` folder so the fix only had to happen in one place, and any future change won't need to be copy-pasted three times. Also added `smoke_test.py` at the repo root - a fast (~1 min) check I can run after touching the code instead of waiting for the full 200-rep run every time.

## For next meeting

- Went with the pooled AR(1) fix (option 2) over marking it N/A - want to check that's the right call, especially given the AR(1)/RF tie above.
- Next step per the clustering idea from before: cluster countries first (growth volatility / region) and fit one pooled AR(1) per cluster instead of one global pool, so an unseen country borrows dynamics from similar countries rather than the whole world average. Haven't started this yet.
