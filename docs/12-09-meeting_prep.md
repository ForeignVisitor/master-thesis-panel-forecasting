# Meeting prep - Part A / PWT

## What I did this week

Started on the new direction from last week's notes - splitting the thesis into Part A (just one variable Y, no RF) and Part B (add a known covariate X, that's where RF comes back in).

For Part A:
- Added mean and median as predictors - for each country, just guess that country's own historical average or median for a new observation. No RF, no dynamic panel stuff, exactly as asked.
- Kept AR(1) in the comparison too since that was "maybe" in the notes.
- Added a second error metric (average absolute error) next to the ASEP (squared error) we already had, because I realized squared error is technically always minimized by the mean, not the median - so I wanted a fair way to actually check whether median does better instead of just assuming it.
- Built a synthetic "skewed monetary" dataset - fake companies with a typical value each, but occasional one-off spike years, like a real euro/dollar variable would have.
- Also got real Penn World Table data working - GDP per capita and per capita employment, 185 countries, 1950-2023. Had to write a script to download and load it since there's no Python package for PWT, only an Excel file, but got that running.
- Ran the mean/median/AR(1) comparison on all three: our regular GDP-growth panel, the synthetic skewed data, and the real PWT data. ECDF plots for both metrics on all three.

## Key results

1. On our normal GDP-growth data (not skewed) - mean, median and AR(1) all come out basically tied. Makes sense, nothing weird going on there.
2. On the synthetic skewed data - median clearly beats mean, on both squared error AND absolute error. So the "median is more robust to outliers" idea holds up, it's not just a coincidence of picking the right metric.
3. On real PWT data (GDP per capita) - AR(1) wins by a lot, like 10x lower error than mean or median. This one's the interesting one to bring up: GDP per capita has a strong upward trend over decades within a country, so "guess the historical average" is a bad idea for a trending series - you basically guess something from the middle of its whole history instead of where it actually is now. AR(1) uses the last known value so it tracks the trend. This is actually a good explanation for why the original project used GDP *growth* and not GDP *levels* - growth is roughly stable over time so mean/median make sense there, levels don't.

## Questions I think he might ask, and how I'd answer

**Why no Random Forest in Part A?**
Because RF needs something to split decisions on. With just one variable and only its own history, there's no covariate to split on - it would just reduce to guessing a summary statistic anyway, which is literally what mean/median already do. RF only makes sense once we add X in Part B.

**Isn't the mean always the best guess for squared error? Why even compare it to median?**
Yes, in theory, in the population, the mean minimizes squared error. But that's an infinite-data result. With a small number of observations per unit and occasional outliers, the *sample* mean gets pulled around by that one big number a lot more than the *sample* median does. So in practice, with realistic sample sizes, median can come out ahead even on squared error - which is exactly what happened on the skewed synthetic data. I checked this wasn't a fluke by also looking at absolute error, which the median is supposed to win on by definition, and it did win on both.

**Why build synthetic data instead of just using real data?**
Because I needed a dataset where I actually control the skew and know it's there by construction, so I can be sure any "median wins" result isn't an accident of one particular dataset - it's a controlled check before trusting the same pattern if it shows up in real skewed data later.

**What's the deal with the PWT result - why does AR(1) win so much there?**
GDP per capita levels are non-stationary - they trend up over decades. A historical mean or median completely ignores time and just averages across a country's whole history, which is a bad idea when the series has moved a lot since then. AR(1) at least uses the last observed value, so it captures the trend. It's a good illustration of when "just guess the average" breaks down.

**Are we still avoiding dynamic panel data methods?**
Yes - mean, median, and this simple AR(1) are not dynamic panel estimators (no GMM, no Arellano-Bond instruments). They're just per-unit summary predictors, which is what was asked for.

**What's next / what about Part B?**
Bring in per capita employment as X alongside GDP per capita as Y, and that's where Random Forest becomes relevant again, since it needs a covariate to actually use.

## Notes from the meeting
(fill in after)

## For next week
(fill in after meeting)
