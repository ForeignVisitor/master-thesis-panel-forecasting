# Master Thesis - Panel Data Prediction Performance

## Topic

Evaluating prediction performance for panel data, comparing statistical and machine learning methods.

## Terminology

This project uses "prediction," not "forecast."

A prediction applies to:
- new cross-sectional units, or
- new (future) time periods.

A forecast only applies to new time periods, so "prediction" is the correct general term for this thesis.

## Starting point

The project starts from:

Qu, Timmermann, and Zhu (2024), "Comparing Forecasting Performance with Panel Data"

This paper is a starting point only, not a paper to fully replicate section by section.

## Core focus

The central empirical approach is to compare the prediction error distribution across methods, following the style of Figure 1 in:

Haupt, Schnurbus, and Tschernig (2010), "On Nonparametric Estimation of a Hedonic Price Function"

That figure:
- repeats many random train/test splits
- computes Average Squared Error of Prediction (ASEP) per method per replication
- plots the empirical distribution function of ASEP per method
- checks stochastic dominance between methods
- reports win-rate percentages between method pairs

## Current status

Restarted clean after supervisor feedback on 2026 meeting.

## Structure

- `paper/` reference PDFs
- `docs/` planning and supervisor notes
- `src/` code
- `data/` datasets
- `results/` output tables and plots
- `notebooks/` exploratory notebooks