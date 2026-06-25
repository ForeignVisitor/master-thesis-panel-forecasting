# Master Thesis - Panel Forecasting

This repository contains the current code, notes, and simulation results for my Master's thesis.

## Thesis topic
Evaluating Machine Learning Forecasts for Panel Data

## Current focus
Right now I am working on the first replication step based on:

Qu, Timmermann, and Zhu (2024), *Comparing Forecasting Performance with Panel Data*

The current goal is to replicate part of the simulation setup and get the results reasonably close to the paper before moving on to the forecasting comparison part.

## Current repository structure
- `simulation/` replication scripts
- `results/` simulation outputs
- `docs/` working notes and short planning notes
- `paper/` thesis-related PDFs and reference material

## Current files
- `simulation/dgp1_size_replication.py`
- `results/dgp1_summary.csv`
- `results/dgp1_summary_v2.csv`
- `results/dgp1_runlevel_v2-2.csv`

## Current status
So far I have:
- started with DGP 1 and the size experiment,
- produced a first result table,
- improved the implementation once,
- and compared the first and second result versions.

The next step is to decide when this first replication block is close enough to the paper and then move to a small forecasting comparison with one statistical and one machine learning benchmark.