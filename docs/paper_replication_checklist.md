# Replication checklist

## Goal
Replicate one simple Monte Carlo setting from the paper as closely as possible.

## First target
- Balanced panel
- N = 50
- T = 100
- 2 competing forecast models
- Squared-error loss
- Loss differential = e1^2 - e2^2
- Report rejection rate or average loss comparison over many simulations

## What to match to the paper as much as possible
- Panel structure
- Forecast error persistence
- Repeated Monte Carlo runs
- Squared-error loss
- Comparison of two competing forecasts

## Deliverable for next meeting
- One script
- One CSV of simulation results
- One short note explaining what was matched and what was simplified

## Next paper-matching upgrades
- Use T = 50 and T = 100 versions
- Use N = 50 and N = 100 versions
- Add AR(1) persistence in forecast errors
- Compare rejection behavior under equal predictive ability
- Compute squared-error-loss differential by unit and time