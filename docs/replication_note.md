# Replication note

I started with a simplified Monte Carlo replication designed to be close to the paper.

Current choices:
- balanced panel
- N = 50
- T = 100
- 500 Monte Carlo repetitions
- two competing forecasts
- squared-error loss
- average loss differential as main comparison

Simplifications:
- only one simple dependence structure so far
- not yet matching all DGPs from the paper
- not yet implementing the full test statistics from the paper

Next step:
- move closer to one exact DGP from the paper
- add one test statistic instead of only average loss comparison