# Step 1 status

Done:
- Created balanced simulated panel
- Used unit index and time index
- Generated target y_it with persistence
- Created two forecast competitors: AR(1) and naive lag forecast
- Computed forecast errors and squared-error loss differential

Next:
- Add train/test split
- Re-estimate AR(1) from training data instead of using true rho
- Reproduce one simple result pattern from the paper