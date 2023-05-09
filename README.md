## Snake optimization

Tasks:

1. Run `main.py`, investigate the code in `main.py` and `brain.py`.
2. Complete the implementation of PSO algorithm (useful links: [[1]](https://machinelearningmastery.com/a-gentle-introduction-to-particle-swarm-optimization/) and [[2]](https://en.wikipedia.org/wiki/Particle_swarm_optimization)). 
3. Increase the number of epochs, tune PSO hyperparameters: $w$, $c_1$, $c_2$.
4. Generate random parameters $r_1$ and $r_2$ for each particle or once per epoch. Compare the results.
5. Adjust PSO - remove part that uses particle's best position. Compare the results. Why do we need both parts (global and local)?