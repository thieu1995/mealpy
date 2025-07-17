Reproducibility (Set Seed)
==========================

By default, the `seed` parameter is set to `None`. When `seed=None`, the algorithm will use a different random seed each time it runs, leading to potentially different results across multiple executions.

However, if you set a specific integer value for `seed`, the algorithm will produce the exact same results every time it is run with that particular seed.
This is crucial for debugging, verifying performance, and ensuring that your experiments are reproducible.

For scenarios where you need to run the algorithm multiple times (e.g., n-trials for statistical analysis) while still maintaining reproducibility
for each individual trial, it is best practice to set a *different* seed for each trial. This allows you to recreate each specific trial's outcome if needed,
while still exploring the stochastic nature of the algorithm across different runs.

.. code-block:: python

    from mealpy import SMA

    # Assuming 'problem' is already defined, e.g.:
    # problem = {
    #     "obj_func": lambda solution: sum(x**2 for x in solution),
    #     "bounds": FloatVar(lb=(-10., )*30, ub=(10., )*30),
    #     "minmax": "min",
    # }

    # Example 1: Running with default seed (None) - results may vary each time
    model_default = SMA.OriginalSMA(epoch=100, pop_size=50, pr=0.03)
    g_best_default = model_default.solve(problem=problem)
    print(f"Run 1 (default seed) - Best fitness: {g_best_default.target.fitness}")

    g_best_default_2 = model_default.solve(problem=problem) # Rerunning the same model instance might reset some internal states, but generally results will differ due to new random seed
    print(f"Run 2 (default seed) - Best fitness: {g_best_default_2.target.fitness}")

    # Example 2: Running with a specific seed - results will be identical for seed=10
    model_seeded_1 = SMA.OriginalSMA(epoch=100, pop_size=50, pr=0.03)
    g_best_seeded_1 = model_seeded_1.solve(problem=problem, seed=10)
    print(f"Run with seed=10 - Best fitness: {g_best_seeded_1.target.fitness}")

    model_seeded_2 = SMA.OriginalSMA(epoch=100, pop_size=50, pr=0.03) # Create a new model instance for clarity
    g_best_seeded_2 = model_seeded_2.solve(problem=problem, seed=10)
    print(f"Run with seed=10 (again) - Best fitness: {g_best_seeded_2.target.fitness}") # This will yield the same fitness as above

    # Example 3: Running multiple trials with different seeds for reproducibility of each trial
    print("\nRunning 3 trials with different seeds:")
    for i in range(3):
        current_seed = 100 + i # Use a unique seed for each trial
        model_trial = SMA.OriginalSMA(epoch=100, pop_size=50, pr=0.03)
        g_best_trial = model_trial.solve(problem=problem, seed=current_seed)
        print(f"  Trial {i+1} (seed={current_seed}) - Best fitness: {g_best_trial.target.fitness}")


.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4
