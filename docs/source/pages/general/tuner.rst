=============================
Tuner / Hyperparameter Tuning
=============================

.. toctree::
   :maxdepth: 3


Selecting the optimal hyperparameters for a meta-heuristic algorithm can significantly impact its convergence speed and final performance. To simplify this process, MEALPY provides a dedicated ``Tuner`` class designed to automatically search for the best parameter combinations based on a predefined grid.

.. hint::
    **Advanced Tuning Examples**

    For more complex tuning scenarios (including continuous variables or multi-algorithm tuning), please explore our dedicated examples folder: `Tuner-Examples <https://github.com/thieu1995/mealpy/tree/master/examples>`_.

How it Works
------------

The ``Tuner`` operates similarly to scikit-learn's ``GridSearchCV``. You provide a base algorithm model and a dictionary mapping parameter names to lists of potential values. The tuner will evaluate every possible combination over multiple trials, calculate the mean performance, and return the optimal model configuration.

.. important::
    **Retraining with the Best Parameters (The resolve method)**

    After the tuning process finishes, you do not need to manually copy-paste the best parameters! You can directly use the ``tuner.resolve()`` method. This function internally calls ``solve()`` on the best-found algorithm instance using the same problem, while allowing you to pass additional runtime parameters (like parallel execution modes).

Code Example
------------

.. code-block:: python

    from opfunu.cec_based.cec2017 import F52017
    from mealpy import FloatVar, BBO, Tuner

    ## 1. Define the Problem
    f1 = F52017(30, f_bias=0)

    p1 = {
        "bounds": FloatVar(lb=f1.lb, ub=f1.ub),
        "obj_func": f1.evaluate,
        "minmax": "min",
        "name": "F5",
        "log_to": "console",
    }

    ## 2. Define the Hyperparameter Grid
    # Note: The keys must strictly match the parameter names of the chosen algorithm!
    paras_bbo_grid = {
        "epoch": [10],
        "pop_size": [10],
        "n_elites": [2, 3, 4, 5],
        "p_m": [0.01, 0.02, 0.05]
    }

    ## 3. Define Termination Criteria (Optional but recommended)
    term = {
        "max_epoch": 200,
        "max_time": 20,
        "max_fe": 10000
    }

    if __name__ == "__main__":
        ## 4. Initialize the Model and Tuner
        model = BBO.OriginalBBO()
        tuner = Tuner(model, paras_bbo_grid)

        ## 5. Execute the Tuning Process
        # n_trials=5: Run each parameter combination 5 times to calculate a stable mean score.
        # n_jobs=4: Distribute these trials across 4 CPU cores for parallel execution.
        tuner.execute(problem=p1, termination=term, n_trials=5, n_jobs=4, verbose=True)

        ## 6. Access and Export Results
        print("Best DataFrame Row:\n", tuner.best_row)
        print("Best Score (Mean Fitness):", tuner.best_score)
        print("Best Parameters:", tuner.best_params)

        # Export the comprehensive tuning history to CSV files and generate plots
        tuner.export_results()
        tuner.export_figures()

        ## 7. Re-solve the problem using the discovered optimal parameters
        # The resolve() function re-uses the defined problem but allows you to inject execution modes
        g_best = tuner.resolve(mode="thread", n_workers=4, termination=term)

        print(f"\nRefined Best Solution: {g_best.solution}")
        print(f"Refined Best Fitness: {g_best.target.fitness}")

        # Verify the name of the optimally tuned algorithm instance
        print(f"Optimal Algorithm Instance: {tuner.best_algorithm.get_name()}")
