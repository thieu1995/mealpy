=============================
Tuner / Hyperparameter Tuning
=============================

We build a dedicated class, Tuner, that can help you tune your algorithm's parameters.

Please head to examples folder to learn more about this `Tuner-Examples`_

.. _Tuner-Examples: https://github.com/thieu1995/mealpy/tree/master/examples


**Below is a simple example with Tuner class**

.. code-block:: python

	import numpy as np
	from mealpy.bio_based import BBO
	from mealpy.tuner import Tuner     # We will use this Tuner utility

	### Define problem
	def fitness(solution):
	    return np.sum(solution**2) + np.sum(solution**3)

	problem = {
	    "lb": [-10] * 20,    # 20 dimensions
	    "ub": [10] * 20,
	    "minmax": "min",
	    "fit_func": fitness,
	    "name": "Mixed Square and Cube Problem",
	    "log_to": None,
	}

	### Define model and parameter grid of the model (just like ParameterGrid / GridSearchCV)
	model = BBO.BaseBBO()

	paras_bbo_grid = {
	    "epoch": [100],
	    "pop_size": [50],
	    "elites": [2, 3, 4, 5],
	    "p_m": [0.01, 0.02, 0.05, 0.1, 0.15, 0.2]
	}

	### Define the Tuner and run it
	if __name__ == "__main__":
		tuner = Tuner(model, paras_bbo_grid)

		## Try to run this optimizer on this problem 10 times (n_trials = 10).
	    ## Get the best model by mean value of all trials
	    tuner.execute(problem=problem, n_trials=10, mode="parallel", n_workers=4)

	    ## Better to save the tunning results to CSV for later usage
	    tuner.export_results("history/tuning1", save_as="csv")

	    ## Print out the best pameters
	    print(f"Best parameter: {tuner.best_params}")

	    ## Print out the best score of the best parameter
	    print(f"Best score: {tuner.best_score}")

	    ## Print out the algorithm with the best parameter
	    print(f"Best Optimizer: {tuner.best_algorithm}")


	    ## Now we can even re-train the algorithm with the best parameter by calling resolve() function
	    ## Resolve() function will call the solve() function in algorithm with default problem parameter is removed.
	    ##    other parameters of solve() function is keeped and can be used.

	    best_position, best_fitness = tuner.resolve()
	    print(f"Best solution after re-solve: {best_position}")
	    print(f"Best fitness after re-solve: {best_fitness}")



.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4