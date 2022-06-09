Agent's History
===============

You can access to the history of agent/population in model.history object with variables:
	+ list_global_best: List of global best SOLUTION found so far in all previous generations
	+ list_current_best: List of current best SOLUTION in each previous generations
	+ list_epoch_time: List of runtime for each generation
	+ list_global_best_fit: List of global best FITNESS found so far in all previous generations
	+ list_current_best_fit: List of current best FITNESS in each previous generations
	+ list_diversity: List of DIVERSITY of swarm in all generations
	+ list_exploitation: List of EXPLOITATION percentages for all generations
	+ list_exploration: List of EXPLORATION percentages for all generations
	+ list_population: List of POPULATION in each generations

**Warning**, the last variable 'list_population' can cause the error related to 'memory' when saving model. Better to set parameter 'save_population' to
False in the input problem dictionary to not using it.

.. code-block:: python

	import numpy as np
	from mealpy.swarm_based.PSO import BasePSO

	def fitness_function(solution):
	    return np.sum(solution**2)

	problem_dict = {
	    "fit_func": fitness_function,
	    "lb": [-10, -15, -4, -2, -8],
	    "ub": [10, 15, 12, 8, 20],
	    "minmax": "min",
	    "verbose": True,
	    "save_population": False        # Then you can't draw the trajectory chart
	}
	model = BasePSO(problem_dict, epoch=1000, pop_size=50)

	print(model.history.list_global_best)
	print(model.history.list_current_best)
	print(model.history.list_epoch_time)
	print(model.history.list_global_best_fit)
	print(model.history.list_current_best_fit)
	print(model.history.list_diversity)
	print(model.history.list_exploitation)
	print(model.history.list_exploration)
	print(model.history.list_population)

	## Remember if you set "save_population" to False, then there is no variable: list_population



.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4

