Saving and Loading Model
========================

Based on the tutorials above, we know that we can save the population after each epoch in the model by setting "save_population" to True in the problem dictionary.

.. code-block:: python

	problem_dict1 = {
	  "fit_func": F5,
	  "lb": [-3, -5, 1, -10, ],
	  "ub": [5, 10, 100, 30, ],
	  "minmax": "min",
	  "log_to": "console",
	  "save_population": True,              # Default = False
	}


However, as a warning, if your problem is too big, setting "save_population" to True can cause memory issues when running the model. It is also important to
note that "save_population" here means storing the population of each epoch in the model's history object, and not saving the model to a file. To save and
load the optimizer from a file, you will need to use the "io" module from "mealpy.utils".


.. code-block:: python
	:emphasize-lines: 3,21,24

	import numpy as np
	from mealpy.evolutionary_based.GA import BaseGA
	from mealpy.utils import io

	def fitness_function(solution):
		return np.sum(solution**2)

	problem = {
	    "fit_func": fitness_function,
	    "lb": [-100, ] * 50,
	    "ub": [100, ] * 50,
	    "minmax": "min",
	}

	## Run the algorithm
	model = BaseGA(epoch=100, pop_size=50)
	best_position, best_fitness = model.solve(problem)
	print(f"Best solution: {best_position}, Best fitness: {best_fitness}")

	## Save model to file
	io.save_model(model, "results/model.pkl")

	## Load the model from file
	new_model = io.load_model("results/model.pkl")
	print(f"Best solution: {new_model.solution[0]}, Best fitness: {new_model.solution[1][0]}")


.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4

