Starting Positions
==================

Not recommended to use this utility. But in case you need this:

.. code-block:: python

	from mealpy.human_based import TLO
	import numpy as np


	def frequency_modulated(pos):
		# range: [-6.4, 6.35], f(X*) = 0, phi = 2pi / 100
		phi = 2 * np.pi / 100
		result = 0
		for t in range(0, 101):
			y_t = pos[0] * np.sin(pos[3] * t * phi + pos[1]*np.sin(pos[4] * t * phi + pos[2] * np.sin(pos[5] * t * phi)))
			y_t0 = 1.0 * np.sin(5.0 * t * phi - 1.5 * np.sin(4.8 * t * phi + 2.0 * np.sin(4.9 * t * phi)))
			result += (y_t - y_t0)**2
		return result

	fm_problem = {
		"fit_func": frequency_modulated,
		"lb": [-6.4, ] * 6,
		"ub": [6.35, ] * 6,
		"minmax": "min",
		"log_to": "console",
		"save_population": False,
	}
	term_dict1 = {
	   "mode": "FE",
	   "quantity": 5000    # 100000 number of function evaluation
	}

	## This is an example I use to create starting positions
	## Write your own function, remember the starting positions has to be: list of N vectors or 2D matrix of position vectors
	def create_starting_positions(n_dims=None, pop_size=None, num=1):
		return np.ones((pop_size, n_dims)) * num + np.random.uniform(-1, 1)

	## Define the model
	model = TLO.BaseTLO(fm_problem, epoch=100, pop_size=50, termination=term_dict1)

	## Input your starting positions here
	list_pos = create_starting_positions(6, 50, 2)
	best_position, best_fitness = model.solve(None, starting_positions=list_pos)        ## Remember the keyword: starting_positions
	print(f"Best solution: {model.solution}, Best fitness: {best_fitness}")

	## Training with other starting positions
	list_pos2 = create_starting_positions(6, 50, -1)
	best_position, best_fitness = model.solve(None, starting_positions=list_pos2)
	print(f"Best solution: {model.solution}, Best fitness: {best_fitness}")




.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4

