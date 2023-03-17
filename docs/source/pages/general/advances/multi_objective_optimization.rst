Multi-objective Optimization
============================

We currently offer a "weighting method" to solve multi-objective optimization problems. All you need to do is define your fitness function, which returns a
list of objective values, and set the objective weight corresponding to each value.

	* fit_func: Your fitness function.
	* lb: Lower bound of variables. It should be a list of values.
	* ub: Upper bound of variables. It should be a list of values.
	* minmax: Indicates whether the problem you are trying to solve is a minimum or maximum. The value can be "min" or "max".
	* obj_weights: Optional list of weights for all of your objectives. The default is [1, 1, ..., 1].


* Declare problem dictionary with "obj_weights":

.. code-block:: python

	import numpy as np
	from mealpy.swarm_based import PSO

	## This is how you design multi-objective function
	#### Link: https://en.wikipedia.org/wiki/Test_functions_for_optimization
	def fitness_multi(solution):
	    def booth(x, y):
	        return (x + 2*y - 7)**2 + (2*x + y - 5)**2
	    def bukin(x, y):
	        return 100 * np.sqrt(np.abs(y - 0.01 * x**2)) + 0.01 * np.abs(x + 10)
	    def matyas(x, y):
	        return 0.26 * (x**2 + y**2) - 0.48 * x * y
	    return [booth(solution[0], solution[1]), bukin(solution[0], solution[1]), matyas(solution[0], solution[1])]

	## Design a problem dictionary for multiple objective functions above
	problem_multi = {
	    "fit_func": fitness_multi,
	    "lb": [-10, -10],
	    "ub": [10, 10],
	    "minmax": "min",
	    "obj_weights": [0.4, 0.1, 0.5]               # Define it or default value will be [1, 1, 1]
	}

	## Define the model and solve the problem
	model = PSO.OriginalPSO(epoch=1000, pop_size=50)
	model.solve(problem=problem_multi)



* Declare a custom Problem class:

.. code-block:: python

	import numpy as np
	from mealpy.swarm_based import PSO
	from mealpy.utils.problem import Problem

	## Define a custom child class of Problem class.
	class MOP(Problem):
	    def __init__(self, lb, ub, minmax, name="MOP", **kwargs):
	        super().__init__(lb, ub, minmax, **kwargs)
	        self.name = name

		def booth(x, y):
			return (x + 2*y - 7)**2 + (2*x + y - 5)**2
		def bukin(x, y):
			return 100 * np.sqrt(np.abs(y - 0.01 * x**2)) + 0.01 * np.abs(x + 10)
		def matyas(x, y):
			return 0.26 * (x**2 + y**2) - 0.48 * x * y

	    def fit_func(self, solution):
	        return [self.booth(solution[0], solution[1]), self.bukin(solution[0], solution[1]), self.matyas(solution[0], solution[1])]

	## Create an instance of MOP class
	problem_multi = MOP(lb=[-10, ] * 2, ub=[10, ] * 2, minmax="min", obj_weights=[0.4, 0.1, 0.5])

	## Define the model and solve the problem
	model = PSO.OriginalPSO(epoch=1000, pop_size=50)
	model.solve(problem=problem_multi)



.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4

