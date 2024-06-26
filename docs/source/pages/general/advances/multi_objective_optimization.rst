Multi-objective Optimization
============================

We currently offer a "weighting method" to solve multi-objective optimization problems. All you need to do is define your objective function, which returns a
list of objective values, and set the objective weights corresponding to each value.

	* obj_func: Your objective function.
	* bounds: A list or an instance of problem type.
	* minmax: Indicates whether the problem you are trying to solve is a minimum or maximum. The value can be "min" or "max".
	* obj_weights: Optional list of weights for all of your objectives. The default is [1, 1, ..., 1].


* Declare problem dictionary with "obj_weights":

.. code-block:: python

	import numpy as np
	from mealpy import PSO, FloatVar, Problem

	## This is how you design multi-objective function
	#### Link: https://en.wikipedia.org/wiki/Test_functions_for_optimization
	def objective_multi(solution):
	    def booth(x, y):
	        return (x + 2*y - 7)**2 + (2*x + y - 5)**2
	    def bukin(x, y):
	        return 100 * np.sqrt(np.abs(y - 0.01 * x**2)) + 0.01 * np.abs(x + 10)
	    def matyas(x, y):
	        return 0.26 * (x**2 + y**2) - 0.48 * x * y
	    return [booth(solution[0], solution[1]), bukin(solution[0], solution[1]), matyas(solution[0], solution[1])]

	## Design a problem dictionary for multiple objective functions above
	problem_multi = {
	    "obj_func": objective_multi,
	    "bounds": FloatVar(lb=[-10, -10], ub=[10, 10]),
	    "minmax": "min",
	    "obj_weights": [0.4, 0.1, 0.5]               # Define it or default value will be [1, 1, 1]
	}

	## Define the model and solve the problem
	model = PSO.OriginalPSO(epoch=1000, pop_size=50)
	model.solve(problem=problem_multi)



* Declare a custom Problem class:

.. code-block:: python

	import numpy as np
	from mealpy import PSO, FloatVar, Problem

	## Define a custom child class of Problem class.
	class MOP(Problem):
	    def __init__(self, bounds=None, minmax="min", **kwargs):
	        super().__init__(bounds, minmax, **kwargs)

		def booth(x, y):
			return (x + 2*y - 7)**2 + (2*x + y - 5)**2
		def bukin(x, y):
			return 100 * np.sqrt(np.abs(y - 0.01 * x**2)) + 0.01 * np.abs(x + 10)
		def matyas(x, y):
			return 0.26 * (x**2 + y**2) - 0.48 * x * y

	    def obj_func(self, solution):
	        return [self.booth(solution[0], solution[1]), self.bukin(solution[0], solution[1]), self.matyas(solution[0], solution[1])]

	## Create an instance of MOP class
	problem_multi = MOP(bounds=FloatVar(lb=[-10, ] * 2, ub=[10, ] * 2), minmax="min", obj_weights=[0.4, 0.1, 0.5])

	## Define the model and solve the problem
	model = PSO.OriginalPSO(epoch=1000, pop_size=50)
	model.solve(problem=problem_multi)



.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4

