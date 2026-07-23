Multi-objective Optimization
============================

.. toctree::
   :maxdepth: 3


.. important::
    **The Weighting Method**

    MEALPY currently utilizes the **"weighting method"** (scalarization) to solve multi-objective optimization problems. This straightforward approach aggregates multiple objectives into a single scalar fitness value by multiplying each objective by a predefined weight and summing them up.

To implement this, you simply need to design your objective function to return a **list of objective values** and specify the corresponding weights.

Configuration Parameters
------------------------

* ``obj_func`` *(callable)*: Your objective function, which must return a list of numerical values.
* ``bounds`` *(list/FloatVar)*: The lower and upper boundaries of the problem space.
* ``minmax`` *(str)*: Indicates whether the optimization goal is to minimize (``"min"``) or maximize (``"max"``) the objectives.
* ``obj_weights`` *(list)*: *(Optional)* A list of fractional weights corresponding to each objective.

.. note::
    **Default Weights Behavior**

    If you do not specify ``obj_weights``, the algorithm defaults to assigning an equal weight of ``1`` to all objectives (e.g., ``[1, 1, ..., 1]``).


Approach 1: Using a Problem Dictionary
--------------------------------------

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


Approach 2: Creating a Custom Problem Class
-------------------------------------------

.. hint::
    **Best Practice for Complex Problems**

    Inheriting from the ``Problem`` class is highly recommended for complex multi-objective scenarios. It keeps your codebase modular, readable, and perfectly encapsulated.


.. code-block:: python

	import numpy as np
	from mealpy import PSO, FloatVar, Problem

	## Define a custom child class of Problem class.
	class MOP(Problem):
	    def __init__(self, bounds=None, minmax="min", **kwargs):
	        super().__init__(bounds, minmax, **kwargs)

		def booth(self, x, y):
			return (x + 2*y - 7)**2 + (2*x + y - 5)**2
		def bukin(self, x, y):
			return 100 * np.sqrt(np.abs(y - 0.01 * x**2)) + 0.01 * np.abs(x + 10)
		def matyas(self, x, y):
			return 0.26 * (x**2 + y**2) - 0.48 * x * y

	    def obj_func(self, solution):
	        return [self.booth(solution[0], solution[1]), self.bukin(solution[0], solution[1]), self.matyas(solution[0], solution[1])]

	## Create an instance of MOP class
	problem_multi = MOP(bounds=FloatVar(lb=[-10, ] * 2, ub=[10, ] * 2), minmax="min", obj_weights=[0.4, 0.1, 0.5])

	## Define the model and solve the problem
	model = PSO.OriginalPSO(epoch=1000, pop_size=50)
	model.solve(problem=problem_multi)
