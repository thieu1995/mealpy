Discrete Optimization
=====================

For this type of problem, we recommend creating a custom child class of the Problem class and overriding the necessary functions.
At a minimum, three functions should be overridden:

	* obj_func: the fitness function
	* generate_position: a function that generate solution
	* amend_position: a function that bring back the solution to the boundary


Let's say we want to solve Travelling Salesman Problem (TSP),


.. code-block:: python

	import numpy as np
	from mealpy import PermutationVar       ## For Travelling Salesman Problem, the solution should be a permutation


	class DOP(Problem):
	    def __init__(self, bounds, minmax, name="DOP", CITY_POSITIONS=None, **kwargs):
	        self.name = name
	        self.CITY_POSITIONS = CITY_POSITIONS
	        super().__init__(bounds, minmax, **kwargs)

	    def obj_func(self, solution):
			## Objective for this problem is the sum of distance between all cities that salesman has passed
		    ## This can be change depend on your requirements
		    x = self.decode_solution(solution)["per"]
		    city_coord = self.CITY_POSITIONS[x]
		    line_x = city_coord[:, 0]
		    line_y = city_coord[:, 1]
		    total_distance = np.sum(np.sqrt(np.square(np.diff(line_x)) + np.square(np.diff(line_y))))
		    return total_distance


	## Create an instance of DOP class
	problem_cop = DOP(bounds=PermutationVar(valid_set=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], name="per"),
					minmax="min", log_to="file", log_file="dop-results.txt")

	## Define the model and solve the problem
	model = PSO.OriginalPSO(epoch=1000, pop_size=50)
	model.solve(problem=problem_cop)


.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4

