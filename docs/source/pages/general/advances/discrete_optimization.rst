Discrete Optimization
=====================

For this type of problem, we recommend creating a custom child class of the Problem class and overriding the necessary functions.
At a minimum, three functions should be overridden:

	* fit_func: the fitness function
	* generate_position: a function that generate solution
	* amend_position: a function that bring back the solution to the boundary


Let's say we want to solve Travelling Salesman Problem (TSP),


.. code-block:: python

	import numpy as np

	class DOP(Problem):
	    def __init__(self, lb, ub, minmax, name="DOP", CITY_POSITIONS=None, **kwargs):
	        super().__init__(lb, ub, minmax, **kwargs)
	        self.name = name
	        self.CITY_POSITIONS = CITY_POSITIONS

		def generate_position(lb=None, ub=None):
		    ## For Travelling Salesman Problem, the solution should be a permutation
		    ## Lowerbound: [0, 0,...]
		    ## Upperbound: [N_cities - 1.11, ....]
		    return np.random.permutation(len(lb))

		def amend_position(solution, lb=None, ub=None):
		    # print(f"Raw: {solution}")
		    ## Bring them back to boundary
		    solution = np.clip(solution, lb, ub)

		    solution_set = set(list(range(0, len(solution))))
		    solution_done = np.array([-1, ] * len(solution))
		    solution_int = solution.astype(int)
		    city_unique, city_counts = np.unique(solution_int, return_counts=True)

		    for idx, city in enumerate(solution_int):
		        if solution_done[idx] != -1:
		            continue
		        if city in city_unique:
		            solution_done[idx] = city
		            city_unique = np.where(city_unique == city, -1, city_unique)
		        else:
		            list_cities_left = list(solution_set - set(city_unique) - set(solution_done))
		            solution_done[idx] = list_cities_left[0]
		    # print(f"Final solution: {solution_done}")
		    return solution_done

	    def fit_func(self, solution):
			## Objective for this problem is the sum of distance between all cities that salesman has passed
		    ## This can be change depend on your requirements
		    city_coord = self.CITY_POSITIONS[solution]
		    line_x = city_coord[:, 0]
		    line_y = city_coord[:, 1]
		    total_distance = np.sum(np.sqrt(np.square(np.diff(line_x)) + np.square(np.diff(line_y))))
		    return total_distance


	## Create an instance of DOP class
	problem_cop = DOP(lb=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], ub=[1, 1, 1, 1, 1, 1, 1, 1, 1, 100, 100, 100, 1],
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

