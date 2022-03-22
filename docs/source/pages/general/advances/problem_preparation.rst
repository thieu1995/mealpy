Problem Preparation
===================

**1. WARNING: The memory issues related to mealpy:**

By default, the history of population is saved. This can cause the memory issues if your problem is too big.
You can set "save_population" = False to avoid this. However, you won't be able to draw the trajectory chart of agents.

.. code-block:: python

   problem_dict1 = {
      "fit_func": F5,
      "lb": [-3, -5, 1, -10, ],
      "ub": [5, 10, 100, 30, ],
      "minmax": "min",
      "log_to": "console",
      "save_population": False,              # Default = True
   }


**2. Logging results of training process:**

* 3 options:
	+ Log to console (default): problem_dict1
	+ Log to file: problem_dict2
	+ Don't show log: problem_dict3

.. code-block:: python

   problem_dict1 = {
      "fit_func": F5,
      "lb": [-3, -5, 1, -10, ],
      "ub": [5, 10, 100, 30, ],
      "minmax": "min",
      "log_to": "console",              # Default
   }

   problem_dict2 = {
      "fit_func": F5,
      "lb": [-3, -5, 1, -10, ],
      "ub": [5, 10, 100, 30, ],
      "minmax": "min",
      "log_to": "file",
      "log_file": "result.log",         # Default value = "mealpy.log"
   }

   problem_dict3 = {
      "fit_func": F5,
      "lb": [-3, -5, 1, -10, ],
      "ub": [5, 10, 100, 30, ],
      "minmax": "min",
      "log_to": None,
   }


**3. Set up necessary functions for discrete problem:**

Let's say we want to solve Travelling Salesman Problem (TSP), we need to design at least a function that generate solution, a function that
bring back the solution to the boundary, and finally the fitness function


.. code-block:: python

	import numpy as np


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

	def fitness_function(solution):
	    ## Objective for this problem is the sum of distance between all cities that salesman has passed
	    ## This can be change depend on your requirements
	    city_coord = CITY_POSITIONS[solution]
	    line_x = city_coord[:, 0]
	    line_y = city_coord[:, 1]
	    total_distance = np.sum(np.sqrt(np.square(np.diff(line_x)) + np.square(np.diff(line_y))))
	    return total_distance

	problem = {
	    "fit_func": fitness_function,
	    "lb": LB,
	    "ub": UB,
	    "minmax": "min",
	    "log_to": "console",
	    "generate_position": generate_position,
	    "amend_position": amend_position,
	}


.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4

