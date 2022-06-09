==================
Guide to use model
==================

In this phase, the main task is to find out the global optimal - in this project, we call it named *model* for simple. We designed the classical as well as
the state-of-the-art population-based meta-heuristics models: `Evolutionary-based`_, `Swarm-based`_, `Physics-based`_, `Human-based`_, `Biology-based`_,
`Mathematical-based`_, `Musical-based`_

.. _Evolutionary-based: ../models/mealpy.evolutionary_based.html
.. _Swarm-based: ../models/mealpy.swarm_based.html
.. _Physics-based: ../models/mealpy.physics_based.html
.. _Human-based: ../models/mealpy.human_based.html
.. _Biology-based: ../models/mealpy.bio_based.html
.. _Mathematical-based: ../models/mealpy.math_based.html
.. _Musical-based: ../models/mealpy.music_based.html

All of this methods are used in the same way. So, in this guide, we'll demo with a specific method such as **Genetic Algorithm** in *Evolutionary-based*.


------------
Installation
------------

**Dependencies**

To use the library, your computer must installed all of these packages first:

- Python (>= 3.6)
- Numpy (>= 1.15.1)
- Matplotlib (>=3.1.3)
- Scipy (>= 1.5.2)


**User Installation**

- Install the [current PyPI release](https://pypi.python.org/pypi/mealpy):

::

   $ pip uninstall mealpy
   $ pip install mealpy==2.4.1

- Or install the development version from GitHub:

::

   $ pip install git+https://github.com/thieu1995/mealpy


I accidentally deleted version 2.1.1 on Pypi since it's not synced with version 2.1.1 on Github Release.
If you still want to use version 2.1.1. Please use this command:

::

   $ pip install -e git+https://github.com/thieu1995/mealpy@ead414d2d9aa5317864e779fa5d4ad7b65159181#egg=mealpy


----------------------
Getting started in 30s
----------------------

**Tutorial**
	* Import libraries
	* Define your fitness function
	* Define a problem dictionary
	* Training and get the results

.. code-block:: python

	from mealpy.evolutionary_based import GA
	import numpy as np

	def fitness_func(solution):
	    return np.sum(solution**2)

	problem_dict = {
	    "fit_func": fitness_func,
	    "lb": [-100, ] * 30,
	    "ub": [100, ] * 30,
	    "minmax": "min",
	    "log_to": "file",
	    "log_file": "result.log"
	}

	ga_model = GA.BaseGA(problem_dict, epoch=100, pop_size=50, pc=0.85, pm=0.1)
	best_position, best_fitness_value = ga_model.solve()

	print(best_position)
	print(best_fitness_value)

You can see the error after each iteration which is found by GA:

----------------------------
Fitness Function Preparation
----------------------------
Make sure that your designed Fitness Function take an solution (a numpy vector) and return the fitness value (single real value or list of real value)

We have already included the library *opfunu* which is a framework of benchmark functions for optimization problems. You can use it in the very easy way by:

.. code-block:: python

	from opfunu.type_based.uni_modal import Functions           # or
	from opfunu.cec.cec2014 import Fucntion                     # or
	from opfunu.dimension_based.benchmarknd import Functions

	# Then you need to create an object of Function to get the functions
	type_based = Functions()
	F1 = type_based._sum_squres__
	F2 = type_based.__dixon_price__
	....

But if you don't want to use it, you want to design your own fitness functions. It is ok, all you need to do is write your own function with input is a
numpy vector (the solution) and output is the single objective value or list of multiple objective values.

.. code-block:: python

	import numpy as np

	## This is normal fitness function
	def fitness_normal(solution=None):
		return np.sqrt(solution**2)         # Single value


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


	## This is how you design Constrained Benchmark Function (G01)
	#### Link: https://onlinelibrary.wiley.com/doi/pdf/10.1002/9781119136507.app2
	def fitness_constrained(solution):
		def g1(x):
	        return 2 * x[0] + 2 * x[1] + x[9] + x[10] - 10
	    def g2(x):
	        return 2 * x[0] + 2 * x[2] + x[9] + x[10] - 10
	    def g3(x):
	        return 2 * x[1] + 2 * x[2] + x[10] + x[11] - 10
	    def g4(x):
	        return -8 * x[0] + x[9]
	    def g5(x):
	        return -8 * x[1] + x[10]
	    def g6(x):
	        return -8 * x[2] + x[11]
	    def g7(x):
	        return -2 * x[3] - x[4] + x[9]
	    def g8(x):
	        return -2 * x[5] - x[6] + x[10]
	    def g9(x):
	        return -2 * x[7] - x[8] + x[11]

	    def violate(value):
	        return 0 if value <= 0 else value

	    fx = 5 * np.sum(solution[:4]) - 5 * np.sum(solution[:4] ** 2) - np.sum(solution[4:13])

	    ## Increase the punishment for g1 and g4 to boost the algorithm (You can choice any constraint instead of g1 and g4)
	    fx += violate(g1(solution)) ** 2 + violate(g2(solution)) + violate(g3(solution)) + \
	        2 * violate(g4(solution)) + violate(g5(solution)) + violate(g6(solution)) + \
	        violate(g7(solution)) + violate(g8(solution)) + violate(g9(solution))
	    return fx

-------------------
Problem Preparation
-------------------

You will need to define a problem dictionary with must has keywords ("fit_func", "lb", "ub", "minmax"). For special case, when you are trying to
solve **multiple objective functions**, you need another keyword **"obj_weights"**:

	* fit_func: Your fitness function
	* lb: Lower bound of variables, it should be list of values
	* ub: Upper bound of variables, it should be list of values
	* minmax: The problem you are trying to solve is minimum or maximum, value can be "min" or "max"
	* obj_weights: list weights for all your objectives (Optional, default = [1, 1, ...1])


.. code-block:: python

	## Design a problem dictionary for normal function
	problem_normal = {
	    "fit_func": fitness_normal,
	    "lb": [-100, ] * 30,
	    "ub": [100, ] * 30,
	    "minmax": "min",
	}

	## Design a problem dictionary for multiple objective functions above
	problem_multi = {
	    "fit_func": fitness_multi,
	    "lb": [-10, -10],
	    "ub": [10, 10],
	    "minmax": "min",
	    "obj_weights": [0.4, 0.1, 0.5]               # Define it or default value will be [1, 1, 1]
	}

	## Design a problem dictionary for constrained objective function above
	problem_constrained = {
	  "fit_func": fitness_constrained,
	  "lb": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	  "ub": [1, 1, 1, 1, 1, 1, 1, 1, 1, 100, 100, 100, 1],
	  "minmax": "min",
	}


--------
Training
--------

Start learning by call function **solve()**. There are 4 different training modes include:

1. process: Using multi-cores to update fitness for whole population (Parallel: no effect on updating process)
2. thread: Using multi-threads to update fitness for whole population (Parallel: no effect on updating process)
3. swarm: Updating fitness after the whole population move (Sequential: no effect on updating process)
4. single: Updating fitness after each agent move (Sequential: effect on updating process)


.. code-block:: python

	## Need to import the algorithm that will be used
	from mealpy.bio_based import SMA
	from mealpy.evolutionary_based import GA
	from mealpy.swarm_based import PSO

	sma_model = SMA.BaseSMA(problem_normal, epoch=100, pop_size=50, pr=0.03)
	best_position, best_fitness_value = sma_model.solve()   # default is: single

	sma_model = SMA.BaseSMA(problem_normal, epoch=100, pop_size=50, pr=0.03)
	best_position, best_fitness_value = sma_model.solve(mode="single")

	sma_model = SMA.BaseSMA(problem_normal, epoch=100, pop_size=50, pr=0.03)
	best_position, best_fitness_value = sma_model.solve(mode="swarm")

	ga_model = GA.BaseGA(problem_multi, epoch=1000, pop_size=100, pc=0.9, pm=0.05)
	best_position, best_fitness_value = ga_model.solve(mode="thread")

	pso_model = PSO.BasePSO(problem_constrained, epoch=500, pop_size=80, c1=2.0, c2=1.8, w_min=0.3, w_max=0.8)
	best_position, best_fitness_value = pso_model.solve(mode="process")


The returned results are 2 values :

- best_position: the global best position it found on training process
- best_fitness_value: the global best fitness value


--------
Advances
--------

.. include:: advances/lower_upper_bound.rst
.. include:: advances/termination.rst
.. include:: advances/problem_preparation.rst
.. include:: advances/model_definition.rst
.. include:: advances/starting_positions.rst
.. include:: advances/agent_history.rst
.. include:: advances/import_all_models.rst

.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4