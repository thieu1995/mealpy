============
Simple Guide
============

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

**User Installation**

Install the [current PyPI release](https://pypi.python.org/pypi/mealpy):

::

   $ pip install mealpy==2.5.0

Or install the development version from GitHub:

::

   $ pip install git+https://github.com/thieu1995/mealpy


Check the version of MEALPY:

::

   $ import mealpy
   $ mealpy.__version__

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
	}

	ga_model = GA.BaseGA(epoch=100, pop_size=50, pc=0.85, pm=0.1)
	best_position, best_fitness_value = ga_model.solve(problem_dict)

	print(best_position)
	print(best_fitness_value)

You can see the error after each iteration which is found by GA:

----------------------------
Fitness Function Preparation
----------------------------

Make sure that your designed fitness function takes a solution (a numpy vector) and returns the fitness value (a single real value or a list of real values).

We have already included the *opfunu* library, which is a framework of benchmark functions for optimization problems. You can use it very easily by:


.. code-block:: python

	from opfunu.type_based.uni_modal import Functions           # or
	from opfunu.cec.cec2014 import Fucntion                     # or
	from opfunu.dimension_based.benchmarknd import Functions

	# Then you need to create an object of Function to get the functions
	type_based = Functions()
	F1 = type_based._sum_squres__
	F2 = type_based.__dixon_price__
	....


If you prefer not to use the opfunu library and want to design your own fitness functions, that's okay too. All you need to do is write your own function
that takes a numpy vector (the solution) as input and returns a single objective value or a list of multiple objective values.


.. code-block:: python

	import numpy as np

	## This is normal fitness function
	def fitness_normal(solution=None):
		return np.sqrt(solution**2)         # Single value

-------------------
Problem Preparation
-------------------

You will need to define a problem dictionary with must has keywords ("fit_func", "lb", "ub", "minmax").

	* fit_func: Your fitness function
	* lb: Lower bound of variables, it should be list of values
	* ub: Upper bound of variables, it should be list of values
	* minmax: The problem you are trying to solve is minimum or maximum, value can be "min" or "max"


.. code-block:: python

	## Design a problem dictionary for normal function
	problem_normal = {
	    "fit_func": fitness_normal,
	    "lb": [-100, ] * 30,
	    "ub": [100, ] * 30,
	    "minmax": "min",
	}

--------
Training
--------

To start learning, call the **solve()** function. There are four different training modes available:

1. **process**: Uses multiple cores to update fitness for the entire population (parallel processing; no effect on updating process).
2. **thread**: Uses multiple threads to update fitness for the entire population (parallel processing; no effect on updating process).
3. **swarm**: Updates fitness after the entire population moves (sequential processing; no effect on updating process).
4. **single**: Updates fitness after each agent moves (sequential processing; affects updating process).



.. code-block:: python

	## Need to import the algorithm that will be used
	from mealpy.bio_based import SMA
	from mealpy.evolutionary_based import GA
	from mealpy.swarm_based import PSO

	sma_model = SMA.BaseSMA(epoch=100, pop_size=50, pr=0.03)
	best_position, best_fitness_value = sma_model.solve(problem_normal)   # default is: single

	sma_model = SMA.BaseSMA(epoch=100, pop_size=50, pr=0.03)
	best_position, best_fitness_value = sma_model.solve(problem_normal, mode="single")

	sma_model = SMA.BaseSMA(epoch=100, pop_size=50, pr=0.03)
	best_position, best_fitness_value = sma_model.solve(problem_normal, mode="swarm")

	ga_model = GA.BaseGA(epoch=1000, pop_size=100, pc=0.9, pm=0.05)
	best_position, best_fitness_value = ga_model.solve(problem_multi, mode="thread")

	pso_model = PSO.OriginalPSO(epoch=500, pop_size=80, c1=2.0, c2=1.8, w_min=0.3, w_max=0.8)
	best_position, best_fitness_value = pso_model.solve(problem_constrained, mode="process")



You can set the number of workers when using "Parallel" training.

.. code-block:: python

	from mealpy.bio_based import SMA

	sma_model = SMA.BaseSMA(epoch=100, pop_size=50, pr=0.03)
	best_position, best_fitness_value = sma_model.solve(problem_normal, mode="thread", n_workers=8)
	# Using 8 threads to solve this problem


The returned results are 2 values :

- best_position: the global best position it found on training process
- best_fitness_value: the global best fitness value


.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4