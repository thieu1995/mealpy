=================
Multitask Solving
=================

We build a dedicated class, Multitask, that can help you run several different scenarios. For example:

1. Run 1 algorithm with 1 problem, and multiple trials
2. Run 1 algorithm with multiple problems, and multiple trials
3. Run multiple algorithms with 1 problem, and multiple trials
4. Run multiple algorithms with multiple problems, and multiple trials

Please head to examples folder to learn more about this `Multitask-Examples`_

.. _Multitask-Examples: https://github.com/thieu1995/mealpy/tree/master/examples


**Below is a simple example with Multitask class**

.. code-block:: python

	import numpy as np
	#### Using multiple algorithm to solve multiple problems with multiple trials

	## Import libraries

	from opfunu.cec_based.cec2017 import F52017, F102017, F292017
	from mealpy.bio_based import BBO
	from mealpy.evolutionary_based import DE
	from mealpy.multitask import Multitask          # Remember this


	## Define your own problems

	f1 = F52017(30, f_bias=0)
	f2 = F102017(30, f_bias=0)
	f3 = F292017(30, f_bias=0)

	p1 = {
	    "lb": f1.lb.tolist(),
	    "ub": f1.ub.tolist(),
	    "minmax": "min",
	    "fit_func": f1.evaluate,
	    "name": "F5",
	    "log_to": None,
	}

	p2 = {
	    "lb": f2.lb.tolist(),
	    "ub": f2.ub.tolist(),
	    "minmax": "min",
	    "fit_func": f2.evaluate,
	    "name": "F10",
	    "log_to": None,
	}

	p3 = {
	    "lb": f3.lb.tolist(),
	    "ub": f3.ub.tolist(),
	    "minmax": "min",
	    "fit_func": f3.evaluate,
	    "name": "F29",
	    "log_to": None,
	}

	## Define models

	model1 = BBO.BaseBBO(epoch=10, pop_size=50)
	model2 = BBO.OriginalBBO(epoch=10, pop_size=50)
	model3 = DE.BaseDE(epoch=10, pop_size=50)


	## Define and run Multitask

	if __name__ == "__main__":
	    multitask = Multitask(algorithms=(model1, model2, model3), problems=(p1, p2, p3))
	    multitask.execute(n_trials=3, mode="parallel", n_workers=4, save_path="history", save_as="csv", save_convergence=True, verbose=True)





.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4