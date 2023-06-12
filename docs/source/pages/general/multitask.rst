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
        multitask = Multitask(algorithms=(model1, model2, model3), problems=(p1, p2, p3), terminations=(term, ), modes=("thread", ))
        # default modes = "single", default termination = epoch (as defined in problem dictionary)

        multitask.execute(n_trials=5, n_jobs=5, save_path="history", save_as="csv", save_convergence=True, verbose=False)
        # multitask.execute(n_trials=5, save_path="history", save_as="csv", save_convergence=True, verbose=False)


When define Multitask object, you can pass terminations and modes for each optimizer that solve each problem.
Assumption that we have 3 optimizers and 2 problems. Then terminations/modes can be as follows:

.. code-block:: python

	# Each optimizer for each problem with different termination
	terminations = [ (term_11, term_12), (term_21, term_22), (term_31, term_32) ]

	# The same termination for all problems with each optimizer
	terminations = [ term_1, term_2, term_3 ]
	## Then it will be convert into
	terminations = [ (term_1, term_1), (term_2, term_2), (term_3, term_3) ]

	# The same termination for all optimizers with each problem
	terminations = [ term_1, term_2 ]
	## Then it will be convert into
	terminations = [ (term_1, term_2), (term_1, term_2), (term_1, term_2) ]

	# The same termination for all optimizers and all problems
	terminations = [term]
	## Then it will be convert into
	terminations = [ (term, term), (term, term), (term, term) ]

Remember the modes variables here is the mode using in each optimizer, the value can be "thread", "process", "swarm" or "single".

After we have multitask object, we can call the function execute() to run it. There are two important parameters for this functions which are: n_trials and
n_jobs.

* n_trials: the number of time you want to run the set of (optimizers, problems) above.
* n_jobs: defined the number of processes will be used to speed up the computation for `n_trials`.
	* n_jobs <= 1 or None: we run `n_trials` in sequential order.
	* n_jobs >= 2: we use `n_jobs` processes to run `n_trials` task in parallel.

For example, we set n_trials = 10, and n_jobs = 4. Then at first, we create 4 processes to handle 4 trials simultaneously.


.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4