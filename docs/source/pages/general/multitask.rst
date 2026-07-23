=================
Multitask Solving
=================

.. toctree::
   :maxdepth: 3

MEALPY provides a dedicated ``Multitask`` class designed to streamline large-scale experimental setups. Instead of writing boilerplate loops, you can use this class to effortlessly execute combinations of:

1. One algorithm solving one problem across multiple trials.
2. One algorithm solving multiple problems across multiple trials.
3. Multiple algorithms solving one problem across multiple trials.
4. Multiple algorithms solving multiple problems across multiple trials.

.. hint::
    **Explore the Examples Folder**

    For a deep dive into complex experiment configurations, please check out our official `Multitask-Examples <https://github.com/thieu1995/mealpy/tree/master/examples>`_ repository.

Simple Multitask Example
------------------------

In this example, we benchmark two algorithms (BBO and DE) with different variants against three standard CEC2017 benchmark functions.

.. code-block:: python

    import numpy as np
    from opfunu.cec_based.cec2017 import F52017, F102017, F292017
    from mealpy import FloatVar, BBO, DE, Multitask

    ## 1. Define your problems using Opfunu
    f1, f2, f3 = F52017(30, f_bias=0), F102017(30, f_bias=0), F292017(30, f_bias=0)

    p1 = {"bounds": FloatVar(lb=f1.lb, ub=f1.ub), "obj_func": f1.evaluate, "minmax": "min", "name": "F5"}
    p2 = {"bounds": FloatVar(lb=f2.lb, ub=f2.ub), "obj_func": f2.evaluate, "minmax": "min", "name": "F10"}
    p3 = {"bounds": FloatVar(lb=f3.lb, ub=f3.ub), "obj_func": f3.evaluate, "minmax": "min", "name": "F29"}

    ## 2. Define the optimizer models
    model1 = BBO.DevBBO(epoch=10000, pop_size=50)
    model2 = BBO.OriginalBBO(epoch=10000, pop_size=50)
    model3 = DE.OriginalDE(epoch=10000, pop_size=50)
    model4 = DE.SAP_DE(epoch=10000, pop_size=50)

    ## 3. Define termination criteria
    term = {"max_fe": 3000}

    ## 4. Initialize and execute Multitask
    if __name__ == "__main__":
        multitask = Multitask(
            algorithms=(model1, model2, model3, model4),
            problems=(p1, p2, p3),
            terminations=(term, ),
            modes=("single", )  # Default is "single"
        )

        # Execute 5 independent trials for all algorithm-problem combinations
        multitask.execute(
            n_trials=5,
            n_jobs=None,
            save_path="history",
            save_as="csv",
            save_convergence=True,
            verbose=False
        )

Advanced Configuration: Terminations & Modes
--------------------------------------------

When defining a ``Multitask`` object, you can pass a highly specific grid of ``terminations`` and ``modes`` (e.g., "thread", "process", "swarm", "single") for each optimizer solving each problem.

Assuming you have **3 optimizers** and **2 problems**, MEALPY intelligently broadcasts your inputs. Here is how you can configure the mapping:

.. code-block:: python

    # 1. Exact Mapping (List of Lists/Tuples)
    # Define a specific termination for every single algorithm-problem pair
    terminations = [ (term_11, term_12), (term_21, term_22), (term_31, term_32) ]

    # 2. Algorithm-Specific Mapping (Matches number of algorithms)
    # The same termination applies to all problems, mapped by optimizer
    terminations = [ term_1, term_2, term_3 ]
    # Internally converts to: [ (term_1, term_1), (term_2, term_2), (term_3, term_3) ]

    # 3. Problem-Specific Mapping (Matches number of problems)
    # The same termination applies to all optimizers, mapped by problem
    terminations = [ term_1, term_2 ]
    # Internally converts to: [ (term_1, term_2), (term_1, term_2), (term_1, term_2) ]

    # 4. Universal Mapping (Single element)
    # Applies exactly one termination rule to ALL algorithms and ALL problems
    terminations = [term]
    # Internally converts to: [ (term, term), (term, term), (term, term) ]

Execution & Parallel Trials
---------------------------

Once the ``Multitask`` object is created, call ``execute()`` to start the experiment. Two critical parameters dictate the workload:

* ``n_trials``: The number of independent runs for each (optimizer, problem) pair to ensure statistical significance.
* ``n_jobs``: The number of CPU processes used to run these *trials* in parallel.
    * ``n_jobs <= 1`` or ``None``: Runs trials sequentially.
    * ``n_jobs >= 2``: Spawns processes to run trials concurrently (e.g., ``n_jobs=4`` runs 4 trials simultaneously).

.. important::
    **Do Not Nest Parallelism (n_jobs vs modes)**

    Be extremely careful when mixing multiprocessing levels.

    * ``n_jobs`` parallelizes at the **Trial Level**.
    * ``modes="process"`` parallelizes at the **Agent Fitness Level** (inside the algorithm).

    If you set both ``n_jobs=4`` and ``modes=("process",)`` with ``n_workers=4``, your machine will attempt to spawn 4 × 4 = 16 intensive processes simultaneously. This can lead to severe CPU bottlenecking or memory crashes. **Rule of thumb: Choose only one level of parallelization.**
