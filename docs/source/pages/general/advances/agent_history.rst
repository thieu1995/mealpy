Agent's History (Trajectory)
============================

.. toctree::
   :maxdepth: 3


By default, MEALPY tracks essential metrics (like best/worst fitness, runtime, and diversity) across generations without saving the complete state of every agent. This ensures high performance and low memory consumption. 

However, if you want to visualize the trajectory chart of search agents, you can enable full population tracking by setting the ``save_population`` keyword to ``True`` in the problem definition.

.. warning::
    **Memory Out-of-Bounds Risk**

    Enabling ``save_population: True`` will save the full position and fitness of every single agent in every single generation. For large population sizes, high dimensions, or a massive number of epochs, this will rapidly consume your RAM and may crash your program. **Only enable this for small-scale problems or when strictly necessary for trajectory visualization.**

Enabling Population Tracking
----------------------------

.. code-block:: python

    from mealpy import FloatVar

    problem_dict1 = {
        "obj_func": lambda x: sum(x**2),
        "bounds": FloatVar(lb=[-3, -5, 1, -10], ub=[5, 10, 100, 30]),
        "minmax": "min",
        "log_to": "console",
        "save_population": True,  # Set to True to enable trajectory tracking. Default is False.
    }

Accessing the History Object
----------------------------

After the optimization process finishes, you can access the detailed history of the agents and the population through the ``model.history`` object. It contains the following built-in lists:

* **Solutions (Agents):**
    * ``list_global_best``: The global best SOLUTION found so far across all previous generations.
    * ``list_current_best``: The best SOLUTION in each specific generation.
    * ``list_global_worst``: The global worst SOLUTION found so far across all previous generations.
    * ``list_current_worst``: The worst SOLUTION in each specific generation.

* **Fitness Values:**
    * ``list_global_best_fit``: The global best FITNESS found so far across all previous generations.
    * ``list_current_best_fit``: The best FITNESS in each specific generation.

* **Metrics & Analytics:**
    * ``list_epoch_time``: The runtime (in seconds) for each generation.
    * ``list_diversity``: The spatial diversity of the swarm across generations.
    * ``list_exploitation``: The exploitation percentage metric for each generation.
    * ``list_exploration``: The exploration percentage metric for each generation.

* **Full Trajectory:**
    * ``list_population``: The complete POPULATION array in each generation. *(Only populated if ``save_population=True``)*

Example: Retrieving History Data
--------------------------------

.. code-block:: python

    import numpy as np
    from mealpy import PSO, FloatVar

    def objective_function(solution):
        return np.sum(solution**2)

    problem_dict = {
        "obj_func": objective_function,
        "bounds": FloatVar(lb=[-3, -5, 1, -10], ub=[5, 10, 100, 30]),
        "minmax": "min",
        "verbose": True,
        "save_population": False  # You cannot draw trajectory charts with this set to False
    }
    
    model = PSO.OriginalPSO(epoch=1000, pop_size=50)
    model.solve(problem=problem_dict)

    # Accessing essential metrics (Available even if save_population=False)
    print("Global best solutions:", model.history.list_global_best)
    print("Epoch runtime history:", model.history.list_epoch_time)
    print("Global best fitness history:", model.history.list_global_best_fit)
    print("Exploration history:", model.history.list_exploration)

    # Accessing full population history
    # NOTE: This will be empty or missing because save_population was set to False
    print("Full population history:", model.history.list_population)
