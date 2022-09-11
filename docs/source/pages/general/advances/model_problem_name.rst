Set Up Model's/Problem's Name
=============================

You don't really need to set the name for the optimizer and the problem. But it will help in saving results with the name of model and problem (especially in
multitask problems).


**1. Name the problem:**

.. code-block:: python

   from mealpy.swarm_based import PSO

   problem = {
      "fit_func": F5,
      "lb": [-3, -5, 1, -10, ],
      "ub": [5, 10, 100, 30, ],
      "minmax": "min",
      "name": "Benchmark Function 5th"
   }


**2. Name the optimizer model:**

.. code-block:: python

   model = PSO.OriginalPSO(epoch=10, pop_size=50, name="Normal PSO")
   model.solve(problem=problem)


**3. Get the name of problem and model**


.. code-block:: python

   print(model.name)            # Normal PSO
   print(model.problem.name)    # Benchmark Function 5th



.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4