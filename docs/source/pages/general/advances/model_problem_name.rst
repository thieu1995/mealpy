Set Up Model's/Problem's Name
=============================

You do not necessarily need to set names for the optimizer and the problem, but doing so can help in saving results with the names of the model and the
problem, especially in multitask problems.


**1. Name the problem:**

.. code-block:: python

   from mealpy.swarm_based import PSO

   problem = {
      "fit_func": F5,
      "bounds": FloatVar(lb=[-3, -5, 1, -10, ], ub=[5, 10, 100, 30, ]),
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