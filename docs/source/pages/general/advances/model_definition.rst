Model Definition
================

**1. Name the optimizer model and name the fitness function:**

.. code-block:: python

   from mealpy.swarm_based import PSO

   problem = {
      "fit_func": F5,
      "lb": [-3, -5, 1, -10, ],
      "ub": [5, 10, 100, 30, ],
      "minmax": "min",
   }

   model = PSO.BasePSO(problem, epoch=10, pop_size=50, name="Normal PSO", fit_name="Benchmark Function 5th")

   print(model.name)
   print(model.fit_name)


**2. Set up Stopping Condition for an optimizer:**

.. code-block:: python

   term_dict = {
      "mode": "TB",
      "quantity": 60  # 60 seconds = 1 minute to run this algorithm only
   }

   model = PSO.BasePSO(problem, epoch=100, pop_size=50, termination=term_dict)
   model.solve()


**3. Hint Validation for setting up the hyper-parameters:**

If you don't know how to set up hyper-parameters and valid range for it. Try to set different type for that hyper-parameter.

.. code-block:: python

   model = PSO.BasePSO(problem, epoch="hello", pop_size="world")
   model.solve()

   # $ 2022/03/22 08:59:16 AM, ERROR, mealpy.utils.validator.Validator [line: 31]: 'epoch' is an integer and value should be in range: [1, 100000].

   model = PSO.BasePSO(problem, epoch=10, pop_size="world")
   model.solve()

   # $ 2022/03/22 09:01:51 AM, ERROR, mealpy.utils.validator.Validator [line: 31]: 'pop_size' is an integer and value should be in range: [10, 10000].



.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4