Stopping Condition (Termination)
================================

By default, when create an optimizer, the default stopping condition (termination) is epochs (generations, iterations)
But there are different stopping condition you can try by creating an Termination dictionary. There are 4 termination cases:

**1. FE (Number of Function Evaluation)**

.. code-block:: python

   term_dict1 = {
      "mode": "FE",
      "quantity": 100000    # 100000 number of function evaluation
   }

**2. MG (Maximum Generations / Epochs): This is default in all algorithms**

.. code-block:: python

   term_dict2 = {  # When creating this object, it will override the default epoch you define in your model
      "mode": "MG",
      "quantity": 1000  # 1000 epochs
   }

**3. ES (Early Stopping): Same idea in training neural network (If the global best solution not better an epsilon after K epoch then stop the program**

.. code-block:: python

   term_dict3 = {
      "mode": "ES",
      "quantity": 30  # after 30 epochs, if the global best doesn't improve then we stop the program
   }

**4. TB (Time Bound): You just want your algorithm run in K seconds. Especially when comparing different algorithms**

.. code-block:: python

   term_dict4 = {
      "mode": "TB",
      "quantity": 60  # 60 seconds = 1 minute to run this algorithm only
   }

**After import and create a termination object, and an optimizer object, you can pass termination object to solve() function**

.. code-block:: python

   model3 = SMA.BaseSMA(epoch=100, pop_size=50, pr=0.03)
   model3.solve(problem_dict1, termination=term_dict4)

.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4

