Lower/Upper Bound
=================

There are a few ways to set up the lower bound (LB) and upper bound (UB). It is also depend on the values of LB and UB


**1. When you have different lower bound and upper bound for each parameters**

.. code-block:: python

   problem_dict1 = {
      "fit_func": F5,
      "lb": [-3, -5, 1, -10, ],
      "ub": [5, 10, 100, 30, ],
      "minmax": "min",
      "verbose": True,
   }

**2. When you have same lower bound and upper bound for each variable, then you can use:**

.. code-block:: python

   ## 2.1 number: then you need to specify your problem size / number of dimensions (n_dims)
   problem_dict2 = {
      "fit_func": F5,
      "lb": -10,
      "ub": 30,
      "minmax": "min",
      "verbose": True,
      "n_dims": 30,  # Remember the keyword "n_dims"
   }

   ## 2.2 array: Then there are 2 ways
   problem_dict3 = {
      "fit_func": F5,
      "lb": [-5],
      "ub": [10],
      "minmax": "min",
      "verbose": True,
      "n_dims": 30,  # Remember the keyword "n_dims"
   }

   ## or
   n_dims = 100
   problem_dict4 = {
      "fit_func": F5,
      "lb": [-5] * n_dims,
      "ub": [10] * n_dims,
      "minmax": "min",
      "verbose": True,
   }


.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4
