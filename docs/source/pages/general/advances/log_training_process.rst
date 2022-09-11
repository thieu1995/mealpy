Log Training Process
====================

We currently provide three logging options: print out training process on console, log training process to file, and don't show or save log process.

* By default, if you don't declare "log_to" keyword in problem, it will log to "console"

.. code-block:: python

   problem_dict1 = {
      "fit_func": F5,
      "lb": [-3, -5, 1, -10, ],
      "ub": [5, 10, 100, 30, ],
      "minmax": "min",
      # Default = "console"
   }

   problem_dict1 = {
      "fit_func": F5,
      "lb": [-3, -5, 1, -10, ],
      "ub": [5, 10, 100, 30, ],
      "minmax": "min",
      "log_to": "console",
   }


* If you want to log to the file, you need an additional keyword "log_file",

.. code-block:: python

   problem_dict2 = {
      "fit_func": F5,
      "lb": [-3, -5, 1, -10, ],
      "ub": [5, 10, 100, 30, ],
      "minmax": "min",
      "log_to": "file",
      "log_file": "result.log",         # Default value = "mealpy.log"
   }


* Set it to None if you don't want to log

.. code-block:: python

   problem_dict3 = {
      "fit_func": F5,
      "lb": [-3, -5, 1, -10, ],
      "ub": [5, 10, 100, 30, ],
      "minmax": "min",
      "log_to": None,
   }



.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4

