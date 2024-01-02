Log Training Process
====================

We currently offer three logging options: printing the training process on the console, logging the training process to a file, and not displaying or saving
the log process.

* By default, if the "log_to" keyword is not declared in the problem dictionary, it will be logged to the console.

.. code-block:: python

   problem_dict1 = {
      "obj_func": F5,
      "bounds": FloatVar(lb=[-3, -5, 1, -10, ], ub=[5, 10, 100, 30, ]),
      "minmax": "min",
      # Default = "console"
   }

   problem_dict1 = {
      "obj_func": F5,
      "bounds": FloatVar(lb=[-3, -5, 1, -10, ], ub=[5, 10, 100, 30, ]),
      "minmax": "min",
      "log_to": "console",
   }


* If you want to log to the file, you need an additional keyword "log_file",

.. code-block:: python

   problem_dict2 = {
      "obj_func": F5,
      "bounds": FloatVar(lb=[-3, -5, 1, -10, ], ub=[5, 10, 100, 30, ]),
      "minmax": "min",
      "log_to": "file",
      "log_file": "result.log",         # Default value = "mealpy.log"
   }


* Set it to None if you don't want to log

.. code-block:: python

   problem_dict3 = {
      "obj_func": F5,
      "bounds": FloatVar(lb=[-3, -5, 1, -10, ], ub=[5, 10, 100, 30, ]),
      "minmax": "min",
      "log_to": None,
   }



.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4

