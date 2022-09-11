Model's Parameters
==================

**1. Hint Validation for setting up the hyper-parameters:**

In case you don't know how to set up the parameter for the optimizer. You can try to set that parameter to anything you want.
It will show a "hint validation" that will help you how to set valid parameters.

.. code-block:: python

   model = PSO.OriginalPSO(epoch="hello", pop_size="world")
   model.solve(problem)

   # $ 2022/03/22 08:59:16 AM, ERROR, mealpy.utils.validator.Validator [line: 31]: 'epoch' is an integer and value should be in range: [1, 100000].

   model = PSO.OriginalPSO(epoch=10, pop_size="world")
   model.solve(problem)

   # $ 2022/03/22 09:01:51 AM, ERROR, mealpy.utils.validator.Validator [line: 31]: 'pop_size' is an integer and value should be in range: [10, 10000].



**2. Set up model's parameters as a dictionary:**

.. code-block:: python

   from mealpy.evolutionary_based import DE

   problem = {
      "fit_func": F5,
      "lb": [-10]*10,
      "ub": [30]*10,
      "minmax": "min",
   }

   paras_de = {
      "epoch": 20,
      "pop_size": 50,
      "wf": 0.7,
      "cr": 0.9,
      "strategy": 0,
   }

   model = DE.BaseDE(**paras_de)
   model.solve(problem)


This will definitely be helpful when using ParameterGrid/GridSearchCV from the scikit-learn library to tune the parameter of the models. For example;


.. code-block:: python

	from sklearn.model_selection import ParameterGrid
	from mealpy.evolutionary_based import DE

	problem = {
		"fit_func": F5,
		"lb": [-10]*10,
		"ub": [30]*10,
		"minmax": "min",
	}

	paras_de_grid = {
		"epoch": [100, 200, 300, 500, 1000],
		"pop_size": [50, 100],
		"wf": [0.5, 0.6, 0.7, 0.8, 0.9],
		"cr": [0.6, 0.7, 0.8, 0.9],
		"strategy": [0, 1, 2, 3, 4],
	}

	for paras_de in list(ParameterGrid(paras_de_grid)):
		model = DE.BaseDE(**paras_de)
		model.solve(problem)


**3. Get the parameters of the model**

Using this method below will return model's parameters as a python dictionary.
If you want to convert it to string, we recommend to use built-in python method: str

.. code-block:: python

	model.get_parameters()          # Return dictionary

	str(model.get_parameter())      # Return a string


.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4
