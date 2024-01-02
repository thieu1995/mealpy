Model's Parameters
==================

**1. Hint Validation for setting up the hyper-parameters:**

If you are unsure how to set up a parameter for the optimizer, you can try setting it to any value. The optimizer will then provide a "hint validation" that
can help you determine how to set valid parameters.


.. code-block:: python

   model = PSO.OriginalPSO(epoch="hello", pop_size="world")
   model.solve(problem)

   # $ 2022/03/22 08:59:16 AM, ERROR, mealpy.utils.validator.Validator [line: 31]: 'epoch' is an integer and value should be in range: [1, 100000].

   model = PSO.OriginalPSO(epoch=10, pop_size="world")
   model.solve(problem)

   # $ 2022/03/22 09:01:51 AM, ERROR, mealpy.utils.validator.Validator [line: 31]: 'pop_size' is an integer and value should be in range: [10, 10000].



**2. Set up model's parameters as a dictionary:**

.. code-block:: python

   from mealpy import DE, FloatVar

   problem = {
      "obj_func": F5,
      "bounds": FloatVar(lb=[-10,]*10, ub=[30,]*10),
      "minmax": "min",
   }

   paras_de = {
      "epoch": 20,
      "pop_size": 50,
      "wf": 0.7,
      "cr": 0.9,
      "strategy": 0,
   }

   model = DE.OriginalDE(**paras_de)
   model.solve(problem)


This will definitely be helpful when using ParameterGrid/GridSearchCV from the scikit-learn library to tune the parameter of the models. For example;


.. code-block:: python

	from sklearn.model_selection import ParameterGrid
	from mealpy import DE, FloatVar

	problem = {
		"obj_func": F5,
		"bounds": FloatVar(lb=[-10,]*10, ub=[30,]*10),
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
		model = DE.OriginalDE(**paras_de)
		model.solve(problem)


**3. Get the parameters of the model**

Using the method below will return the model's parameters as a Python dictionary. If you want to convert it to a string,  we recommend using the built-in
Python method: str().

.. code-block:: python

	model.get_parameters()          # Return dictionary

	str(model.get_parameter())      # Return a string


.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4
