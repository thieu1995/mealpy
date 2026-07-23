Model's Parameters
==================

.. toctree::
   :maxdepth: 3


1. Hint Validation for Hyper-parameters
---------------------------------------

If you are unsure about the acceptable range or data type for a specific optimizer parameter, MEALPY has a built-in safety net. You can deliberately set it to an invalid value, and the optimizer's **Hint Validation** system will raise an informative error, guiding you on how to set it correctly.

.. code-block:: python

    from mealpy import PSO

    # Passing strings instead of integers
    model = PSO.OriginalPSO(epoch="hello", pop_size="world")
    model.solve(problem)

    # Console output:
    # 2022/03/22 08:59:16 AM, ERROR, mealpy.utils.validator.Validator [line: 31]: 'epoch' is an integer and value should be in range: [1, 100000].

    model = PSO.OriginalPSO(epoch=10, pop_size="world")
    model.solve(problem)

    # Console output:
    # 2022/03/22 09:01:51 AM, ERROR, mealpy.utils.validator.Validator [line: 31]: 'pop_size' is an integer and value should be in range: [10, 10000].

2. Setting Parameters via Dictionary
------------------------------------

Instead of passing arguments sequentially, you can define your hyperparameters in a standard Python dictionary and unpack them using the ``**kwargs`` syntax. This keeps your code clean and highly modular.

.. code-block:: python

    from mealpy import DE, FloatVar

    problem = {
        "obj_func": lambda x: sum(x**2),
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

.. hint::
    **Hyperparameter Tuning**

    The dictionary unpacking approach is exceptionally powerful when you want to tune your model's parameters using grid search libraries, such as ``ParameterGrid`` from **scikit-learn**.

.. code-block:: python

    from sklearn.model_selection import ParameterGrid
    from mealpy import DE, FloatVar

    problem = {
        "obj_func": lambda x: sum(x**2),
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

    # Iterate through all combinations in the grid
    for paras_de in list(ParameterGrid(paras_de_grid)):
        model = DE.OriginalDE(**paras_de)
        model.solve(problem)

3. Retrieving Model Parameters
------------------------------

If you need to log or inspect the configuration of an instantiated model, you can easily retrieve its hyper-parameters using the ``get_parameters()`` method.

.. code-block:: python

    # Returns the model's parameters as a Python dictionary
    params_dict = model.get_parameters()          

    # If you want to log or print it, convert it to a string using the built-in str() function
    params_str = str(model.get_parameters())
