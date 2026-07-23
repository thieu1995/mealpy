Setting Model and Problem Names
===============================

.. toctree::
   :maxdepth: 3


While it is not strictly required to assign custom names to your optimizer and problem, doing so is considered a best practice. It significantly helps in tracking, logging, and organizing results, especially when conducting multi-task experiments or benchmarking various algorithms.

.. hint::
    **Better Data Exporting and Visualization**

    When exporting your results to DataFrames, CSV/JSON files, or generating visual plots using MEALPY's visualization tools, the library will automatically utilize these defined names. This prevents confusion when analyzing large batches of experimental data!

1. Naming the Problem
---------------------

You can assign a name to your problem directly within the problem dictionary by adding the ``name`` keyword. *(Note: If you are defining a custom child class of the Problem class, you can pass this via the ``name`` parameter in the initialization).*

.. code-block:: python

    from mealpy import FloatVar

    # Define a dummy objective function
    def F5(solution):
        return sum(x**2 for x in solution)

    problem = {
        "obj_func": F5,
        "bounds": FloatVar(lb=[-3, -5, 1, -10], ub=[5, 10, 100, 30]),
        "minmax": "min",
        "name": "Benchmark Function 5th"   # Set the problem's custom name here
    }

2. Naming the Optimizer Model
-----------------------------

Similarly, you can assign a specific name to your optimizer during initialization by passing the ``name`` argument. This is exceptionally useful when you are testing the same algorithm with different hyperparameters.

.. code-block:: python

    from mealpy import PSO

    model = PSO.OriginalPSO(epoch=10, pop_size=50, name="Normal PSO")
    model.solve(problem=problem)

3. Retrieving the Names
-----------------------

Once instantiated, you can dynamically access these names programmatically through the model object's attributes.

.. code-block:: python

    print(model.name)            # Output: Normal PSO
    print(model.problem.name)    # Output: Benchmark Function 5th
