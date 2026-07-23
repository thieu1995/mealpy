Saving and Loading Model
========================

.. toctree::
   :maxdepth: 3


From the above tutorials, you learned that you can track the population's history after each epoch by setting ``"save_population": True`` in the problem dictionary.

.. warning::
    **In-Memory vs. Disk Storage**

    Setting ``save_population: True`` strictly means storing the population of each epoch inside the model's ``history`` object **in your system's RAM**. As warned previously, if your problem size or epoch count is too large, this will cause memory overflow issues. **It does NOT save the model to a file.**

To physically save your trained optimizer to a file on your local machine (and load it back later), you must use the ``io`` module from ``mealpy.utils``. 

.. important::
    **Pickle Serialization**

    The ``io.save_model()`` function utilizes Python's built-in pickle serialization. This allows you to completely freeze the optimizer's state, export it as a ``.pkl`` file, and resume or analyze your model in an entirely different Python session.

Code Example: Exporting and Importing
-------------------------------------

.. code-block:: python
    :emphasize-lines: 3, 21, 24

    import numpy as np
    from mealpy import GA, FloatVar
    from mealpy.utils import io

    def objective_function(solution):
        return np.sum(solution**2)

    problem = {
        "obj_func": objective_function,
        "bounds": FloatVar(lb=[-100, ] * 50, ub=[100, ] * 50),
        "minmax": "min",
    }

    ## 1. Run the algorithm
    model = GA.BaseGA(epoch=100, pop_size=50)
    g_best = model.solve(problem)
    print(f"Best solution: {g_best.solution}, Best fitness: {g_best.target.fitness}")

    ## 2. Save the fully trained model to a file
    # Note: Ensure the "results" directory exists or provide a valid path
    io.save_model(model, "results/model.pkl")

    ## 3. Load the model from the file in a new session
    optimizer = io.load_model("results/model.pkl")
    print(f"Loaded solution: {optimizer.g_best.solution}, Loaded fitness: {optimizer.g_best.target.fitness}")
