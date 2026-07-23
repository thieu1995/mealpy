Log Training Process
====================

.. toctree::
   :maxdepth: 3


MEALPY offers flexible logging options to help you monitor the optimization process. You can choose to print the progress to the console, save it to a log file, or disable logging entirely to keep your output clean.

.. note::
    **Default Behavior**

    If the ``log_to`` keyword is not specified in your problem definition, MEALPY will automatically default to logging the training progress directly to the **console**.

.. important::
    **Logging to a File**

    When setting ``log_to="file"``, you should specify the target destination using the ``log_file`` keyword (e.g., ``"result.log"``). If you omit ``log_file``, MEALPY will automatically save the output to a default file named ``mealpy.log``.

Here is how you can configure each of the three available logging modes:

.. code-block:: python

    from mealpy import FloatVar

    # Option 1: Log to Console (Default)
    problem_console = {
        "obj_func": F5,
        "bounds": FloatVar(lb=[-3, -5, 1, -10], ub=[5, 10, 100, 30]),
        "minmax": "min",
        "log_to": "console",       # Optional: This is the default behavior
    }

    # Option 2: Log to File
    problem_file = {
        "obj_func": F5,
        "bounds": FloatVar(lb=[-3, -5, 1, -10], ub=[5, 10, 100, 30]),
        "minmax": "min",
        "log_to": "file",
        "log_file": "result.log",  # Optional: Defaults to "mealpy.log" if not provided
    }

    # Option 3: Disable Logging (Silent Mode)
    problem_silent = {
        "obj_func": F5,
        "bounds": FloatVar(lb=[-3, -5, 1, -10], ub=[5, 10, 100, 30]),
        "minmax": "min",
        "log_to": None,            # Set to None for no console output and no file generation
    }
