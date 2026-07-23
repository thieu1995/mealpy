Starting Solutions (Custom Initialization)
==========================================

.. toctree::
   :maxdepth: 3


.. warning::
    **Use With Caution!**

    We generally **do not recommend** manually providing starting solutions. Meta-heuristic algorithms rely heavily on uniform random initialization to effectively explore the entire search space. Forcing specific initial positions can bias the swarm and severely increase the risk of premature convergence at local optima.

However, there are advanced scenarios (such as hybrid algorithms, warm-starting, or injecting known good solutions from prior runs) where custom initialization is necessary. You can inject these solutions using the ``starting_solutions`` parameter in the ``solve()`` method.

.. important::
    **Data Shape Requirement**

    Your custom ``starting_solutions`` must be strictly structured as a **2D array (matrix)** or a list of vectors. The number of rows must exactly match your ``pop_size``, and the number of columns must exactly match the problem's dimensionality (number of variables).

Code Example
------------

.. code-block:: python
    :emphasize-lines: 30, 35

    import numpy as np
    from mealpy import TLO, FloatVar

    def frequency_modulated(pos):
        # range: [-6.4, 6.35], f(X*) = 0, phi = 2pi / 100
        phi = 2 * np.pi / 100
        result = 0
        for t in range(0, 101):
            y_t = pos[0] * np.sin(pos[3] * t * phi + pos[1] * np.sin(pos[4] * t * phi + pos[2] * np.sin(pos[5] * t * phi)))
            y_t0 = 1.0 * np.sin(5.0 * t * phi - 1.5 * np.sin(4.8 * t * phi + 2.0 * np.sin(4.9 * t * phi)))
            result += (y_t - y_t0)**2
        return result

    fm_problem = {
        "obj_func": frequency_modulated,
        "bounds": FloatVar(lb=[-6.4, ] * 6, ub=[6.35, ] * 6),
        "minmax": "min",
        "log_to": "console",
    }

    ## Custom function to generate starting positions (2D Matrix: pop_size x n_dims)
    def create_starting_solutions(n_dims, pop_size, base_value=1):
        # Adding random uniform noise to avoid identical overlapping agents
        return np.ones((pop_size, n_dims)) * base_value + np.random.uniform(-1, 1, size=(pop_size, n_dims))

    ## 1. Define the model
    model = TLO.OriginalTLO(epoch=100, pop_size=50)

    ## 2. Generate and inject starting positions via the "starting_solutions" keyword
    list_pos = create_starting_solutions(n_dims=6, pop_size=50, base_value=2)
    best_agent = model.solve(fm_problem, starting_solutions=list_pos)        
    print(f"Run 1 - Best solution: {model.g_best.solution}, Best fitness: {best_agent.target.fitness}")

    ## 3. Training again with completely different starting positions
    list_pos2 = create_starting_solutions(n_dims=6, pop_size=50, base_value=-1)
    best_agent2 = model.solve(fm_problem, starting_solutions=list_pos2)
    print(f"Run 2 - Best solution: {model.g_best.solution}, Best fitness: {best_agent2.target.fitness}")
