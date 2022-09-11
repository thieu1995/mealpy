=============
Visualization
=============

Drawing all available figures. There are 8 different figures for each algorithm.

**1. Based on fitness value (global best and local best fitness chart)**:

.. image:: /_static/images/results/Global-best-convergence-chart.png
    :width: 49 %
.. image:: /_static/images/results/Current-best-convergence-chart.png
    :width: 49 %


**2. Based on objective values (global best and local best objective chart)**:

.. image:: /_static/images/results/global-objective-chart.png
    :width: 49 %
.. image:: /_static/images/results/local-objective-chart.png
    :width: 49 %


**3. Based on runtime value (runtime for each epoch)**

**4. Based on exploration verse exploration value**

.. image:: /_static/images/results/Runtime-per-epoch-chart.png
   :width: 49 %
.. image:: /_static/images/results/explore_exploit_chart.png
   :width: 49 %

**5. Based on diversity of population**

**6. Based on trajectory value (1D, 2D only)**

.. image:: /_static/images/results/diversity_chart.png
   :width: 49 %
.. image:: /_static/images/results/1d_trajectory.png
   :width: 49 %


**How to call the functions?**

.. code-block:: python

	model = SMA.BaseSMA(epoch=100, pop_size=50, pr=0.03)
	model.solve(problem)

	## You can access them all via object "history" like this:
	model.history.save_global_objectives_chart(filename="hello/goc")
	model.history.save_local_objectives_chart(filename="hello/loc")

	model.history.save_global_best_fitness_chart(filename="hello/gbfc")
	model.history.save_local_best_fitness_chart(filename="hello/lbfc")

	model.history.save_runtime_chart(filename="hello/rtc")

	model.history.save_exploration_exploitation_chart(filename="hello/eec")

	model.history.save_diversity_chart(filename="hello/dc")

	model.history.save_trajectory_chart(list_agent_idx=[3, 5], selected_dimensions=[3], filename="hello/tc")



.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4