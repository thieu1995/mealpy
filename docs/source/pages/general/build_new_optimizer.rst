===================
Build New Optimizer
===================

The figure below shows the flow of the Optimizer class and which methods can be overridden and which methods should not be overridden to take advantage of
the Optimizer class.


.. image:: /_static/images/mealpy_flow.png


Based on this flow, we have created an example in "examples/build_new_optimizer.py" to show you how to do this in code.


**How to create a new optimizer?**

.. code-block:: python

	import numpy as np
	from mealpy import Optimizer, FloatVar


	class MyAlgorithm(Optimizer):
	    """
	    This is an example how to build new optimizer
	    """

	    def __init__(self, epoch=10000, pop_size=100, m_clusters=2, p1=0.75, **kwargs):
	        super().__init__(**kwargs)
	        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
	        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
	        self.m_clusters = self.validator.check_int("m_clusters", m_clusters, [2, 5])
	        self.p1 = self.validator.check_float("p1", p1, (0, 1.0))

	        self.sort_flag = True
	        # Determine to sort the problem or not in each epoch
	        ## if True, the problem always sorted with fitness value increase
	        ## if False, the problem is not sorted

	    def initialize_variables(self):
	        """
	        This is method is called before initialization() method.
	        """
	        ## Support variables
	        self.n_agents = int(self.pop_size / self.m_clusters)
	        self.space = self.problem.ub - self.problem.lb

	    def initialization(self):
	        """
	        Override this method if needed. But the first 2 lines of code is required.
	        """
	        ### Required code
	        if self.pop is None:
	            self.pop = self.generate_population(self.pop_size)

	        ### Your additional code can be implemented here
	        self.mean_pos = np.mean([agent[self.ID_POS] for agent in self.pop])

	    def evolve(self, epoch):
	        """
	        You can do everything in this function (i.e., Loop through the population multiple times)

	        Args:
	            epoch (int): The current iteration
	        """
	        epsilon = 1.0 - epoch / self.epoch      # The epsilon in each epoch is changing based on this equation

	        ## 1. Replace the almost worst agent by random agent
	        if self.generator.uniform() < self.p1:
	            idx = self.generator.integers(self.n_agents, self.pop_size)
	            self.pop[idx] = self.generate_agent()

	        ## 2. Replace all bad solutions by current_best + noise
	        for idx in range(self.n_agents, self.pop_size):
	            pos_new = self.pop[0].solution + epsilon * self.space * self.generator.normal(0, 1)
	            pos_new = self.correct_solution(pos_new)
	            agent = self.generate_agent(pos_new)
	            if self.compare_target(agent.target, self.pop[idx].target, self.problem.minmax):
	                self.pop[idx] = agent

	        ## 3. Move all good solutions toward current best solution
	        for idx in range(0, self.n_agents):
	            if idx == 0:
	                pos_new = self.pop[idx].solution + epsilon * self.space * self.generator.uniform(0, 1)
	            else:
	                pos_new = self.pop[idx].solution + epsilon * self.space * (self.pop[0].solution - self.pop[idx].solution)
	            pos_new = self.correct_solution(pos_new)
	            agent = self.generate_agent(pos_new)
	            if self.compare_target(agent.target, self.pop[idx].target, self.problem.minmax):
	                self.pop[idx] = agent

	        ## Do additional works here if needed.


	## Time to test our new optimizer
	def objective_function(solution):
	    return np.sum(solution**2)

	problem_dict1 = {
	    "obj_func": objective_function,
	    "bounds": FloatVar(lb=[-100, ]*100, ub=[100, ]*100),
	    "minmax": "min",
	}

	epoch = 50
	pop_size = 50
	model = MyAlgorithm(epoch, pop_size)
	g_best = model.solve(problem_dict1)
	print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")


.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4