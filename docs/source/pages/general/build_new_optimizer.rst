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
	from mealpy.optimizer import Optimizer


	class MyAlgorithm(Optimizer):
	    """
	    This is an example how to build new optimizer

	    Notes
	    ~~~~~
	    + Read more at: https://mealpy.readthedocs.io/en/latest/pages/build_new_optimizer.html
	    """

	    def __init__(self, epoch=10000, pop_size=100, m_clusters=2, p1=0.75, **kwargs):
	        """
	        Args:
	            epoch (int): maximum number of iterations, default = 10000
	            pop_size (int): number of population size, default = 100
	            m_clusters (int): number of clusters
	            p1 (float): the probability of updating the worst solution
	        """
	        super().__init__(**kwargs)
	        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
	        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
	        self.m_clusters = self.validator.check_int("m_clusters", m_clusters, [2, 5])
	        self.p1 = self.validator.check_float("p1", p1, (0, 1.0))

	        self.nfe_per_epoch = self.pop_size
	        self.sort_flag = True
	        # Determine to sort the problem or not in each epoch
	        ## if True, the problem always sorted with fitness value increase
	        ## if False, the problem is not sorted

	    def initialize_variables(self):
	        """
	        This is method is called before initialization() method.
	        Returns:

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
	            self.pop = self.create_population(self.pop_size)

	        ### Your additional code can be implemented here
	        self.mean_pos = np.mean([agent[self.ID_POS] for agent in self.pop])

	    def evolve(self, epoch):
	        """
	        You can do everything in this function (i.e., Loop through the population multiple times)

	        Args:
	            epoch (int): The current iteration
	        """
	        epxilon = 1 - 1 * (epoch + 1) / self.epoch      # The epxilon in each epoch is changing based on this equation

	        ## 1. Replace the almost worst agent by random agent
	        if np.random.uniform() < self.p1:
	            idx = np.random.randint(self.n_agents, self.pop_size)
	            solution_new = self.create_solution(self.problem.lb, self.problem.ub)
	            self.pop[idx] = solution_new

	        ## 2. Replace all bad solutions by current_best + noise
	        for idx in range(self.n_agents, self.pop_size):
	            pos_new = self.pop[0][self.ID_POS] + epxilon * self.space * np.random.normal(0, 1)
	            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
	            fit_new = self.get_target_wrapper(pos_new)
	            if self.compare_agent([pos_new, fit_new], self.pop[idx]):
	                self.pop[idx] = [pos_new, fit_new]

	        ## 3. Move all good solutions toward current best solution
	        for idx in range(0, self.n_agents):
	            if idx == 0:
	                pos_new = self.pop[idx][self.ID_POS] + epxilon * self.space * np.random.uniform(0, 1)
	            else:
	                pos_new = self.pop[idx][self.ID_POS] + epxilon * self.space * (self.pop[0][self.ID_POS] - self.pop[idx][self.ID_POS])
	            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
	            fit_new = self.get_target_wrapper(pos_new)
	            if self.compare_agent([pos_new, fit_new], self.pop[idx]):
	                self.pop[idx] = [pos_new, fit_new]

	        ## Do additional works here if needed.


	## Time to test our new optimizer
	def fitness(solution):
	    return np.sum(solution**2)

	problem_dict1 = {
	    "fit_func": fitness,
	    "lb": [-100, ]*100,
	    "ub": [100, ]*100,
	    "minmax": "min",
	}

	epoch = 50
	pop_size = 50
	model = MyAlgorithm(epoch, pop_size)
	best_position, best_fitness = model.solve(problem_dict1)
	print(f"Solution: {best_position}, Fitness: {best_fitness}")




.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4