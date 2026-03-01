import numpy as np
from mealpy.optimizer import Optimizer


class OriginalRFO(Optimizer):
    """
    Red Fox Optimization (RFO)
    Dehghani et al., Artificial Intelligence Review, 2021
    DOI: 10.1007/s10462-020-09904-6
    """

    def __init__(self, epoch=1000, pop_size=50, **kwargs):
        super().__init__(**kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = False

    def evolve(self, epoch):
        a = 2 * (1 - epoch / self.epoch)
        pop_new = []

        for i in range(self.pop_size):
            r = np.random.uniform(-1, 1, self.problem.n_dims)
            dist = np.abs(self.g_best.solution - self.pop[i].solution)
            pos_new = self.g_best.solution + a * r * dist
            pos_new = self.amend_solution(pos_new)

            agent = self.generate_empty_agent(pos_new)
            agent.target = self.get_target(pos_new)
            pop_new.append(agent)

        self.pop = self.update_target_for_population(pop_new)
