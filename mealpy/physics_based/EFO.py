#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 21:19, 17/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

import numpy as np
from mealpy.optimizer import Optimizer


class BaseEFO(Optimizer):
    """
    My version of : Electromagnetic Field Optimization (EFO)
        (Electromagnetic field optimization: A physics-inspired metaheuristic optimization algorithm)
    Link:
        https://www.sciencedirect.com/science/article/abs/pii/S2210650215000528
    Notes:
        + The flow of algorithm is changed like other metaheuristics.
        + Change equations using g_best solution
    """

    def __init__(self, problem, epoch=10000, pop_size=100, r_rate=0.3, ps_rate=0.85, p_field=0.1, n_field=0.45, **kwargs):
        """
        Args:
            problem ():
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            r_rate (): default = 0.3     Like mutation parameter in GA but for one variable
            ps_rate (): default = 0.85    Like crossover parameter in GA
            p_field (): default = 0.1     portion of population, positive field
            n_field (): default = 0.45    portion of population, negative field
            **kwargs ():
        """
        super().__init__(problem, kwargs)
        self.nfe_per_epoch = pop_size
        self.sort_flag = True

        self.epoch = epoch
        self.pop_size = pop_size
        self.r_rate = r_rate
        self.ps_rate = ps_rate
        self.p_field = p_field
        self.n_field = n_field
        self.phi = (1 + np.sqrt(5)) / 2  # golden ratio

    def evolve(self, epoch):
        """
        Args:
            epoch (int): The current iteration
        """
        pop_new = []
        for idx in range(0, self.pop_size):
            r_idx1 = np.random.randint(0, int(self.pop_size * self.p_field))  # top
            r_idx2 = np.random.randint(int(self.pop_size * (1 - self.n_field)), self.pop_size)  # bottom
            r_idx3 = np.random.randint(int((self.pop_size * self.p_field) + 1), int(self.pop_size * (1 - self.n_field)))  # middle
            if np.random.uniform() < self.ps_rate:
                # new = g_best + phi* r1 * (top - middle) + r2 (top - bottom)
                # pos_new = g_best[self.ID_POS] + \
                #            phi * np.random.uniform() * (pop[r_idx1][self.ID_POS] - pop[r_idx3][self.ID_POS]) + \
                #            np.random.uniform() * (pop[r_idx1][self.ID_POS] - pop[r_idx2][self.ID_POS])
                # new = top + phi * r1 * (g_best - bottom) + r2 * (g_best - middle)
                pos_new = self.pop[r_idx1][self.ID_POS] + self.phi * np.random.uniform() * (self.g_best[self.ID_POS] - self.pop[r_idx3][self.ID_POS]) \
                          + np.random.uniform() * (self.g_best[self.ID_POS] - self.pop[r_idx2][self.ID_POS])
            else:
                # new = top
                # pos_new = self.levy_flight(epoch + 1, self.pop[idx][self.ID_POS], self.g_best[self.ID_POS])
                pos_new = np.random.uniform(self.problem.lb, self.problem.ub)

            # replacement of one electromagnet of generated particle with a random number
            # (only for some generated particles) to bring diversity to the population
            if np.random.uniform() < self.r_rate:
                RI = np.random.randint(0, self.problem.n_dims)
                pos_new[np.random.randint(0, self.problem.n_dims)] = np.random.uniform(self.problem.lb[RI], self.problem.ub[RI])

            # checking whether the generated number is inside boundary or not
            pos_new = self.amend_position_random(pos_new)
            pop_new.append([pos_new, None])
        pop_new = self.update_fitness_population(pop_new)
        self.pop = self.greedy_selection_population(self.pop, pop_new)


class OriginalEFO(BaseEFO):
    """
    The original version of : Electromagnetic Field Optimization (EFO)
        (Electromagnetic field optimization: A physics-inspired metaheuristic optimization algorithm)
    Link:
        https://www.mathworks.com/matlabcentral/fileexchange/52744-electromagnetic-field-optimization-a-physics-inspired-metaheuristic-optimization-algorithm

        https://www.mathworks.com/matlabcentral/fileexchange/73352-equilibrium-optimizer-eo
    """

    def __init__(self, problem, epoch=10000, pop_size=100, r_rate=0.3, ps_rate=0.85, p_field=0.1, n_field=0.45, **kwargs):
        """
        Args:
            problem ():
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            r_rate (): default = 0.3     Like mutation parameter in GA but for one variable
            ps_rate (): default = 0.85    Like crossover parameter in GA
            p_field (): default = 0.1     portion of population, positive field
            n_field (): default = 0.45    portion of population, negative field
            **kwargs ():
        """
        super().__init__(problem, epoch, pop_size, r_rate, ps_rate, p_field, n_field, **kwargs)
        self.nfe_per_epoch = pop_size
        self.sort_flag = True

    def initialization(self):
        pop = self.create_population(self.pop_size)
        self.pop, self.g_best = self.get_global_best_solution(pop)

        # %random vectors (this is to increase the calculation speed instead of determining the random values in each
        # iteration we allocate them in the beginning before algorithm start
        self.r_index1 = np.random.randint(0, int(self.pop_size * self.p_field), (self.problem.n_dims, self.epoch))
        # random particles from positive field
        self.r_index2 = np.random.randint(int(self.pop_size * (1 - self.n_field)), self.pop_size, (self.problem.n_dims, self.epoch))
        # random particles from negative field
        self.r_index3 = np.random.randint(int((self.pop_size * self.p_field) + 1), int(self.pop_size * (1 - self.n_field)), (self.problem.n_dims, self.epoch))
        # random particles from neutral field
        self.ps = np.random.uniform(0, 1, (self.problem.n_dims, self.epoch))
        # Probability of selecting electromagnets of generated particle from the positive field
        self.r_force = np.random.uniform(0, 1, self.epoch)
        # random force in each generation
        self.rp = np.random.uniform(0, 1, self.epoch)
        # Some random numbers for checking randomness probability in each generation
        self.randomization = np.random.uniform(0, 1, self.epoch)
        # Coefficient of randomization when generated electro magnet is out of boundary
        self.RI = 0
        # index of the electromagnet (variable) which is going to be initialized by random number

    def evolve(self, epoch):
        """
        Args:
            epoch (int): The current iteration
        """
        r = self.r_force[epoch]
        x_new = np.zeros(self.problem.n_dims)  # temporary array to store generated particle
        for i in range(0, self.problem.n_dims):
            if self.ps[i, epoch] > self.ps_rate:
                x_new[i] = self.pop[self.r_index3[i, epoch]][self.ID_POS][i] + \
                           self.phi * r * (self.pop[self.r_index1[i, epoch]][self.ID_POS][i] - self.pop[self.r_index3[i, epoch]][self.ID_POS][i]) + \
                           r * (self.pop[self.r_index3[i, epoch]][self.ID_POS][i] - self.pop[self.r_index2[i, epoch]][self.ID_POS][i])
            else:
                x_new[i] = self.pop[self.r_index1[i, epoch]][self.ID_POS][i]

        # replacement of one electromagnet of generated particle with a random number (only for some generated particles) to bring diversity to the population
        if self.rp[epoch] < self.r_rate:
            x_new[self.RI] = self.problem.lb[self.RI] + (self.problem.ub[self.RI] - self.problem.lb[self.RI]) * self.randomization[epoch]
            RI = self.RI + 1
            if RI >= self.problem.n_dims:
                self.RI = 0

        # checking whether the generated number is inside boundary or not
        pos_new = self.amend_position_random(x_new)
        fit_new = self.get_fitness_position(pos_new)
        # Updating the population if the fitness of the generated particle is better than worst fitness in
        #     the population (because the population is sorted by fitness, the last particle is the worst)
        self.pop[-1] = [pos_new, fit_new]

