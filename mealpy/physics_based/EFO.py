#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 21:19, 17/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from numpy import sqrt, zeros, where, logical_or
from numpy.random import uniform, randint
from copy import deepcopy
from mealpy.root import Root


class BaseEFO(Root):
    """
    This is my version of : Electromagnetic Field Optimization (EFO)
        (Electromagnetic field optimization: A physics-inspired metaheuristic optimization algorithm)
    Link:
        The flow of algorithm is changed like other metaheuristics.
    """

    def __init__(self, objective_func=None, problem_size=50, domain_range=(-1, 1), log=True,
                 epoch=750, pop_size=100, r_rate=0.3, ps_rate=0.85, p_field=0.1, n_field=0.45):
        Root.__init__(self, objective_func, problem_size, domain_range, log)
        self.epoch = epoch
        self.pop_size = pop_size

        self.r_rate = r_rate        # default = 0.3     # Like mutation parameter in GA but for one variable
        self.ps_rate = ps_rate      # default = 0.85    # Like crossover parameter in GA
        self.p_field = p_field      # default = 0.1
        self.n_field = n_field      # default = 0.45


    def _train__(self):
        phi = (1 + sqrt(5)) / 2     # golden ratio
        r_force = uniform(0, 1, self.epoch)  # random force in each generation

        pop = [self._create_solution__() for _ in range(self.pop_size)]
        pop, g_best = self._sort_pop_and_get_global_best__(pop, self.ID_FIT, self.ID_MIN_PROB)
        pop_new = deepcopy(pop)

        for epoch in range(0, self.epoch):
            r = r_force[epoch]
            for i in range(0, self.pop_size):
                r_idx1 = randint(0, int(self.pop_size * self.p_field))
                r_idx2 = randint(int(self.pop_size * (1 - self.n_field)), self.pop_size)
                r_idx3 = randint(int((self.pop_size * self.p_field) + 1), int(self.pop_size * (1 - self.n_field)))
                if uniform() < self.ps_rate:
                    pop_new[i][self.ID_POS] = pop[r_idx3][self.ID_POS] + \
                               phi * r * (pop[r_idx1][self.ID_POS] - pop[r_idx3][self.ID_POS]) + \
                               r * (pop[r_idx3][self.ID_POS] - pop[r_idx2][self.ID_POS])
                else:
                    pop_new[i][self.ID_POS] = pop[r_idx1][self.ID_POS]

            for i in range(0, self.pop_size):
                # replacement of one electromagnet of generated particle with a random number (only for some generated particles) to bring diversity to the population
                if uniform() < self.r_rate:
                    RI = randint(0, self.problem_size)
                    pop_new[i][self.ID_POS][RI] = uniform(self.domain_range[0], self.domain_range[1])

            # checking whether the generated number is inside boundary or not
            for i in range(0, self.pop_size):
                temp = pop_new[i][self.ID_POS]
                temp = where(logical_or(temp < self.domain_range[0], temp > self.domain_range[1]), uniform(self.domain_range[0], self.domain_range[1]), temp)
                fit = self._fitness_model__(temp)
                if fit < pop[i][self.ID_FIT]:
                    pop[i] = [temp, fit]

            pop, g_best = self._sort_pop_and_update_global_best__(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.log:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))

        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class OriginalEFO(BaseEFO):
    """
    The original version of : Electromagnetic Field Optimization (EFO)
        (Electromagnetic field optimization: A physics-inspired metaheuristic optimization algorithm)
    Link:
        https://www.mathworks.com/matlabcentral/fileexchange/52744-electromagnetic-field-optimization-a-physics-inspired-metaheuristic-optimization-algorithm

        https://www.mathworks.com/matlabcentral/fileexchange/73352-equilibrium-optimizer-eo
    """
    def __init__(self, objective_func=None, problem_size=50, domain_range=(-1, 1), log=True,
                 epoch=750, pop_size=100, r_rate=0.3, ps_rate=0.2, p_field=0.1, n_field=0.45):
        BaseEFO.__init__(self, objective_func, problem_size, domain_range, log, epoch, pop_size, r_rate, ps_rate, p_field, n_field)

    def _train__(self):
        phi = (1 + sqrt(5)) / 2     # golden ratio

        pop = [self._create_solution__() for _ in range(self.pop_size)]
        pop, g_best = self._sort_pop_and_get_global_best__(pop, self.ID_FIT, self.ID_MIN_PROB)

        # %random vectors (this is to increase the calculation speed instead of determining the random values in each
        # iteration we allocate them in the beginning before algorithm start
        r_index1 = randint(0, int(self.pop_size * self.p_field), (self.problem_size, self.epoch)) #random particles from positive field
        r_index2 = randint(int(self.pop_size * (1-self.n_field)), self.pop_size, (self.problem_size, self.epoch))   # random particles from negative field
        r_index3 = randint(int((self.pop_size * self.p_field) + 1), int(self.pop_size * (1-self.n_field)), (self.problem_size, self.epoch))    # random particles from neutral field
        ps = uniform(0, 1, (self.problem_size, self.epoch))     # Probability of selecting electromagnets of generated particle from the positive field
        r_force = uniform(0, 1, self.epoch) #random force in each generation
        rp = uniform(0, 1, self.epoch)    # Some random numbers for checking randomness probability in each generation
        randomization = uniform(0, 1, self.epoch)     # Coefficient of randomization when generated electro magnet is out of boundary
        RI = 0      # index of the electromagnet (variable) which is going to be initialized by random number

        for epoch in range(0, self.epoch):
            r = r_force[epoch]
            x_new = zeros(self.problem_size)  # temporary array to store generated particle
            for i in range(0, self.problem_size):

                if ps[i, epoch] > self.ps_rate:
                    x_new[i] = pop[r_index3[i, epoch]][self.ID_POS][i] + \
                        phi * r * (pop[r_index1[i, epoch]][self.ID_POS][i] - pop[r_index3[i, epoch]][self.ID_POS][i]) + \
                            r * (pop[r_index3[i, epoch]][self.ID_POS][i] - pop[r_index2[i, epoch]][self.ID_POS][i])
                else:
                    x_new[i] = pop[r_index1[i, epoch]][self.ID_POS][i]

            # replacement of one electromagnet of generated particle with a random number (only for some generated particles) to bring diversity to the population
            if rp[epoch] < self.r_rate:
                x_new[RI] = self.domain_range[0] + (self.domain_range[1] - self.domain_range[0]) * randomization[epoch]
                RI = RI + 1
                if RI >= self.problem_size:
                    RI = 0

            # checking whether the generated number is inside boundary or not
            for i in range(0, self.problem_size):
                if x_new[i] < self.domain_range[0] or x_new[i] > self.domain_range[1]:
                    x_new[i] = uniform(self.domain_range[0], self.domain_range[1])
            fit = self._fitness_model__(x_new)
            # Updating the population if the fitness of the generated particle is better than worst fitness in
            #     the population (because the population is sorted by fitness, the last particle is the worst)
            pop[self.ID_MAX_PROB] = [x_new, fit]

            pop, g_best = self._sort_pop_and_update_global_best__(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.log:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train

