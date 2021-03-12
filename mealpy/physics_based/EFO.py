#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 21:19, 17/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

from numpy import sqrt, zeros
from numpy.random import uniform, randint
from mealpy.root import Root


class BaseEFO(Root):
    """
    My version of : Electromagnetic Field Optimization (EFO)
        (Electromagnetic field optimization: A physics-inspired metaheuristic optimization algorithm)
    Notes:
        + The flow of algorithm is changed like other metaheuristics.
        + Apply levy-flight for large-scale optimization problems
        + Change equations using g_best solution
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100,
                 r_rate=0.3, ps_rate=0.85, p_field=0.1, n_field=0.45, **kwargs):
        Root.__init__(self, obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.r_rate = r_rate        # default = 0.3     # Like mutation parameter in GA but for one variable
        self.ps_rate = ps_rate      # default = 0.85    # Like crossover parameter in GA
        self.p_field = p_field      # default = 0.1
        self.n_field = n_field      # default = 0.45

    def train(self):
        phi = (1 + sqrt(5)) / 2     # golden ratio
        pop = [self.create_solution() for _ in range(self.pop_size)]
        pop, g_best = self.get_sorted_pop_and_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)

        for epoch in range(0, self.epoch):
            for i in range(0, self.pop_size):
                r_idx1 = randint(0, int(self.pop_size * self.p_field))                                              # top
                r_idx2 = randint(int(self.pop_size * (1 - self.n_field)), self.pop_size)                            # bottom
                r_idx3 = randint(int((self.pop_size * self.p_field) + 1), int(self.pop_size * (1 - self.n_field)))  # middle
                if uniform() < self.ps_rate:
                    # new = g_best + phi* r1 * (top - middle) + r2 (top - bottom)
                    # pos_new = g_best[self.ID_POS] + \
                    #            phi * uniform() * (pop[r_idx1][self.ID_POS] - pop[r_idx3][self.ID_POS]) + \
                    #            uniform() * (pop[r_idx1][self.ID_POS] - pop[r_idx2][self.ID_POS])
                    # new = top + phi * r1 * (g_best - bottom) + r2 * (g_best - middle)
                    pos_new = pop[r_idx1][self.ID_POS] + \
                              phi * uniform() * (g_best[self.ID_POS] - pop[r_idx3][self.ID_POS]) + \
                              uniform() * (g_best[self.ID_POS] - pop[r_idx2][self.ID_POS])
                else:
                    # new = top
                    pos_new = self.levy_flight(epoch+1, pop[i][self.ID_POS], g_best[self.ID_POS])

                # replacement of one electromagnet of generated particle with a random number
                # (only for some generated particles) to bring diversity to the population
                if uniform() < self.r_rate:
                    RI = randint(0, self.problem_size)
                    pos_new[randint(0, self.problem_size)] = uniform(self.lb[RI], self.ub[RI])

                # checking whether the generated number is inside boundary or not
                pos_new = self.amend_position_random_faster(pos_new)
                fit_new = self.get_fitness_position(pos_new)
                if fit_new < pop[i][self.ID_FIT]:
                    pop[i] = [pos_new, fit_new]

                # batch size idea
                if self.batch_idea:
                    if (i + 1) % self.batch_size == 0:
                        pop, g_best = self.update_sorted_population_and_global_best_solution(pop, self.ID_MIN_PROB, g_best)
                else:
                    if (i + 1) % self.pop_size == 0:
                        pop, g_best = self.update_sorted_population_and_global_best_solution(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print(">Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class OriginalEFO(BaseEFO):
    """
    The original version of : Electromagnetic Field Optimization (EFO)
        (Electromagnetic field optimization: A physics-inspired metaheuristic optimization algorithm)
    Link:
        https://www.mathworks.com/matlabcentral/fileexchange/52744-electromagnetic-field-optimization-a-physics-inspired-metaheuristic-optimization-algorithm

        https://www.mathworks.com/matlabcentral/fileexchange/73352-equilibrium-optimizer-eo
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100,
                 r_rate=0.3, ps_rate=0.85, p_field=0.1, n_field=0.45, **kwargs):
        BaseEFO.__init__(self, obj_func, lb, ub, verbose, epoch, pop_size, r_rate, ps_rate, p_field, n_field, kwargs=kwargs)

    def train(self):
        phi = (1 + sqrt(5)) / 2     # golden ratio

        pop = [self.create_solution() for _ in range(self.pop_size)]
        pop, g_best = self.get_sorted_pop_and_global_best_solution(pop, self.ID_FIT, self.ID_MIN_PROB)

        # %random vectors (this is to increase the calculation speed instead of determining the random values in each
        # iteration we allocate them in the beginning before algorithm start
        r_index1 = randint(0, int(self.pop_size * self.p_field), (self.problem_size, self.epoch))
        #random particles from positive field
        r_index2 = randint(int(self.pop_size * (1-self.n_field)), self.pop_size, (self.problem_size, self.epoch))
        # random particles from negative field
        r_index3 = randint(int((self.pop_size * self.p_field) + 1), int(self.pop_size * (1-self.n_field)), (self.problem_size, self.epoch))
        # random particles from neutral field
        ps = uniform(0, 1, (self.problem_size, self.epoch))
        # Probability of selecting electromagnets of generated particle from the positive field
        r_force = uniform(0, 1, self.epoch)
        #random force in each generation
        rp = uniform(0, 1, self.epoch)
        # Some random numbers for checking randomness probability in each generation
        randomization = uniform(0, 1, self.epoch)
        # Coefficient of randomization when generated electro magnet is out of boundary
        RI = 0
        # index of the electromagnet (variable) which is going to be initialized by random number

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
                x_new[RI] = self.lb[RI] + (self.ub[RI] - self.lb[RI]) * randomization[epoch]
                RI = RI + 1
                if RI >= self.problem_size:
                    RI = 0

            # checking whether the generated number is inside boundary or not
            x_new = self.amend_position_random_faster(x_new)
            fit = self.get_fitness_position(x_new)
            # Updating the population if the fitness of the generated particle is better than worst fitness in
            #     the population (because the population is sorted by fitness, the last particle is the worst)
            pop[self.ID_MAX_PROB] = [x_new, fit]

            pop, g_best = self.update_sorted_population_and_global_best_solution(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print(">Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train

