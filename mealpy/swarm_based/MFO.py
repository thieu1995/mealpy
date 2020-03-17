#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 11:59, 17/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from numpy import abs, exp, cos, pi
from numpy.random import uniform
from copy import deepcopy
from mealpy.root import Root


class BaseMFO(Root):
    """
    Standard version of Moth-flame optimization (MFO)
        (Moth-flame optimization algorithm: A novel nature-inspired heuristic paradigm)
        https://www.mathworks.com/matlabcentral/fileexchange/52269-moth-flame-optimization-mfo-algorithm?s_tid=FX_rc1_behav

    It will look so difference in comparison with the mathlab version above. Simply the matlab version above is not working
    (or bad at convergence characteristics). I changed a little bit and it worked now.!!!)
    """

    def __init__(self, root_algo_paras=None, epoch=750, pop_size=100):
        Root.__init__(self, root_algo_paras)
        self.epoch = epoch
        self.pop_size = pop_size

    def _train__(self):
        pop_moths = [self._create_solution__() for _ in range(self.pop_size)]
        # Update the position best flame obtained so far
        pop_flames, g_best= self._sort_pop_and_get_global_best__(pop_moths, self.ID_FIT, self.ID_MIN_PROB)

        for epoch in range(self.epoch):
            # Number of flames Eq.(3.14) in the paper (linearly decreased)
            num_flame = round(self.pop_size - (epoch + 1) * ((self.pop_size - 1) / self.epoch))

            # a linearly decreases from -1 to -2 to calculate t in Eq. (3.12)
            a = -1 + (epoch + 1) * ((-1) / self.epoch)

            for i in range(self.pop_size):

                temp = deepcopy(pop_moths[i][self.ID_POS])
                for j in range(self.problem_size):
                    #   D in Eq.(3.13)
                    distance_to_flame = abs(pop_flames[i][self.ID_POS][j] - pop_moths[i][self.ID_POS][j])
                    t = (a - 1) * uniform() + 1
                    b = 1
                    if i <= num_flame:  # Update the position of the moth with respect to its corresponding flame
                        # Eq.(3.12)
                        temp[j] = distance_to_flame * exp(b * t) * cos(t * 2 * pi) + pop_flames[i][self.ID_POS][j]
                    else:   # Update the position of the moth with respect to one flame
                        # Eq.(3.12).
                        ## Here is a changed, I used the best solution of flames not the solution num_flame th (as original code)
                        temp[j] = distance_to_flame * exp(b * t) * cos(t * 2 * pi) + g_best[self.ID_POS][j]

                ## This is the way I make this algorithm working. I tried to run matlab code with large dimension and it will not convergence.
                fit = self._fitness_model__(temp)
                if fit < pop_moths[i][self.ID_FIT]:
                    pop_moths[i] = [temp, fit]

            ## C1: This is the right way in the paper and original matlab code, but it will make this algorithm face
            ##      with early convergence at local optima
            pop_flames = pop_flames + deepcopy(pop_moths)
            pop_flames = sorted(pop_flames, key=lambda temp: temp[self.ID_FIT])
            pop_flames = pop_flames[:self.pop_size]

            ## C2: I tried this way, but it's not working.
            # Sort the moths and update the flames
            # for i in range(self.pop_size):
            #     if pop_flames[i][self.ID_FIT] > pop_moths[i][self.ID_FIT]:
            #         pop_flames[i] = deepcopy(pop_moths[i])

            # Update the global best flame
            g_best = self._update_global_best__(pop_flames, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.print_train:
                print("> Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))

        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train

