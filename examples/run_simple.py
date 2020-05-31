#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 10:11, 16/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from opfunu.cec.cec2013.unconstraint import Model as ObjFunc13
from opfunu.cec.cec2014.unconstraint import Model as ObjFunc14
from mealpy.evolutionary_based.GA import BaseGA
from mealpy.swarm_based.WOA import BaseWOA
from mealpy.human_based.TLO import BaseTLO
from mealpy.physics_based.HGSO import BaseHGSO, LevyHGSO


def elliptic__(solution=None):
    solution = solution.reshape((-1))
    result = 0
    for i in range(len(solution)):
        result += (10 ** 6) ** (i / (len(solution) - 1)) * solution[i] ** 2
    return result

## Setting parameters
problem_size = 30
func = ObjFunc14(problem_size)
domain_range = [-15, 15]
log = True
epoch = 1000
pop_size = 50

pc = 0.95
pm = 0.025
md = BaseGA(func.F1, problem_size, domain_range, log, epoch, pop_size, pc, pm)
best_position, best_fit, list_loss = md._train__()
print(best_fit)

md2 = BaseWOA(elliptic__, problem_size, domain_range, log, epoch, pop_size)
best_position2, best_fit2, list_loss2 = md2._train__()
print(best_fit2)

md3 = BaseTLO(func.F1, problem_size, domain_range, log, epoch, pop_size)
best_position3, best_fit3, list_loss3 = md3._train__()
print(best_fit3)

md4 = BaseHGSO(func.F5, problem_size, domain_range, log, epoch, pop_size)
best_position4, best_fit4, list_loss4 = md4._train__()
print(best_fit4)

md5 = LevyHGSO(func.F5, problem_size, domain_range, log, epoch, pop_size)
best_position5, best_fit5, list_loss5 = md5._train__()
print(best_fit5)