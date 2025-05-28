#!/usr/bin/env python
# Created by "Thieu" at 10:07, 02/03/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from mealpy import get_all_optimizers, get_optimizer_by_class, get_optimizer_by_name

get_all_optimizers(verbose=True)

get_optimizer_by_class(class_name="BaseGA", verbose=True)

get_optimizer_by_name("GA", verbose=True)
get_optimizer_by_name("PSO", verbose=True)
