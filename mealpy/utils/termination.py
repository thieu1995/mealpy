#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu" at 17:29, 13/10/2021                                                               %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Nguyen_Thieu2                                  %
#       Github:     https://github.com/thieu1995                                                        %
# ------------------------------------------------------------------------------------------------------%

class Termination:

    DEFAULT_MAX_MG = 1000  # Maximum number of epochs / generations (Default: 1000 epochs)
    DEFAULT_MAX_FE = 100000  # Maximum number of function evaluation (Default: 100000 FE)
    DEFAULT_MAX_TB = 20  # Maximum number of time bound (Default: 20 seconds)
    DEFAULT_MAX_ES = 20  # Maximum number of early stopping iterations (Default: 20 loops / generations)

    def __init__(self, termination):
        """
        Set the stopping condition
        Args:
            termination: dictionary of stopping condition

        Examples:
             termination = {
                "mode": "TB",       # MG: maximum generation, FE: function evaluation, TB: time bound, ES: early stopping
                "quantity": 120,    # The value of stopping condition
            }
        """

        if isinstance(termination, dict):
            if "mode" in termination:
                self.mode = termination["mode"]
                if self.mode is not None:
                    if self.mode is None or self.mode == "FE":
                        self.__check_input__(termination, "Function Evaluation", self.DEFAULT_MAX_FE)
                    elif self.mode == "TB":
                        self.__check_input__(termination, "Time Bound", self.DEFAULT_MAX_TB)
                    elif self.mode == "ES":
                        self.__check_input__(termination, "Early Stopping", self.DEFAULT_MAX_ES)
                    elif self.mode == "MG":
                        self.__check_input__(termination, "Maximum Generation", self.DEFAULT_MAX_MG)
                    else:
                        print("Your stopping condition is not support. Please choice other one.")
                        exit(0)
                else:
                    print("Please enter your stopping condition.")
                    exit(0)
            else:
                print("Select your termination mode (FE, TB, ES or MG)!")
                exit(0)
        else:
            self.quantity = self.DEFAULT_MAX_MG
            self.mode = "MG"
            print(f"Stopping condition mode (default): Maximum Generation, with default value is: {self.quantity} generations")

        for key, value in termination.items():
            setattr(self, key, value)

    def __check_input__(self, termination, name, default_value):
        if "quantity" in termination:
            self.quantity = termination["quantity"]
            if type(self.quantity) is int and self.quantity > 0:
                print(f"Stopping condition mode: {name}, with maximum value is: {self.quantity}")
            else:
                print(f"Maximum {name} should be int number and > 0.")
                exit(0)
        else:
            self.quantity = default_value
            print(f"Stopping condition mode: {name}, with maximum {name} default is: {self.quantity}")

    def logging(self, verbose=True):
        if verbose:
            print(f"Stopping criterion with mode {self.mode} occurs. End program!")