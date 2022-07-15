from mealpy.swarm_based.WOA import BaseWOA
import numpy as np


class CollisionAvoidance():
    def __init__(self,number_of_waypoint):
        self.number_of_wp = number_of_waypoint
        self.obstacle_x = [1,5,6,7,8]
        self.obstacle_y = [1,5,6,7,8]
        pass
    
    def objective_function(self,x):
        fx = 0
        
        def g1(x):
            return x[0] - 1
        
        def g2(x):
            return x[1] -1
        
        def g3(x):
            return x[-1] - 10
        
        def g4(x):
            return x[-2] -10
        def violate(value):
            return 0 if value <= 0 else value
        
        x[1] = 4
        x[-1] = 10
        
        
        for i in range(0,len(self.obstacle_x)):
            fx += np.sqrt((x[2*i] - self.obstacle_x[i])**2 + (x[2*i+1]-self.obstacle_y[i])**2) - 5      
        
        fx += violate(g1(x)) + violate(g2(x))
        return fx
    
    def run(self):
        problem_dict1 = {
        "fit_func": self.objective_function,
        "lb": [0,0,0,0,0,0,0,0,0,0],
        "ub": [10,10,10,10,10,10,10,10,10,10],
        "minmax": "min",
        }
        
        print(len(problem_dict1["lb"]))
        print(len(problem_dict1["ub"]))
        
        ## Run the algorithm
        model1 = BaseWOA(problem_dict1, epoch=100  , pop_size=500)
        best_position, best_fitness = model1.solve()
        print(f"Best solution: {best_position}, Best fitness: {best_fitness}")
        


if __name__=="__main__":
    A = CollisionAvoidance(10)
    A.run()