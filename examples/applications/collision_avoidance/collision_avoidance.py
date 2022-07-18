from mealpy.swarm_based.WOA import BaseWOA
import numpy as np
import matplotlib.pyplot as plt


class CollisionAvoidance():
    def __init__(self,number_of_waypoint):
        self.number_of_wp = number_of_waypoint
        self.obstacle_x = [1,5,6,7,8]
        self.obstacle_y = [1,5,6,7,8]
        
    
    def objective_function(self,x):
        fx = 0
        
        def violate(value):
            return 0 if value <= 0 else value
        
        x[0] = 0
        x[1] = 0
        x[-1] = 10
        x[-2] = 10
        
        fx = np.sqrt((x[0] - x [2])**2 + (x[1] - x [3])**2)
        fx += np.sqrt((x[-1] - x [-3])**2 + (x[-2] - x [4])**2)
        
        for i in range(1,self.number_of_wp-1):
            fx += np.sqrt((x[2*i] - x[2*i+2])**2 + (x[2*i+1] - x[2*i+3])**2)
        
        """for i in range(0,len(self.obstacle_x)): 
            fx += np.sqrt((x[2*i] - self.obstacle_x[i])**2 + (x[2*i+1]-self.obstacle_y[i])**2) - 1   """   
        
        return fx
    
    def run(self):
        problem_dict1 = {
        "fit_func": self.objective_function,
        "lb": [0,0,0,0,0,0,0,0,0,0],
        "ub": [15,15,15,15,15,15,15,15,15,15],
        "minmax": "min",
        }
        
        print(len(problem_dict1["lb"]))
        print(len(problem_dict1["ub"]))
        
        ## Run the algorithm
        model1 = BaseWOA(problem_dict1, epoch=100  , pop_size=400)
        best_position, best_fitness = model1.solve()
        print(f"Best solution: {best_position}, Best fitness: {best_fitness}")
        self.plot(best_position)
        
        
    def plot(self,solutions):
        
        x = []
        y = []
        
        for i in range(0,2*self.number_of_wp):
            if i%2 == 0:
                x.append(solutions[i])
            else:
                y.append(solutions[i])
        print(x)
        print(y)         
        plt.scatter(x, y)
        plt.scatter(self.obstacle_x,self.obstacle_y)
        plt.show()
        
        


if __name__=="__main__":
    A = CollisionAvoidance(5)
    A.run()