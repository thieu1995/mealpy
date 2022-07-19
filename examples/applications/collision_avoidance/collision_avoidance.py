from mealpy.swarm_based.WOA import BaseWOA ,HI_WOA
import numpy as np
import matplotlib.pyplot as plt


class CollisionAvoidance():
    def __init__(self,number_of_waypoint):
        self.number_of_wp = number_of_waypoint
        self.obstacle_x = [1,5,6,7,8]
        self.obstacle_y = [1,5,6,7,8]
        self.number_of_obs = len(self.obstacle_x)
    
    def Euclidean_distance(self,waypoint_one, waypoint_two):
        return np.sqrt(pow(waypoint_one[0]-waypoint_two[0],2) + pow(waypoint_one[1]-waypoint_two[1],2)  )
        
    def constr_one(self,x):
        """_summary_
        This function constraint for problem 

        Args:
            x (list): _description_ waypoint list x and y
        """
        gx = 0
        for i in range(0,self.number_of_wp-1):
            gx += self.Euclidean_distance(waypoint_one=[x[2*i],x[2*i+1]],waypoint_two=[x[2*i+2],x[2*i+3]]) - self.distance_to_target(x)/self.number_of_wp
            #print(i,"zzzz",np.sqrt((x[2*i] - x[2*i+2])**2 + (x[2*i+1] - x[2*i+3])**2), self.distance_to_target(x)/self.number_of_wp,gx)
        return gx 
    
    def constr_two(self,x):
        gx = 0
        for i in range(0,self.number_of_wp):
            for j in range(0,self.number_of_obs):
                gx += 1
        pass
    
    def violate(self,value):
            return 0 if value >= 0 else value
        
    def distance_to_target(self,x):
        return np.sqrt((x[-2]-x[0])**2 + (x[-1]-x[1])**2)
        
    def objective_function(self,x):
        fx = 0
        
        x[0] = 0
        x[1] = 0
        x[-1] = 100
        x[-2] = 100
        
        fx = np.sqrt((x[0] - x [2])**2 + (x[1] - x [3])**2)
        fx += np.sqrt((x[-1] - x [-3])**2 + (x[-2] - x [-4])**2)
        
        for i in range(1,self.number_of_wp-1):
            fx += np.sqrt((x[2*i] - x[2*i+2])**2 + (x[2*i+1] - x[2*i+3])**2) 
        
        const1 = self.violate(self.constr_one(x))
        fx += const1
        
        return fx
    
    def run(self):
        problem_dict1 = {
        "fit_func": self.objective_function,
        "lb": [0,0,0,0,0,0],
        "ub": [150,150,150,150,150,150],
        "minmax": "min",
        }
        
        print(len(problem_dict1["lb"]))
        print(len(problem_dict1["ub"]))
        
        ## Run the algorithm
        model1 = BaseWOA(problem_dict1, epoch=100  , pop_size=1000)
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
    A = CollisionAvoidance(3)
    A.run()