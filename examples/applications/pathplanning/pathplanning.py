from mealpy.swarm_based.WOA import BaseWOA ,HI_WOA
import numpy as np
import matplotlib.pyplot as plt


class CollisionAvoidance():
    def __init__(self,number_of_waypoint):
        self.number_of_wp = number_of_waypoint
        self.obstacle_x = [5]
        self.obstacle_y = [5]
        self.number_of_obs = len(self.obstacle_x)
        self.distance_between_wp = 2
        self.distance_from_obstacle = 2
    
    def Euclidean_distance(self,point_one, point_two):
        return np.sqrt(pow(point_one[0]-point_two[0],2) + pow(point_one[1]-point_two[1],2))
        
    def constr_one(self,x):
        """_summary_
        This function is constraint function to use equal distance

        Args:
            x (list): _description_ waypoint list x and y
        """
        gx = []
        for i in range(0,self.number_of_wp-1):
            gx.append((self.Euclidean_distance(point_one=[x[2*i],x[2*i+1]],point_two=[x[2*i+2],x[2*i+3]]) -self.distance_between_wp))
            #print(i,"zzzz",np.sqrt((x[2*i] - x[2*i+2])**2 + (x[2*i+1] - x[2*i+3])**2), self.distance_to_target(x)/self.number_of_wp,gx)
        return gx
    
    def constr_two(self,x):
        """
        This function use for obstacle avoidance constraint
        
        Args:
            x (_type_): _description_ waypoint list x and y
        """
        gx = []
        for i in range(1,self.number_of_wp-1):
            for j in range(0,self.number_of_obs):
                gx.append((self.Euclidean_distance(point_one=[x[2*i],x[2*i+1]],point_two=[self.obstacle_x[j],self.obstacle_y[j]]) -self.distance_from_obstacle))
    
        return gx
    
    def constr_tree(self,x):
        return pow((x[2]**2+x[3]**2),0.5)-10
    
    def constr_four(self,x):
        return pow((x[4]-x[2])**2 +(x[5]-x[3])**2 ,0.5)-50
    
    def violate(self,value):
            return 0 if value <= 0 else value
        
    def distance_to_target(self,x):
        return np.sqrt((x[-2]-x[0])**2 + (x[-1]-x[1])**2)
        
    def objective_function(self,x):
        fx = 0
        
        x[0] = 0
        x[1] = 0
        x[-1] = 10
        x[-2] = 10
        
        fx = self.Euclidean_distance(point_one=[x[0],x[1]],point_two=[x[2],x[3]])        
        fx += self.Euclidean_distance(point_one=[x[-1],x[-2]],point_two=[x[2],x[3]])
        
        for i in range(1,self.number_of_wp-2):
            fx += self.Euclidean_distance(point_one=[x[2*i],x[2*i+1]],point_two=[x[2*i+2],x[2*i+3]])

        
        #const1 = self.violate(self.constr_one(x))
        # this is static penalty function 
        # link == > https://www.researchgate.net/profile/Oezguer-Yeniay/publication/228339797_Penalty_Function_Methods_for_Constrained_Optimization_with_Genetic_Algorithms/links/56d1ecda08ae85c8234ade07/Penalty-Function-Methods-for-Constrained-Optimization-with-Genetic-Algorithms.pdf
        
        """print("const1",self.constr_one(x))
        print("<<<<<<<<<<<<")
        print(self.constr_two(x))"""
        print(self.Euclidean_distance(point_one=[x[2],x[3]],point_two=[self.obstacle_x[0],self.obstacle_y[0]]))
        fx += max(self.constr_one(x))**2+max(self.constr_two(x))**2
        return fx
    
    def run(self):
        problem_dict1 = {
        "fit_func": self.objective_function,
        "lb": [0,0,0,0,0,0],
        "ub": [150,150,150,150,150,150],
        "minmax": "min",
        }
        
        
        
        ## Run the algorithm
        model1 = BaseWOA(problem_dict1, epoch=100  , pop_size=500)
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
        plt.scatter(self.obstacle_x,self.obstacle_y,s=(10 * self.distance_from_obstacle)**2 )
        plt.show()
        
        


if __name__=="__main__":
    A = CollisionAvoidance(3)
    A.run()