from mealpy.swarm_based.WOA import BaseWOA 
from mealpy.swarm_based.PSO import BasePSO
from mealpy.swarm_based.GWO import BaseGWO
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt


class PathPlanning():
    def __init__(self,initial_point:list, final_point:list, number_of_waypoint:list):
        """_summary_
        This class generate path to reach final point and avoid collision between obstacles and waypoints
        
        reference:
        https://link.springer.com/article/10.1007/s10489-018-1384-y

        
        Args:
            initial_point (list): _description_
            final_point (list): _description_
            number_of_waypoint (list): _description_
        """
        self.number_of_wp = number_of_waypoint
        self.initial_point = initial_point
        self.final_point = final_point
        self.obstacle_x = [50,95,100]   # X position of obstacles(m)
        self.obstacle_y = [50,75,115]   # Y position of obstacles(m)
        self.number_of_obs = len(self.obstacle_x) # Number of obstacles(m)
        self.distance_from_obstacle = 20    # Distance from obstacles(m)
        self.coeficient = 100
            
    def constr_one(self,X,Y):
        """_summary_
        This function constraint to waypoint for avoiding

        Args:
            X (list): _description_ list of X positions of path
            Y (list): _description_ list of Y positions of path 

        Returns:
            _type_: _description_ cost value of constraint
        """
        
        lenpx = np.zeros(X.shape[0])
        count = 0
        
        
        for i in range(0,self.number_of_obs):
            x_center = self.obstacle_x[i]
            y_center = self.obstacle_y[i]
            
            d = np.sqrt((X - x_center) ** 2 + (Y - y_center) ** 2)
            
            inside = self.distance_from_obstacle > d
            
            cost = np.where(inside, self.coeficient /d, 0)
            
            if (inside.any()):
                count += 1
            
            lenpx += np.nanmean(cost)
        
        return lenpx
            
                                
    def objective_function(self,x):
        """_summary_
        The objective function of problem
        Args:
            x (list): _description_ generated waypoint from optimization algorithm

        Returns:
            _type_: _description_
        """
        fx = 0
        
        x[0] = self.initial_point[0]                       # initial X position
        x[self.number_of_wp-1] = self.initial_point[1]    # final X position
        
        x[self.number_of_wp] = self.final_point[0]        # initial Y position
        x[2*self.number_of_wp-1] = self.final_point[1]  # final Y position
        
        interpolator ='quadratic'
        # interpolator = 'cubic'
        # interpolator = 'slinear'
        # interpolator = 'zero'
        
        
        d = 50
        ns = 1 + (self.number_of_wp + 1) * d         # Number of points along the spline

        
        x1 = x[0:self.number_of_wp]
        y1 = x[self.number_of_wp:2*self.number_of_wp]
        
        
        t = np.linspace(0, 1, self.number_of_wp)
        
        CSx = interp1d(t, x1 ,kind=interpolator, assume_sorted=True)
        CSy = interp1d(t, y1, kind=interpolator, assume_sorted=True)
        
        # Coordinates of the discretized path
        s = np.linspace(0, 1, ns)
        
        Px = CSx(s)
        Py = CSy(s)
       
        dX = np.diff(Px)
        dY = np.diff(Py)
        L = np.sqrt(dX ** 2 + dY ** 2)
        
        gx = self.constr_one(Px,Py)
       
       
        Cost = L + (1+gx[0:len(L)])
       
        return Cost
    
    def run(self):
        """_summary_
        Main function of problem 
        """
        lb = []
        ub = []
        
        # add bound to waypoint
        for i in range(0,self.number_of_wp*2):
            lb.append(0)
            ub.append(150)
            
  
        problem_dict1 = {
        "fit_func": self.objective_function,
        "lb": lb,
        "ub": ub,
        "minmax": "min",
        }
        
        ## Run the algorithm
        model1 = BaseGWO(problem_dict1, epoch=80  , pop_size=200)
        best_position, best_fitness = model1.solve()
        print(f"Best solution: {best_position}, Best fitness: {best_fitness}")
        self.plot(best_position)
        
        
    def plot(self,solutions):
        """_summary_
        plot solution waypoint with obstacles

        Args:
            solutions (list): _description_ solution list of problem
        """
        
        x = solutions[0:self.number_of_wp]
        y = solutions[self.number_of_wp:2*self.number_of_wp]
        
                 
        plt.scatter(x, y)
        plt.scatter(self.obstacle_x,self.obstacle_y,s=( self.distance_from_obstacle)**2 )
        plt.show()
        
        


if __name__=="__main__":
    A = CollisionAvoidance(5)
    A.run()