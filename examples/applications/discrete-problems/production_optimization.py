#!/usr/bin/env python
# Created by "Thieu" at 20:42, 07/11/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

# Production optimization, also known as production planning, is the process of determining the most efficient and effective way to produce goods
# or services in a manufacturing or production environment. It involves making strategic decisions to allocate resources, schedule production
# activities, and optimize various factors to achieve the desired production goals.

# The objective of production optimization is to find the optimal production plan that maximizes production output, minimizes production costs,
# and ensures efficient utilization of resources while meeting customer demand and other relevant constraints. The production plan typically
# includes decisions related to production quantities, production schedules, resource allocation, inventory management, and more.

# The production optimization process typically involves the following steps:
#   1. Demand Forecasting: Estimating the future demand for the products or services to be produced. This forecast provides a
#       basis for planning production quantities and schedules.
#   2. Capacity Planning: Assessing the available production capacity and determining if it is sufficient to meet the forecasted demand.
#       Capacity planning involves evaluating current resources, such as machinery, labor, and facilities, and making decisions on
#       resource allocation and potential expansion or contraction.
#   3. Production Scheduling: Creating a detailed schedule that specifies when and how each production task or operation will
#       be executed. This includes determining the order of production, the allocation of resources to specific tasks, and the
#       sequencing of activities to optimize efficiency.
#   4. Inventory Management: Determining the optimal levels of raw materials, work-in-progress (WIP), and finished goods inventory
#       to ensure smooth production flow, minimize stockouts, and control inventory holding costs.
#   5. Resource Allocation: Assigning resources such as labor, equipment, and materials to different production tasks or processes
#       in the most efficient manner. This involves considering factors such as resource availability, skill requirements,
#       and minimizing idle time or bottlenecks.
#   6. Performance Monitoring and Optimization: Continuously monitoring the production process and performance metrics to
#       identify areas for improvement and make adjustments to the production plan. This may involve analyzing key performance indicators (KPIs)
#       like production output, cycle time, lead time, quality, and cost to identify bottlenecks, inefficiencies, or opportunities for optimization.

# By optimizing the production plan, businesses can improve operational efficiency, reduce costs, enhance customer satisfaction through
# timely delivery, and gain a competitive advantage in the market. It's worth noting that production optimization can vary significantly
# depending on the specific industry, production processes, and objectives of the organization. Different industries may have unique
# considerations, such as perishable goods, complex supply chains, or regulatory requirements, which need to be incorporated into the
# production planning process. Overall, production optimization aims to strike a balance between maximizing production output, minimizing costs,
# and meeting customer demands, while considering various constraints and factors specific to the production environment.


#============================================== EXAMPLE ========================================================

# Let's consider a simplified example of production optimization in the context of a manufacturing company that produces electronic devices,
# such as smartphones. The objective is to maximize production output while minimizing production costs.

#   1. Demand Forecasting: The company analyzes market trends, historical sales data, and other relevant factors to forecast the demand for
#       smartphones. Based on this forecast, they estimate that the demand for the upcoming quarter will be 10,000 smartphones.
#   2. Capacity Planning: The company evaluates its production capacity, including factors such as available machinery, labor resources,
#       and production floor space. They determine that their current capacity allows them to produce up to 12,000 smartphones per quarter.
#   3. Production Scheduling: The production team creates a detailed schedule that specifies the production quantities and sequences.
#       They decide to divide the production into two batches: one batch of 8,000 smartphones in the first half of the quarter and another
#       batch of 4,000 smartphones in the second half. This scheduling decision is based on optimizing the utilization of resources and minimizing setup times.
# 4. Inventory Management: The company maintains an inventory of raw materials and components required for smartphone production. They monitor
#       the inventory levels and ensure that they have sufficient materials to meet the production schedule without excessive stockouts or
#       excess inventory. By optimizing inventory management, they minimize holding costs while ensuring uninterrupted production.
# 5. Resource Allocation: The production team assigns the available labor and machinery to the production tasks based on their capabilities
#       and requirements. They optimize the allocation of resources to balance the workload and minimize idle time. This may involve cross-training
#       employees to handle multiple tasks and optimizing the utilization of machinery to avoid bottlenecks.
# 6. Performance Monitoring and Optimization: Throughout the production process, the company closely monitors key performance indicators
#       (KPIs) such as production output, cycle time, defect rate, and production costs. They identify areas of improvement, such as optimizing
#       production line layouts, streamlining assembly processes, or implementing quality control measures. By continuously monitoring and
#       optimizing performance, they strive to improve efficiency and reduce costs.

# By following these steps and continuously optimizing their production plan, the company can maximize their smartphone production output
# while minimizing production costs. This, in turn, helps them meet customer demand, reduce time-to-market, and enhance their competitiveness in the market.
# It's important to note that this example is simplified for illustrative purposes.

# This example uses binary representations for production configurations, assuming each task can be assigned to a resource (1) or not (0).
# You may need to adapt the representation and operators to suit your specific production optimization problem.


import numpy as np
from mealpy import BinaryVar, WOA, Problem

# Define the problem parameters
num_tasks = 10
num_resources = 5

# Example task processing times
task_processing_times = np.array([2, 3, 4, 2, 3, 2, 3, 4, 2, 3])

# Example resource capacity
resource_capacity = np.array([10, 8, 6, 12, 15])

# Example production costs and outputs
production_costs = np.array([5, 6, 4, 7, 8, 9, 5, 6, 7, 8])
production_outputs = np.array([20, 18, 16, 22, 25, 24, 20, 18, 19, 21])

# Example maximum total production time
max_total_time = 50

# Example maximum defect rate
max_defect_rate = 0.2

# Penalty for invalid solution
penalty = -1000

data = {
    "num_tasks": num_tasks,
    "num_resources": num_resources,
    "task_processing_times": task_processing_times,
    "resource_capacity": resource_capacity,
    "production_costs": production_costs,
    "production_outputs": production_outputs,
    "max_defect_rate": max_defect_rate,
    "penalty": penalty
}


class SupplyChainProblem(Problem):
    def __init__(self, bounds=None, minmax=None, data=None, **kwargs):
        self.data = data
        super().__init__(bounds, minmax, **kwargs)

    def obj_func(self, x):
        x_decoded = self.decode_solution(x)
        x = x_decoded["placement_var"].reshape((self.data["num_tasks"], self.data["num_resources"]))

        # If any row has all 0 value, it indicates that this task is not allocated to any resource
        if np.any(np.all(x==0, axis=1)) or np.any(np.all(x==0, axis=0)):
            return self.data["penalty"]

        # Check violated constraints
        violated_constraints = 0

        # Calculate resource utilization
        resource_utilization = np.sum(x, axis=0)
        # Resource capacity constraint
        if np.any(resource_utilization > self.data["resource_capacity"]):
            violated_constraints += 1

        # Time constraint
        total_time = np.sum(np.dot(self.data["task_processing_times"].reshape(1, -1), x))
        if total_time > max_total_time:
            violated_constraints += 1

        # Quality constraint
        defect_rate = np.dot(self.data["production_costs"].reshape(1, -1), x) / np.dot(self.data["production_outputs"], x)
        if np.any(defect_rate > max_defect_rate):
            violated_constraints += 1

        # Calculate the fitness value based on the objectives and constraints
        profit = np.sum(np.dot(self.data["production_outputs"].reshape(1, -1), x)) - np.sum(np.dot(self.data["production_costs"].reshape(1, -1), x))
        if violated_constraints > 0:
            return profit + self.data["penalty"] * violated_constraints       # Penalize solutions with violated constraints
        return profit


bounds = BinaryVar(n_vars=num_tasks * num_resources, name="placement_var")
problem = SupplyChainProblem(bounds=bounds, minmax="max", data=data)

model = WOA.OriginalWOA(epoch=50, pop_size=20)
model.solve(problem)

print(f"Best agent: {model.g_best}")                    # Encoded solution
print(f"Best solution: {model.g_best.solution}")        # Encoded solution
print(f"Best fitness: {model.g_best.target.fitness}")
print(f"Best real scheduling: {model.problem.decode_solution(model.g_best.solution)['placement_var'].reshape((num_tasks, num_resources))}")
