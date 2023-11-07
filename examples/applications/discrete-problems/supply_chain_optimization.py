#!/usr/bin/env python
# Created by "Thieu" at 17:03, 07/11/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

# Supply chain optimization refers to the process of maximizing the efficiency and effectiveness of a supply chain network. It involves making
# strategic decisions and implementing operational improvements to optimize the flow of goods, services, and information from suppliers to customers.

# A supply chain encompasses all the activities, processes, and resources involved in delivering a product or service to the end customer.
# It typically includes suppliers, manufacturers, distributors, retailers, and logistics providers. Supply chain optimization aims to
# streamline and enhance various aspects of the supply chain, such as inventory management, production planning, transportation logistics,
# and demand forecasting.

# The primary goals of supply chain optimization are to:
# 1. Minimize Costs: Reduce operational costs associated with inventory holding, transportation, warehousing, and other supply chain activities.
# 2. Improve Customer Service: Enhance customer satisfaction by ensuring timely and accurate product delivery, minimizing stockouts,
#       and improving responsiveness to customer demands.
# 3. Increase Efficiency: Optimize the utilization of resources, reduce waste, and improve the overall efficiency of supply chain processes.
# 4. Enhance Agility: Develop flexibility and responsiveness in the supply chain to adapt to changing market conditions,
#       customer requirements, and unforeseen disruptions.

# To achieve these goals, supply chain optimization involves various strategies and techniques, including:
# 1. Network Design: Determining the optimal configuration of the supply chain network, including the number and location of
#       facilities (e.g., warehouses, distribution centers) and the allocation of demand to these facilities.
# 2. Inventory Optimization: Balancing inventory levels to minimize costs while meeting customer demand. This includes optimizing
#       reorder points, safety stock levels, and order quantities.
# 3. Demand Planning and Forecasting: Accurately predicting future customer demand to optimize production and
#       procurement plans, minimizing stockouts and overstock situations.
# 4. Transportation and Logistics Optimization: Optimizing transportation routes, modes, and carriers to minimize transportation costs,
#       reduce delivery lead times, and improve overall logistics efficiency.
# 5. Supplier Relationship Management: Developing strategies to collaborate closely with suppliers, negotiate favorable terms,
#       and improve visibility into the supplier base for better demand planning and risk management.
# 6. Performance Metrics and Analytics: Implementing key performance indicators (KPIs) and data analytics to monitor and measure
#       supply chain performance, identify areas for improvement, and support decision-making processes.


#######=================================== ANALYZE AN EXAMPLE ===============================

# Consider a manufacturing company that produces and distributes consumer electronics products. The company has multiple manufacturing plants,
# distribution centers, and retail stores located in different regions. The goal is to optimize the supply chain to minimize costs while
# meeting customer demand and ensuring timely product delivery.
#
# 1. Network Design: The company can use supply chain optimization techniques to determine the optimal location and number of manufacturing
# plants and distribution centers. By considering factors such as production costs, transportation costs, and customer demand patterns, the
# company can design an efficient network that minimizes overall costs while meeting service level requirements.
#
# 2. Inventory Optimization: The company can optimize inventory levels to minimize carrying costs while avoiding stockouts and ensuring
# product availability. By analyzing demand patterns, lead times, and production capacities, the company can determine optimal reorder
# points, safety stock levels, and replenishment strategies for each product and location.
#
# 3. Production Planning: Supply chain optimization can help in optimizing the production planning process. By considering factors such as
# production capacities, raw material availability, and customer demand forecasts, the company can create an optimal production plan that
# minimizes costs, maximizes resource utilization, and meets customer demand in an efficient manner.
#
# 4. Transportation and Logistics Optimization: The company can optimize its transportation and logistics operations by selecting the most
# cost-effective transportation modes, routes, and carriers. By considering factors such as transportation costs, transit times, and service
# level requirements, the company can design an optimal transportation network that minimizes costs and ensures timely product delivery to customers.
#
# 5. Supplier Management: Supply chain optimization can help in managing supplier relationships effectively. By analyzing supplier performance,
# lead times, and costs, the company can identify the most reliable and cost-effective suppliers. It can also optimize order quantities,
# delivery schedules, and payment terms to balance costs and service levels.
#
# 6. Demand Planning and Forecasting: Accurate demand planning and forecasting are crucial for supply chain optimization. By analyzing
# historical sales data, market trends, and customer insights, the company can improve demand forecasts, resulting in better inventory
# management, production planning, and resource allocation.

# By implementing supply chain optimization strategies and leveraging advanced analytical tools, the manufacturing company can achieve several benefits, including:

# 1. Reduced costs through optimized inventory levels, production planning, and transportation logistics.
# 2. Improved customer service by ensuring product availability and timely delivery.
# 3. Enhanced operational efficiency through better resource utilization and reduced waste.
# 4. Increased agility and ability to respond to market dynamics and customer demands.
# 5. Improved visibility and control over the supply chain network and processes.
# 6. Better decision-making through data-driven insights and analytics.

## Overall, supply chain optimization plays a vital role in improving the efficiency, responsiveness, and profitability of manufacturing
# companies by optimizing various aspects of the supply chain and aligning them with business objectives and customer requirements.


######================================================ EXAMPLE =================================================

# Let's assume we have a supply chain network with 5 distribution centers (DC1, DC2, DC3, DC4, DC5) and 10 products (P1, P2, P3, ..., P10).
# Our goal is to determine the optimal allocation of products to the distribution centers in a way that minimizes the total transportation cost.

# Each solution represents an allocation of products to distribution centers.
# We can use a binary matrix with dimensions (10, 5) where each element (i, j) represents whether product i is allocated to distribution center j.
# For example, a chromosome [1, 0, 1, 0, 1] would mean that product 1 is allocated to DC1, DC3, DC5.

#  We can add the maximum capacity of each distribution center, therefor we need penalty term to the fitness evaluation function to penalize
#  solutions that violate this constraint. The penalty can be based on the degree of violation or a fixed penalty value.


import numpy as np
from mealpy import BinaryVar, WOA, Problem

# Define the problem parameters
num_products = 10
num_distribution_centers = 5

# Define the transportation cost matrix (randomly generated for the example)
transportation_cost = np.random.randint(1, 10, size=(num_products, num_distribution_centers))

data = {
    "num_products": num_products,
    "num_distribution_centers": num_distribution_centers,
    "transportation_cost": transportation_cost,
    "max_capacity": 4,      # Maximum capacity of each distribution center
    "penalty": 1e10         # Define a penalty value for maximum capacity of each distribution center
}


class SupplyChainProblem(Problem):
    def __init__(self, bounds=None, minmax=None, data=None, **kwargs):
        self.data = data
        super().__init__(bounds, minmax, **kwargs)

    def obj_func(self, x):
        x_decoded = self.decode_solution(x)
        x = x_decoded["placement_var"].reshape((self.data["num_products"], self.data["num_distribution_centers"]))

        if np.any(np.all(x==0, axis=1)):    # If any row has all 0 value, it indicates that this product is not allocated to any distribution center.
            return 0

        total_cost = np.sum(self.data["transportation_cost"] * x)
        # Penalty for violating maximum capacity constraint
        excess_capacity = np.maximum(np.sum(x, axis=0) - self.data["max_capacity"], 0)
        penalty = np.sum(excess_capacity)
        # Calculate fitness value as the inverse of the total cost plus the penalty
        fitness = 1 / (total_cost + penalty)
        return fitness


bounds = BinaryVar(n_vars=num_products * num_distribution_centers, name="placement_var")
problem = SupplyChainProblem(bounds=bounds, minmax="max", data=data)

model = WOA.OriginalWOA(epoch=50, pop_size=20)
model.solve(problem)

print(f"Best agent: {model.g_best}")                    # Encoded solution
print(f"Best solution: {model.g_best.solution}")        # Encoded solution
print(f"Best fitness: {model.g_best.target.fitness}")
print(f"Best real scheduling: {model.problem.decode_solution(model.g_best.solution)['placement_var'].reshape((num_products, num_distribution_centers))}")
