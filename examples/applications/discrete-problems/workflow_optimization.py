#!/usr/bin/env python
# Created by "Thieu" at 18:43, 07/11/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

# Workflow optimization refers to the process of improving the efficiency, productivity, and effectiveness of workflows within an organization
# or system. A workflow represents a series of interconnected tasks or activities that are performed to achieve a specific outcome or goal.

# A workflow optimization problem involves analyzing and optimizing various aspects of a workflow, such as task allocation, task sequencing,
# resource allocation, and process automation. The objective is to streamline the flow of work, eliminate bottlenecks, reduce delays,
# minimize resource utilization, and enhance overall performance.

# Workflow optimization problems can arise in various domains and industries, including manufacturing, healthcare, logistics,
# software development, finance, and many others. Here are a few examples of workflow optimization problems in different contexts:

# 1. Manufacturing Workflow Optimization:
#   + Optimizing the sequence of production tasks to minimize production time and maximize resource utilization.
#   + Allocating resources (e.g., machines, labor) to different production tasks to achieve the highest efficiency.
#   + Optimizing inventory management and supply chain processes to reduce lead times and minimize stockouts.

# 2. Healthcare Workflow Optimization:
#   + Optimizing patient scheduling and appointment sequencing to reduce waiting times and improve patient satisfaction.
#   + Allocating healthcare professionals and resources efficiently to different patient care tasks.
#   + Streamlining administrative processes, such as patient registration and documentation, to reduce paperwork and improve data accuracy.

# 3. Software Development Workflow Optimization:
#   + Optimizing the allocation of development tasks to software engineers to maximize productivity and minimize project delays.
#   + Streamlining the code review and testing processes to ensure high-quality software and faster delivery.
#   + Implementing agile methodologies and continuous integration/continuous delivery (CI/CD) practices to streamline development and deployment workflows.

# 4. Financial Workflow Optimization:
#   + Optimizing the approval and authorization processes for financial transactions to reduce processing time and improve accuracy.
#   + Automating routine financial tasks, such as invoice processing and expense management, to improve efficiency and reduce manual errors.
#   + Optimizing cash flow management and forecasting processes to ensure optimal utilization of financial resources.

# Workflow optimization problems can be addressed using various optimization techniques, such as mathematical programming, simulation,
# machine learning, and process mining. The specific approach depends on the complexity of the workflow, available data, problem constraints,
# and the desired optimization objectives.


## Let's consider an example of healthcare workflow optimization in a hospital setting. One common challenge in healthcare is optimizing
# the patient flow and resource allocation to ensure efficient and timely delivery of care. Here's an example scenario:

# Scenario: Emergency Department (ED) Workflow Optimization

# Problem: The emergency department of a hospital is experiencing high patient wait times, overcrowding, and inefficient
#   resource allocation. The goal is to optimize the workflow to reduce patient waiting times, improve resource utilization,
#   and enhance the overall efficiency of the emergency department.

# Solution Approach:
#   1. Patient Triage: Implement an efficient patient triage system to categorize patients based on the severity of their condition.
#       This helps prioritize patients who require immediate attention, ensuring that critical cases are attended to promptly.
#   2. Resource Allocation: Analyze the availability and utilization of resources, such as doctors, nurses, examination rooms, and medical equipment.
#       Optimize the allocation of resources based on patient demand and the severity of their conditions. This may involve adjusting staff schedules,
#       reallocating resources between different areas of the emergency department, or identifying bottlenecks in resource availability.
#   3. Workflow Redesign: Analyze the current patient flow and identify bottlenecks or inefficiencies. Redesign the workflow to streamline
#       the patient journey from arrival to discharge. This may involve optimizing the sequence of activities, improving communication
#       between departments, and implementing standardized protocols for common procedures.
#   4. Capacity Planning: Analyze historical data to understand patient arrival patterns, peak hours, and seasonal variations. Use this
#       information to plan for capacity adjustments, such as staffing levels and resource availability, during peak demand periods.
#       This helps ensure that sufficient resources are available to handle the expected patient volume.
#   5. Process Automation: Identify routine administrative tasks that can be automated to reduce manual effort and improve efficiency.
#       For example, automate patient registration, data entry, and documentation processes to free up staff time for patient care activities.
#   6. Data Analytics: Utilize data analytics techniques to monitor key performance indicators, track patient flow metrics, and identify
#       areas for further improvement. This includes analyzing patient wait times, length of stay, resource utilization rates, and patient
#       satisfaction scores. Data-driven insights can help identify trends, patterns, and potential areas of optimization.
#   7. Continuous Improvement: Establish a culture of continuous improvement by regularly reviewing and evaluating the effectiveness of implemented changes.
#       Solicit feedback from staff, patients, and stakeholders to identify further opportunities for optimization and address any unforeseen challenges.

# Define a chromosome representation that encodes the allocation of resources and patient flow in the emergency department.
# This could be a binary matrix where each row represents a patient and each column represents a resource (room).
# If the element is 1, it means the patient is assigned to that particular room, and if the element is 0, it means the patient is not assigned to that room

# Please note that this implementation is a basic template and may require further customization based on the specific objectives, constraints,
# and evaluation criteria of your healthcare workflow optimization problem. You'll need to define the specific fitness
# function and optimization objectives based on the factors you want to optimize, such as patient waiting times, resource utilization, and
# other relevant metrics in the healthcare workflow context.


import numpy as np
from mealpy import BinaryVar, WOA, Problem

# Define the problem parameters
num_patients = 50  # Number of patients
num_resources = 10  # Number of resources (room)

# Define the patient waiting time matrix (randomly generated for the example)
# Why? May be, doctors need time to prepare tools,...
waiting_matrix = np.random.randint(1, 10, size=(num_patients, num_resources))

data = {
    "num_patients": num_patients,
    "num_resources": num_resources,
    "waiting_matrix": waiting_matrix,
    "max_resource_capacity": 10,        # Maximum capacity of each room
    "max_waiting_time": 60,             # Maximum waiting time
    "penalty_value": 1e2,         # Define a penalty value
    "penalty_patient": 1e10
}


class SupplyChainProblem(Problem):
    def __init__(self, bounds=None, minmax=None, data=None, **kwargs):
        self.data = data
        super().__init__(bounds, minmax, **kwargs)

    def obj_func(self, x):
        x_decoded = self.decode_solution(x)
        x = x_decoded["placement_var"].reshape(self.data["num_patients"], self.data["num_resources"])

        # If any row has all 0 value, it indicates that this patient is not allocated to any room.
        # If a patient a assigned to more than 3 room, not allow
        if np.any(np.all(x==0, axis=1)) or np.any(np.sum(x>3, axis=1)):
            return self.data["penalty_patient"]

        # Calculate fitness based on optimization objectives
        room_used = np.sum(x, axis=0)
        wait_time = np.sum(x * self.data["waiting_matrix"], axis=1)
        violated_constraints = np.sum(room_used > self.data["max_resource_capacity"]) + np.sum(wait_time > self.data["max_waiting_time"])

        # Calculate the fitness value based on the objectives
        resource_utilization_fitness = 1 - np.mean(room_used) / self.data["max_resource_capacity"]
        waiting_time_fitness = 1 - np.mean(wait_time) / self.data["max_waiting_time"]

        fitness = resource_utilization_fitness + waiting_time_fitness + self.data['penalty_value'] * violated_constraints
        return fitness


bounds = BinaryVar(n_vars=num_patients * num_resources, name="placement_var")
problem = SupplyChainProblem(bounds=bounds, minmax="min", data=data)

model = WOA.OriginalWOA(epoch=50, pop_size=20)
model.solve(problem)

print(f"Best agent: {model.g_best}")                    # Encoded solution
print(f"Best solution: {model.g_best.solution}")        # Encoded solution
print(f"Best fitness: {model.g_best.target.fitness}")
print(f"Best real scheduling: {model.problem.decode_solution(model.g_best.solution)['placement_var'].reshape((num_patients, num_resources))}")
