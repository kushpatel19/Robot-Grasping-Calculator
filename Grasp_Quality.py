# Introduction to Robot Grasping
# Homework- 2
# Question- 1
############## Author : Kush Patel ################
'''
Grasp Quality Evaluator

This script evaluates the grasp quality for a multi-fingered robot hand
holding a circular object at specified contact points in 2D/3D. It calculates
four grasp quality metrics: Q_MSV(G), Q_EV(G), Q_GI(G), & Q_GPRI() and finds the best finger
configurations according to each of quality measures.
'''
# Import necessary libraries
import numpy as np
from grasp_matrix_circle import GraspMatrixCalculator
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class GraspQualityEvaluator:    # Create a class for evaluating grasp quality
    def __init__(self, grasp_matrix_calculator):
        self.grasp_matrix_calculator = grasp_matrix_calculator

    def calculate_grasp_polygon_regularity_index(self, contact_points): # Calculate the internal angles of the grasp polygon formed by connecting contact points
        k = len(contact_points)
        internal_angles = [] # Initialize a list to store the internal angles of the grasp polygon
        for i in range(k):   # Loop through each vertex of the grasp polygon
            # Calculate the vectors between consecutive contact points
            v1 = contact_points[i]
            v2 = contact_points[(i + 1) % k]   # Wrap around to the first point when reaching the end
            v3 = contact_points[(i + 2) % k]   # Wrap around to the first point and skip one
            vec1 = np.array(v2) - np.array(v1) # Calculate the vectors from v1 and v2
            vec2 = np.array(v3) - np.array(v1) # Calculate the vectors from v1 and v3
            dot_product = np.dot(vec1, vec2)   # Calculate the dot product between the vectors
            magnitude1 = np.linalg.norm(vec1)  # Calculate the magnitudes of the vector 1
            magnitude2 = np.linalg.norm(vec2)  # Calculate the magnitudes of the vector 1
            if magnitude1 < 1e-6 or magnitude2 < 1e-6:  # Check if either magnitude is close to zero (avoid division by zero)
                continue
            angle = np.arccos(np.clip((dot_product / (magnitude1 * magnitude2)), -1, 1))  # Calculate the angle between the two vectors using the dot product formula
            internal_angles.append((angle))                             # Append the angle to the list of internal angles after converting it to degrees
        
        internal_angles = np.array(internal_angles)                     # Convert internal_angles to a NumPy array for further calculations
        gamma = np.pi - 2 * np.pi / k                                   # Calculate the parameter 'gamma' needed for the regularity index formula
        theta_max = (k - 2) * (np.pi - gamma) + 2 * gamma               # Calculate the parameter 'theta_max' needed for the regularity index formula
        regularity_index = (1 / theta_max) * np.sum(np.abs(internal_angles - gamma)) # Calculate the grasp polygon regularity index (Q_GPRI) using the formula
        return regularity_index

    def calculate_grasp_quality(self):
        # Initialize a dictionary to store the best configurations for Q_MSV(G) and Q_EV(G)
        best_configurations = {
            'Q_MSV(G)': {
                'value': 0,
                'configuration': None
            },
            'Q_EV(G)': {
                'value': 0,
                'configuration': None
            },
            'Q_GI(G)': {  # Add Q_GI(G) metric
                'value': 0,
                'configuration': None
            },
            'Grasp polygon regularity index': {  # Add Q_GPRI
                'value': float('inf'),  # Initialize with positive infinity
                'configuration': None
            }
        }
        Q_MSV_values = []
        Q_MSV_angles = []
        Q_EV_values = []
        Q_EV_angles = []
        Q_GI_values = []
        Q_GI_angles = []
        GPRI_values = []
        GPRI_angles = []

        def iterate_finger_angles(finger_positions, angle_increment): # Define a generator function to iterate through possible finger angles
            num_fingers = len(finger_positions)
            current_angles = np.array(finger_positions)
            while True:
                yield current_angles
                # Increment the angles for the last finger
                current_angles[-1] += angle_increment
                # Check if the last finger's angle exceeds 360 degrees, and propagate the carry to the previous finger
                for i in range(num_fingers - 1, 0, -1):
                    if current_angles[i] >= 2 * np.pi:
                        current_angles[i] -= 2 * np.pi
                        current_angles[i - 1] += angle_increment
                # Check if all angles have exceeded 360 degrees, indicating completion
                if current_angles[0] >= 2 * np.pi:
                    break

        # Iterate through all possible finger configurations
        for angles in iterate_finger_angles(self.grasp_matrix_calculator.finger_positions, np.deg2rad(30)):
            self.grasp_matrix_calculator.finger_positions = angles  # Update finger positions

            # Calculate grasp matrix for all three fingers
            grasp_matrix = self.grasp_matrix_calculator.calculate_full_grasp_matrix()

            # Calculate Q_MSV(G)
            singular_values = np.linalg.svd(grasp_matrix, compute_uv=False)
            Q_MSV_value = np.min(singular_values)

            # Calculate Q_EV(G)
            Q_EV_value = np.prod(singular_values)

            # Calculate Q_GI(G) 
            sigma_min = np.min(singular_values)
            sigma_max = np.max(singular_values)
            Q_GI_value = sigma_min / sigma_max

            # Calculate Grasp polygon regularity index (Q_GPRI) 
            contact_points = []   # Calculate contact points for all three fingers
            for finger_position in angles:
                contact_points.append(self.grasp_matrix_calculator.calculate_contact_positions(finger_position))
            GPRI_value = self.calculate_grasp_polygon_regularity_index(contact_points)

            # Update best configurations if necessary
            if Q_MSV_value > best_configurations['Q_MSV(G)']['value']:
                best_configurations['Q_MSV(G)']['value'] = Q_MSV_value
                best_configurations['Q_MSV(G)']['configuration'] = {
                    'Finger 1': np.rad2deg(angles[0]),
                    'Finger 2': np.rad2deg(angles[1]),
                    'Finger 3': np.rad2deg(angles[2])
                }

            if Q_EV_value > best_configurations['Q_EV(G)']['value']:
                best_configurations['Q_EV(G)']['value'] = Q_EV_value
                best_configurations['Q_EV(G)']['configuration'] = {
                    'Finger 1': np.rad2deg(angles[0]),
                    'Finger 2': np.rad2deg(angles[1]),
                    'Finger 3': np.rad2deg(angles[2])
                }

            if Q_GI_value > best_configurations['Q_GI(G)']['value']:  # Update for Q_GI(G)
                best_configurations['Q_GI(G)']['value'] = Q_GI_value
                best_configurations['Q_GI(G)']['configuration'] = {
                    'Finger 1': np.rad2deg(angles[0]),
                    'Finger 2': np.rad2deg(angles[1]),
                    'Finger 3': np.rad2deg(angles[2])
                }

            if 0.0 < GPRI_value < best_configurations['Grasp polygon regularity index']['value']:  # Update for GPRI
                best_configurations['Grasp polygon regularity index']['value'] = GPRI_value
                best_configurations['Grasp polygon regularity index']['configuration'] = {
                    'Finger 1': np.rad2deg(angles[0]),
                    'Finger 2': np.rad2deg(angles[1]),
                    'Finger 3': np.rad2deg(angles[2])
                }
            
            # Store values and angles for plotting
            Q_MSV_values.append(Q_MSV_value)
            Q_MSV_angles.append([np.rad2deg(angles[1]), np.rad2deg(angles[2])])
            Q_EV_values.append(Q_EV_value)
            Q_EV_angles.append([np.rad2deg(angles[1]), np.rad2deg(angles[2])])
            Q_GI_values.append(Q_GI_value)
            Q_GI_angles.append([np.rad2deg(angles[1]), np.rad2deg(angles[2])])
            GPRI_values.append(GPRI_value)
            GPRI_angles.append([np.rad2deg(angles[1]), np.rad2deg(angles[2])])
        
        return best_configurations, Q_MSV_values, Q_MSV_angles, Q_EV_values, Q_EV_angles, Q_GI_values, Q_GI_angles, GPRI_values, GPRI_angles
    
    def plot_grasp_quality(self, Q_MSV_values, Q_MSV_angles, Q_EV_values, Q_EV_angles, Q_GI_values, Q_GI_angles, GPRI_values, GPRI_angles):
        # Convert angle variables to NumPy arrays
        Q_MSV_angles = np.array(Q_MSV_angles)
        Q_EV_angles = np.array(Q_EV_angles)
        Q_GI_angles = np.array(Q_GI_angles)
        GPRI_angles = np.array(GPRI_angles)

        # Create a figure with subplots
        fig = plt.figure(figsize=(15, 5))

        # Create a 3D scatter plot for data points
        ax1 = fig.add_subplot(141, projection='3d')
        ax1.set_xlabel('Finger 2')
        ax1.set_ylabel('Finger 3')
        ax1.set_zlabel('Value')
        ax1.set_title('Q_MSV(G)')
        ax1.scatter(Q_MSV_angles[:, 0], Q_MSV_angles[:, 1], Q_MSV_values, c=Q_MSV_values, cmap='viridis')
        fig.colorbar(ax1.scatter(Q_MSV_angles[:, 0], Q_MSV_angles[:, 1], Q_MSV_values, c=Q_MSV_values, cmap='viridis'), ax=ax1, shrink=0.5, aspect=20)
        
        ax2 = fig.add_subplot(142, projection='3d')
        ax2.set_xlabel('Finger 2')
        ax2.set_ylabel('Finger 3')
        ax2.set_zlabel('Value')
        ax2.set_title('Q_EV(G)')
        ax2.scatter(Q_EV_angles[:, 0], Q_EV_angles[:, 1], Q_EV_values, c=Q_EV_values, cmap='viridis')
        fig.colorbar(ax2.scatter(Q_EV_angles[:, 0], Q_EV_angles[:, 1], Q_EV_values, c=Q_EV_values, cmap='viridis'), ax=ax2, shrink=0.5, aspect=20)
        
        ax3 = fig.add_subplot(143, projection='3d')
        ax3.set_xlabel('Finger 2')
        ax3.set_ylabel('Finger 3')
        ax3.set_zlabel('Value')
        ax3.set_title('Q_GI(G)')
        ax3.scatter(Q_GI_angles[:, 0], Q_GI_angles[:, 1], Q_GI_values, c=Q_GI_values, cmap='viridis')
        fig.colorbar(ax3.scatter(Q_GI_angles[:, 0], Q_GI_angles[:, 1], Q_GI_values, c=Q_GI_values, cmap='viridis'), ax=ax3, shrink=0.5, aspect=20)
        
        ax4 = fig.add_subplot(144, projection='3d')
        ax4.set_xlabel('Finger 2')
        ax4.set_ylabel('Finger 3')
        ax4.set_zlabel('Value')
        ax4.set_title('Grasp Polygon Regularity Index (Q_GPRI)')
        ax4.scatter(GPRI_angles[:, 0], GPRI_angles[:, 1], GPRI_values, c=GPRI_values, cmap='viridis')
        fig.colorbar(ax4.scatter(GPRI_angles[:, 0], GPRI_angles[:, 1], GPRI_values, c=GPRI_values, cmap='viridis'), ax=ax4, shrink=0.5, aspect=20)

        plt.show()

# Create a GraspMatrixCalculator instance with appropriate parameters
k = 3  # Number of fingers
m = 6  # Degrees of freedom (3 for 2D, 6 for 3D)
p = 3  # Number of independent forces/torques
euler_angles = [0.0, 0.0, 0.0]     # Replace with actual Euler angles
finger_positions = [0.0, 0.0, 0.0]  # Replace with actual initial finger positions w.r.t angle
object_radius = 1.0                # Radius of the circular object (should be same as in grasp_matrix_circle.py script)

# Create an instance of the GraspMatrixCalculator class with the specified parameters
grasp_matrix_calculator = GraspMatrixCalculator(k, m, p, euler_angles, finger_positions)

# Create an instance of the GraspQualityEvaluator class with the GraspMatrixCalculator instance
grasp_quality_evaluator = GraspQualityEvaluator(grasp_matrix_calculator)

# Calculate the best finger configurations 
best_configurations, Q_MSV_values, Q_MSV_angles, Q_EV_values, Q_EV_angles, Q_GI_values, Q_GI_angles, GPRI_values, GPRI_angles = grasp_quality_evaluator.calculate_grasp_quality()

# Print the results
print("Best Configuration for Q_MSV(G):")
print("Value:", best_configurations['Q_MSV(G)']['value'])
print("Configuration:", {key: f'{value:.2f}' for key, value in best_configurations['Q_MSV(G)']['configuration'].items()})

print("\nBest Configuration for Q_EV(G):")
print("Value:", best_configurations['Q_EV(G)']['value'])
print("Configuration:", {key: f'{value:.2f}' for key, value in best_configurations['Q_EV(G)']['configuration'].items()})

print("\nBest Configuration for Q_GI(G):")  # Print the results for Q_GI(G)
print("Value:", best_configurations['Q_GI(G)']['value'])
print("Configuration:", {key: f'{value:.2f}' for key, value in best_configurations['Q_GI(G)']['configuration'].items()})

print("\nBest Configuration for Grasp polygon regularity index (Q_GPRI):")
print("Value:", best_configurations['Grasp polygon regularity index']['value'])
print("Configuration:", {key: f'{value:.2f}' for key, value in best_configurations['Grasp polygon regularity index']['configuration'].items()})

# Plot the grasp quality metrics
grasp_quality_evaluator.plot_grasp_quality(Q_MSV_values, Q_MSV_angles, Q_EV_values, Q_EV_angles, Q_GI_values, Q_GI_angles, GPRI_values, GPRI_angles)
