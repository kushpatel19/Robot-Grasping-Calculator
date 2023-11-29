# Introduction to Robot Grasping
# Homework- 1
# Question- 1
############## Author : Kush Patel ################

'''
Grasp Matrix Calculator

This script calculates the grasp matrix for a multi-fingered robot hand
holding an object with a given surface equation of any random obejct at specified contact points.
It can visualize the object's surface and contact points in 3D.

Copyright 2023 Kush Patel

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class GraspMatrixCalculator:
    
    def __init__(self, k, m, p, euler_angles, object_surface_equation, contact_points):
        # Initialize the GraspMatrixCalculator with provided parameters
        self.k = int(k)                           # Number of fingers
        self.m = int(m)                           # Degrees of freedom (3 for 2D, 6 for 3D)
        self.p = int(p)                           # Number of independent forces/torques
        self.euler_angles = euler_angles          # Input Euler angles for orientation
        self.object_surface_equation = object_surface_equation # Equation of the object surface
        self.contact_points = contact_points      # List of contact points in object coordinates
        self.J_ = np.array([[0, -1, -1],
                            [1, 0, -1],
                            [1, 1, 0]])
        if len(contact_points) < k:
            print(f"Warning: There are fewer contact points ({len(contact_points)}) than fingers ({k}).")
            print("Please provide enough contact points for each finger.")
        self.rotation_matrix = self.calculate_rotation_matrix() # Initialize rotation matrix based on Euler angles

    def calculate_rotation_matrix(self):
        phi, theta, psi = self.euler_angles       # Extract Euler angles
        # Define the rotation matrices for each axis
        R_x = np.array([[1, 0, 0],
                        [0, np.cos(phi), -np.sin(phi)],
                        [0, np.sin(phi), np.cos(phi)]])
 
        R_y = np.array([[np.cos(theta), 0, np.sin(theta)],
                        [0, 1, 0],
                        [-np.sin(theta), 0, np.cos(theta)]])
 
        R_z = np.array([[np.cos(psi), -np.sin(psi), 0],
                        [np.sin(psi), np.cos(psi), 0],
                        [0, 0, 1]])
        rotation_matrix = np.dot(R_z, np.dot(R_y, R_x)) # Calculate the rotation matrix by multiplying the individual matrices
        return rotation_matrix

    def calculate_normal_vector(self, contact_point):
        if len(contact_point) != 3:
            print("Error: Contact point should have three coordinates (x, y, z).")
            return None                                          # Return None to indicate an error
        x, y, z = sp.symbols('x y z')                            # Create symbolic variables for x, y, and z coordinates
        surface_equation = self.object_surface_equation(x, y, z) # Define the symbolic expression for the object's surface equation
        gradient = [sp.diff(surface_equation, var) for var in (x, y, z)]   # Calculate the gradient of the surface equation
        gradient_values = [gradient[0].subs({x: contact_point[0], y: contact_point[1], z: contact_point[2]}),
                           gradient[1].subs({x: contact_point[0], y: contact_point[1], z: contact_point[2]}),
                           gradient[2].subs({x: contact_point[0], y: contact_point[1], z: contact_point[2]})]   # Evaluate the gradient at the given contact point
        gradient_values = [int(x) for x in gradient_values]      # Convert each element to an integer using a list comprehension
        gradient_magnitude = np.linalg.norm(gradient_values)     # Normalize the gradient vector
        if gradient_magnitude == 0:
            print("Error: The gradient magnitude is zero at the contact point.")
            return None  # Return None to indicate an error
        normalized_gradient = [value / gradient_magnitude for value in gradient_values]
        return normalized_gradient

    def calculate_contact_frames(self):
        contact_frames = []
        for position in self.contact_points:            
            n = self.calculate_normal_vector(position)       # Calculate the normal vector (n) based on the object surface equation and contact point
            n = [-x for x in n]                              # Reverse the direction of the vector by multiplying it by -1
            n = np.array(n)                                  # Convert the type from list to numpy array
            if np.allclose(n, [0, 1, 0]) or np.allclose(n, [0, -1, 0]):  # Choose an arbitrary initial vector (t0) that is not collinear with n
                t0 = np.array([1, 0, 0], dtype=float)
            else:
                t0 = np.array([0, 1, 0], dtype=float)            
            t0 -= np.dot(t0, n) * n                           # Ensure t0 is orthogonal to n
            t0 /= np.linalg.norm(t0)                          # Normalize t0 to ensure it's a unit vector
            s = np.cross(t0, n)                               # Calculate the second vector (s) that satisfies s x t = n or (t x n = s)
            s /= np.linalg.norm(s)                            # Normalize s to ensure it's a unit vector
            n /= np.linalg.norm(n)                            # Normalize n to ensure it's a unit vector
            t = np.cross(n, s)                                # Calculate the final vector (t) orthogonal to n and s
            n = self.rotation_matrix @ n                      # Convert the n vector from body frame to world frame
            t = self.rotation_matrix @ t                      # Convert the n vector from body frame to world frame
            s = self.rotation_matrix @ s                      # Convert the n vector from body frame to world frame  
            contact_frame = np.column_stack((s, t, n))        # Append the contact frame (s, t, n) to the list
            contact_frames.append(contact_frame)
        if len(contact_frames) < self.k:
            print(f"Warning: There are fewer contact frames ({len(contact_frames)}) than fingers ({self.k}).")
            print("Please provide enough contact points for each finger.")
        return contact_frames

    def calculate_body_vector(self, poses):
        body_vectors = []
        for pos in poses:
            bi = pos                      # Calculate the vector bi from the center of the circle to the contact point
            body_vectors.append(bi)
        B = np.column_stack(body_vectors) # Stack the body vectors to create the matrix B
        if B.shape[1] < self.k:
            print(f"Warning: There are fewer body vectors ({B.shape[1]}) than fingers ({self.k}).")
            print("Please provide enough body vectors for each finger.")
        return B

    def calculate_grasp_submatrix(self, contact_frames):
        p = self.p                                  # Get the number of independent forces/torques (p)
        k = self.k
        grasp_submatrices = []
        for i in range(k):                          # Create the grasp submatrix based on the contact type and p
            contact_frame = contact_frames[i]
            
            if self.m == 3:
                if p == 3:
                    grasp_submatrix = np.zeros((6,p))
                else:
                    grasp_submatrix = np.zeros((4,p))
                if p == 1:
                    grasp_submatrix[0:3, 0] = contact_frame[:, 2]  # Normal direction (n)
                    grasp_submatrix[3, 0] = np.dot(self.J_ @ (self.rotation_matrix @ self.B[:, i]), contact_frame[:, 2])
                elif p == 2:
                    grasp_submatrix[0:3, 0] = contact_frame[:, 1]  # Tangential direction (t)
                    grasp_submatrix[3, 0] = np.dot(self.J_ @ (self.rotation_matrix @ self.B[:, i]), contact_frame[:, 1])
                    grasp_submatrix[0:3, 1] = contact_frame[:, 2]  # Normal direction (n)
                    grasp_submatrix[3, 1] = np.dot(self.J_ @ (self.rotation_matrix @ self.B[:, i]), contact_frame[:, 2])
                elif p == 3:
                    grasp_submatrix[0:3, 0] = contact_frame[:, 0]  # Tangential direction (s)
                    grasp_submatrix[3:6, 0] = np.cross(self.rotation_matrix @ self.B[:, i], contact_frame[:, 0])
                    grasp_submatrix[0:3, 1] = contact_frame[:, 1]  # Tangential direction (t)
                    grasp_submatrix[3:6, 1] = np.cross(self.rotation_matrix @ self.B[:, i], contact_frame[:, 1])
                    grasp_submatrix[0:3, 2] = np.zeros((len(contact_frames[0][:,0]),1))[:,0]  # zero vector
                    grasp_submatrix[3:6, 2] = contact_frame[:, 2]  # Normal direction (n)
            
            elif self.m == 6:
                grasp_submatrix = np.zeros((self.m,p))
                if p== 1:
                    grasp_submatrix[0:3, 0] = contact_frame[:, 2]  # Normal direction (n)
                    grasp_submatrix[3:6, 0] = np.cross(self.rotation_matrix @ self.B[:, i], contact_frame[:, 2])
                elif p == 2:
                    grasp_submatrix[0:3, 0] = contact_frame[:, 1]  # Tangential direction (t)
                    grasp_submatrix[3:6, 0] = np.cross(self.rotation_matrix @ self.B[:, i], contact_frame[:, 1])
                    grasp_submatrix[0:3, 1] = contact_frame[:, 2]  # Normal direction (n)
                    grasp_submatrix[3:6, 1] = np.cross(self.rotation_matrix @ self.B[:, i], contact_frame[:, 2])
                elif p == 3:
                    grasp_submatrix[0:3, 0] = contact_frame[:, 0]  # Tangential direction (s)
                    grasp_submatrix[3:6, 0] = np.cross(self.rotation_matrix @ self.B[:, i], contact_frame[:, 0])
                    grasp_submatrix[0:3, 1] = contact_frame[:, 1]  # Tangential direction (t)
                    grasp_submatrix[3:6, 1] = np.cross(self.rotation_matrix @ self.B[:, i], contact_frame[:, 1])
                    grasp_submatrix[0:3, 2] = contact_frame[:, 2]  # Normal direction (n)
                    grasp_submatrix[3:6, 2] = np.cross(self.rotation_matrix @ self.B[:, i], contact_frame[:, 2])
                elif p == 4:
                    grasp_submatrix[0:3, 0] = contact_frame[:, 0]  # Tangential direction (s)
                    grasp_submatrix[3:6, 0] = np.cross(self.rotation_matrix @ self.B[:, i], contact_frame[:, 0])
                    grasp_submatrix[0:3, 1] = contact_frame[:, 1]  # Tangential direction (t)
                    grasp_submatrix[3:6, 1] = np.cross(self.rotation_matrix @ self.B[:, i], contact_frame[:, 1])
                    grasp_submatrix[0:3, 2] = contact_frame[:, 2]  # Normal direction (n)
                    grasp_submatrix[3:6, 2] = np.cross(self.rotation_matrix @ self.B[:, i], contact_frame[:, 2])
                    grasp_submatrix[0:3, 3] = np.zeros((len(contact_frames[0][:,0]),1))[:,0]  # zero vector
                    grasp_submatrix[3:6, 3] = contact_frame[:, 2]  # Normal direction (n)
            grasp_submatrices.append(grasp_submatrix)
        return grasp_submatrices

    def calculate_full_grasp_matrix(self):
        contact_frames = self.calculate_contact_frames()
        self.B = self.calculate_body_vector(self.contact_points)           # Calculate B
        grasp_submatrices = self.calculate_grasp_submatrix(contact_frames) # Calculate grasp submatrices for all fingers
        full_grasp_matrix = np.hstack(grasp_submatrices)                   # Combine grasp submatrices to create the full grasp matrix
        return full_grasp_matrix

    def visualize_grasp(self):
        fig = plt.figure() # Create a 3D plot
        ax = fig.add_subplot(111, projection='3d')
        x = np.linspace(-2, 2, 50)  # Adjust the range as needed
        y = np.linspace(-2, 2, 50)  # Adjust the range as needed

        X, Y= np.meshgrid(x, y)
        Z = self.object_surface_equation(X, Y, 0)            # Calculate the Z values using the object's surface equation
        ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.5)  # Plot the object's surface & Make the object surface slightly transparent
        contact_points = np.array(self.contact_points)       # Plot the contact points on the object's surface
        ax.scatter(contact_points[:, 0], contact_points[:, 1], contact_points[:, 2], c='red', marker='o', s=100)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()         # Show the plot

k = 2                               # Number of fingers
m = 6                               # Dimentionality 2D / 3D
p = 3                               # Number of independant forces/torque based on type of contact
euler_angles = [np.deg2rad(0.0), np.deg2rad(0.0), np.deg2rad(0.0)]      # Replace with actual Euler angles
body_origin  = [0.0, 0.0, 0.0]    # Origin of the body frame
object_surface_equation = lambda x, y, z: (x - body_origin[0])**2 + (y - body_origin[1])**2 - 1        # Object's surface equation (w.r.t. body frame)
# R = 3.0  # Major radius
# r = 1.0  # Minor radius
# object_surface_equation = lambda x, y, z: (x**2 + y**2 + z**2 + R**2 - r**2)**2 - 4 * (x**2 + y**2) * (R**2 - r**2)
contact_points = [[-1.0, 0.0, 0.0], 
                  [0.0, 1.0, 0.0]]  # Example: Contact points on the sphere (w.r.t. body frame)

# Create the GraspMatrixCalculator instance with error checking
if k > len(contact_points):
    print("Error: Number of fingers exceeds the number of contact points.")
    print("Please provide more contact points or reduce the number of fingers.")
else:
    grasp_calculator = GraspMatrixCalculator(k, m, p, euler_angles, object_surface_equation, contact_points)
    grasp_matrix = grasp_calculator.calculate_full_grasp_matrix()
    if m == 3 and p != 3: # After calculating the full grasp matrix, check the dimensionality (m)
        grasp_matrix = np.delete(grasp_matrix, [2], axis=0)          # Remove the 3rd row (index 2) 
    elif m == 3 and p == 3:
        grasp_matrix = np.delete(grasp_matrix, [2,4], axis=0)          # Remove the 3rd row (index 2) and the 5th row (index 4)
    rounded_grasp_matrix = np.round(grasp_matrix, 3)    # Round off all values in the matrix to 3 decimal places
    matrix_str = '\n'.join(['\t'.join([f'{cell:.3f}' for cell in row]) for row in rounded_grasp_matrix])  # Convert the matrix to a human-readable string with formatting

    # Print the matrix to the console
    print('Grasp Matrix :-')
    print(matrix_str)

    # Plot the object and contact points
    grasp_calculator.visualize_grasp()