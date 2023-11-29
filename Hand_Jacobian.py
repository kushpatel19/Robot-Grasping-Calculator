# Introduction to Robot Grasping
# Homework- 1
# Question- 2
############## Author : Kush Patel ################

'''
Hand Jacobian Calculator

This script calculates the hand Jacobian matrix for a multi-fingered robot hand holding an object. 
The robot hand is a serial chain robotic manipulator with specified joint angles and contact positions on the object's surface.
The script computes the hand Jacobian to describe the relationship between joint velocities and the linear and angular velocities of the object.

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

class HandJacobianCalculator:
    
    def __init__(self, k, N, p, n_values, z_values, euler_angles, body_origin, object_surface_equation, finger_base_orientations, finger_joint_angles, contact_positions, link_lengths):  # Initialize the HandJacobianCalculator with provided parameters
        self.k = int(k)                                # Number of fingers
        self.N = int(N)                                # Total number of joints
        self.p = int(p)                                # Number of independent force components
        self.n_values = n_values                       # Number of joints for each finger (list of length k)
        self.z_values = z_values                       # Finger base heights (list of length k)
        self.euler_angles = euler_angles               # Euler angles of object B
        self.object_surface_equation = object_surface_equation    # Equation of the object surface
        self.finger_base_orientations = finger_base_orientations  # Euler angles of each finger base frame (for Rpk_i)
        self.finger_joint_angles = finger_joint_angles # Joint angles of all fingers
        self.contact_positions = contact_positions     # Contact positions of all fingers
        self.link_lengths = link_lengths               # Link lengths of all fingers
        self.body_origin = body_origin                 # Origin of the body frame
        self.hand_jacobian = np.zeros((self.k * self.p, self.N))  # Initialize matrices for hand Jacobian calculation
        self.rotation_matrix = self.calculate_rotation_matrix()   # Initialize rotation matrix based on Euler angles

    def validate_inputs(self):  # Validate all input parameters
        # Check if the number of fingers (k) matches the length of other lists
        if self.k != len(self.n_values) or self.k != len(self.z_values) or self.k != len(self.finger_base_orientations) or self.k != len(self.finger_joint_angles):
            print("Error: The number of fingers (k) does not match the length of other input lists.")
            return False

        # Check if the number of joints for each finger (n_values) matches the sum of all joints (N)
        if sum(self.n_values) != self.N:
            print("Error: The sum of joints in n_values does not match the total number of joints (N).")
            return False

        # Check if the length of finger_joint_angles matches the number of joints for each finger (n_values)
        for i in range(self.k):
            if len(self.finger_joint_angles[i]) != self.n_values[i]:
                print(f"Error: The number of joint angles for finger {i+1} does not match n_values for that finger.")
                return False

        # Check if the length of contact_positions matches the number of fingers (k)
        if len(self.contact_positions) != self.k:
            print("Error: The number of contact positions does not match the number of fingers (k).")
            return False

        # Check if the length of link_lengths matches the number of fingers (k)
        if len(self.link_lengths) != self.k:
            print("Error: The number of link lengths does not match the number of fingers (k).")
            return False

        return True

    def calculate_rotation_matrix(self):  # Calculate the rotation matrix based on Euler angles
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

    def calculate_normal_vector(self, contact_point): # Calculate the normal vector at a contact point on the object's surface
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

    def calculate_contact_frame_basis(self, contact_position):  # Calculate the contact frame basis at a contact position     
        n = self.calculate_normal_vector(contact_position)  # Calculate the normal vector (n) based on the object surface equation and contact point
        n = [-x for x in n]                                 # Reverse the direction of the vector by multiplying it by -1
        n = np.array(n)                                     # Convert the type from list to numpy array
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
        contact_frame = contact_frame.tolist()
        return contact_frame
    
    def calculate_Rpk(self, finger_base_orientation): # Calculate the Rpk matrix for a finger's base orientation
        # Extract Euler angles
        phi   = finger_base_orientation[0]
        theta = finger_base_orientation[1]
        psi   = finger_base_orientation[2]

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

        rotation_matrix = np.dot(R_z, np.dot(R_y, R_x))
        return rotation_matrix
    
    def generate_dh_parameters(self, link_lengths, joint_angles):  # Generate DH parameters for an n-DOF planar robot
        """
        Generates DH parameters for an n-DOF planar robot
        link_lengths: List containing the lengths of the links
        joint_angles: List containing the joint angles
        Returns: List of tuples, each containing the DH parameters (a, alpha, d) for each joint
        """
        if len(link_lengths) != len(joint_angles):
            raise ValueError("Link lengths and joint angles lists must have the same length")
        
        dh_parameters = []
        for i in range(len(link_lengths)):
            a = link_lengths[i]
            theta = joint_angles[i]
            alpha = 0      # alpha is 0 for planar robots
            d = 0          # d is 0 for revolute joints in planar robots
            dh_parameters.append((a, alpha, d, theta))
        return dh_parameters

    def dh_transform(self, a, alpha, d, theta):  # Compute the transformation matrix using DH parameters
        """
        Computes the transformation matrix using DH parameters
        """
        return np.array([
            [np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
            [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
            [0, np.sin(alpha), np.cos(alpha), d],
            [0, 0, 0, 1]
        ])

    def finger_jacobian(self, dh_params):  # Compute the Jacobian matrix
        """
        Computes the Jacobian matrix
        dh_params: List of tuples, each tuple contains the DH parameters (a, alpha, d) for each joint.
        q: List of joint angles.
        """
        num_joints = len(dh_params)
        jacobian = np.zeros((6, num_joints))
        T = np.identity(4)                                 # Initialize to identity matrix
        T_f = T
        for i in range(num_joints):
            a, alpha, d, theta = dh_params[i]
            T_i_i = self.dh_transform(a, alpha, d, theta)  # Transformation matrix from the previous frame to the i-th frame
            T_f = np.matmul(T_f, T_i_i)                    # Overall transformation matrix up to the i-th frame
        p_e = T_f[:3, 3]
        T = np.identity(4)
        for i in range(num_joints):
            a, alpha, d, theta = dh_params[i]
            T_i = self.dh_transform(a, alpha, d, theta)    # Transformation matrix from the previous frame to the i-th frame
            p = T[:3, 3]                                   # Extract the position vector of the i-th frame
            z = T[:3, 2]                                   # Extract the z-axis of the i-th frame (third column of T)
            T = np.matmul(T, T_i)                          # Overall transformation matrix up to the i-th frame
            jacobian[:3, i] = np.cross(z, p_e - p)         # Linear velocity part
            jacobian[3:, i] = z                            # Angular velocity part
        return jacobian[:3,:]

    def calculate_hand_jacobian(self):  # Calculate the hand Jacobian matrix
        # Validate inputs
        if not self.validate_inputs():
            print("Please check your input values and ensure they match the expected format.")
            return None
                
        for i in range(self.k):
            finger_DH = self.generate_dh_parameters(self.link_lengths[i], 
                                                    self.finger_joint_angles[i]) # Calculate individual finger DH parameter
            Ji = self.finger_jacobian(finger_DH)                                 # Calculate individual finger Jacobians
            Wi = self.calculate_contact_frame_basis(self.contact_positions[i])   # Compute the ith contact frame basis Wi(q)
            Rpk_i = self.calculate_Rpk(self.finger_base_orientations[i])         # Calculate Rpk_i for the ith finger
            sub_hand_jacobian = np.dot(np.dot(np.array(Wi).T, Rpk_i), Ji)        # Compute the sub hand Jacobian matrix for the ith finger
            
            # Place the sub hand Jacobian in the correct position in the hand Jacobian matrix
            start_row = i * self.p
            end_row = (i + 1) * self.p
            self.hand_jacobian[start_row:end_row, start_row:end_row] = sub_hand_jacobian[:self.p,:]
        return self.hand_jacobian
    
# Example usage
k = 3                 # Number of fingers
n_values = [3, 3, 3]  # Number of total joints for each finger
N = sum(n_values)     # Total number of joints (sum of joints in all fingers)
p = 3                 # Number of independent force components

# Example input values (replace with actual values)
z_values                 = [0.1, 0.2, 0.3]    # Finger base distances
euler_angles             = [0.1, 0.2, 0.3]    # Euler angles of object B's body frame (in radians)
body_origin              = [0.0, 0.0, 0.0]    # Origin of the body frame
object_surface_equation  = lambda x, y, z: (x - body_origin[0])**2 + (y - body_origin[1])**2 + (z - body_origin[2])**2 - 1        # Object's surface equation (w.r.t. body frame)
finger_base_orientations = [[0.1, 0.2, 0.3],
                            [0.1, 0.2, 0.3],
                            [0.1, 0.2, 0.3]]  # Euler angles of each finger base frame (for Rpk_i) (in radians)
finger_joint_angles      = [[0.1, 0.2, 0.3], 
                            [0.4, 0.5, 0.6],
                            [0.7, 0.8, 0.9]]  # Joint angles of each finger pair
contact_positions        = [[1.0, 0.0, 0.0], 
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0]]  # Contact positions of all fingers on the object (w.r.t to body frame)
link_lengths             = [[0.2, 0.3, 0.2], 
                            [0.3, 0.2, 0.3],
                            [0.2, 0.2, 0.3]]  # Link lengths of all fingers in pairs

# Create the HandJacobianCalculator instance
hand_jacobian_calculator = HandJacobianCalculator(k, N, p, n_values, z_values, euler_angles, 
                                                  body_origin, object_surface_equation, finger_base_orientations, 
                                                  finger_joint_angles, contact_positions, link_lengths)   
hand_jacobian = hand_jacobian_calculator.calculate_hand_jacobian()   # Calculate the hand Jacobian matrix
if hand_jacobian is not None:
    rounded_hand_jacobian = np.round(hand_jacobian, 3)                   # Round off all values in the matrix to 3 decimal places
    matrix_str = '\n'.join(['\t'.join([f'{cell:.3f}' for cell in row]) for row in rounded_hand_jacobian])  # Convert the matrix to a human-readable string with formatting
    # Print the matrix to the console
    print('Hand Jacobian :-')
    print(matrix_str)
