# Introduction to Robot Grasping
# Homework- 2
# Question- 1
############## Author : Kush Patel ################
'''
Grasp Matrix Calculator

This script calculates the grasp matrix for a multi-fingered robot hand
holding a circle at specified contact points in 2D/3D.

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
# Importing necessary libraries
import numpy as np
import warnings

class GraspMatrixCalculator:   # Defining the GraspMatrixCalculator class
    
    def __init__(self, k, m, p, euler_angles, finger_positions):  # Initialization method for the GraspMatrixCalculator
        # Validate the input parameters
        if not self._validate_parameters(k, m, p, euler_angles, finger_positions):
            warnings.warn("Invalid input parameters. GraspMatrixCalculator may not work correctly.", UserWarning)
        # Initialize the GraspMatrixCalculator with provided parameters
        self.k = int(k)                           # Number of fingers
        self.m = int(m)                           # Degrees of freedom (3 for 2D, 6 for 3D)
        self.p = int(p)                           # Number of independent forces/torques
        self.euler_angles = euler_angles          # Input Euler angles for orientation
        self.finger_positions = finger_positions  # Finger positions in radians        
        self.J_ = np.array([[0, -1, -1],
                            [1, 0, -1],
                            [1, 1, 0]])
        self.rotation_matrix = self.calculate_rotation_matrix() # Initialize rotation matrix based on Euler angles

    def _validate_parameters(self, k, m, p, euler_angles, finger_positions): # Private method to validate input parameters
        # Validate the number of fingers (k)
        if not isinstance(k, int) or k <= 0:
            warnings.warn("k (Number of fingers) must be a positive integer.", UserWarning)
            return False
        # Validate the degrees of freedom (m)
        if not isinstance(m, int) or m not in [3, 6]:
            warnings.warn("m (Degrees of freedom) must be 3 for 2D or 6 for 3D.", UserWarning)
            return False
        # Validate the number of independent forces/torques (p)
        if not isinstance(p, int) or p <= 0:
            warnings.warn("p (Number of independent forces/torques) must be a positive integer.", UserWarning)
            return False
        # Validate the Euler angles
        if not isinstance(euler_angles, list) or len(euler_angles) != 3:
            warnings.warn("euler_angles must be a list of 3 Euler angles.", UserWarning)
            return False
        if not all(isinstance(angle, (int, float)) for angle in euler_angles):
            warnings.warn("Euler angles must be numeric values.", UserWarning)
            return False
        # Validate the finger positions
        if not isinstance(finger_positions, list) or len(finger_positions) != k:
            warnings.warn(f"finger_positions must be a list of {k} initial finger positions.", UserWarning)
            return False
        if not all(isinstance(angle, (int, float)) for angle in finger_positions):
            warnings.warn("Finger positions must be numeric values.", UserWarning)
            return False
        return True

    def calculate_rotation_matrix(self):  # Method to calculate the rotation matrix based on Euler angles
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

    def calculate_contact_positions(self, finger_position):   # Method to calculate contact positions for a given finger position
        radius = 1.0                                     # Radius of the circular object (assumed to be 1 for simplicity)
        x_contact = radius * np.cos(finger_position)     # Calculate the (x) coordinates of the contact point in body frame
        y_contact = radius * np.sin(finger_position)     # Calculate the (y) coordinates of the contact point in body frame
        z_contact = 0.0                                  # Assuming z_contact is 0 for simplicity (on the object's plane)
        contact_position = ([x_contact, y_contact, z_contact])   # Return the contact position as a simple list
        return contact_position

    def calculate_contact_frames(self, contact_positions):  # Method to calculate contact frames for given contact positions
        contact_frames = []
        for position in contact_positions:            
            n = -np.array(position)/np.linalg.norm(position)  # Calculate the normal vector (n) towards the center of the circle
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

    def calculate_body_vector(self, poses):  # Method to calculate body vectors based on contact positions
        body_vectors = []
        for pos in poses:
            bi = pos                      # Calculate the vector bi from the center of the circle to the contact point
            body_vectors.append(bi)
        B = np.column_stack(body_vectors) # Stack the body vectors to create the matrix B
        return B

    def calculate_grasp_submatrix(self, contact_frames):  # Method to calculate grasp submatrices based on contact frames
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
 
    def calculate_full_grasp_matrix(self):   # Method to calculate the full grasp matrix
        contact_positions = [self.calculate_contact_positions(fp) for fp in self.finger_positions]
        contact_frames = self.calculate_contact_frames(contact_positions)
        body_vec = contact_positions
        self.B = self.calculate_body_vector(body_vec)                      # Calculate B
        grasp_submatrices = self.calculate_grasp_submatrix(contact_frames) # Calculate grasp submatrices for all fingers
        full_grasp_matrix = np.hstack(grasp_submatrices)                   # Combine grasp submatrices to create the full grasp matrix
        return full_grasp_matrix

k = 3                               # Number of fingers
m = 6                               # Dimentionality
p = 3                               # Number of independant forces/torque based on type of contact
euler_angles = [0.0, 0.0, 0.0]      # Replace with actual Euler angles
finger_positions = [0.0, np.pi, np.pi/2] # Replace with actual finger positions w.r.t angle

# Create the GraspMatrixCalculator instance
grasp_calculator = GraspMatrixCalculator(k, m, p, euler_angles, finger_positions)
grasp_matrix = grasp_calculator.calculate_full_grasp_matrix()
rounded_grasp_matrix = np.round(grasp_matrix, 3)    # Round off all values in the matrix to 3 decimal places
matrix_str = '\n'.join(['\t'.join([f'{cell:.3f}' for cell in row]) for row in rounded_grasp_matrix])  # Convert the matrix to a human-readable string with formatting

# Print the matrix to the console
print('Grasp Matrix :-')
print( matrix_str)