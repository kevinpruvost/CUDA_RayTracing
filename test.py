import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Function to calculate the plane basis
def calculate_plane_basis(normal):
    # Ensure the normal vector is normalized
    normal = normal / np.linalg.norm(normal)
    
    # Choose an arbitrary vector that is not parallel to the normal vector
    arbitrary_vector = np.array([1, 0, 0]) if not np.allclose(normal, [1, 0, 0]) else np.array([0, 1, 0])
    
    # Compute u (orthogonal to normal)
    u = arbitrary_vector - np.dot(arbitrary_vector, normal) * normal
    u = u / np.linalg.norm(u)
    
    # Compute v (orthogonal to both normal and u)
    v = np.cross(normal, u)
    
    return u, v

# Function to project a point onto a plane
def project_point_onto_plane(point, plane_point, u, v):
    w = point - plane_point
    u_coord = np.dot(w, u)
    v_coord = np.dot(w, v)
    return np.array([u_coord, v_coord])

# Define four points
P1 = np.array([0.0, 0.0, 0.0])
P2 = np.array([1.0, 2.0, 1.0])
P3 = np.array([3.0, 3.0, 2.0])

dir = np.array([4.0, 2.0, 2.0])

P4 = P1 + dir
print(P4)
# Calculate the normal vector from the first three points
v1 = P2 - P4
v2 = P3 - P4
normal_vector = np.cross(v1, v2)
normal_vector /= np.linalg.norm(normal_vector)
print(normal_vector)

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the original points
ax.scatter(*P1, color='blue', label='P1')
ax.scatter(*P2, color='red', label='P2')
ax.scatter(*P3, color='green', label='P3')
ax.scatter(*P4, color='orange', label='P4')

# Plot the plane
plane_size = 5
plane_x, plane_y = np.meshgrid(np.linspace(-plane_size, plane_size, 10), np.linspace(-plane_size, plane_size, 10))
plane_z = (-normal_vector[0] * plane_x - normal_vector[1] * plane_y) / normal_vector[2]
ax.plot_surface(plane_x, plane_y, plane_z, alpha=0.5, color='yellow')

# Plot the normal vector
ax.quiver(*P4, *normal_vector, length=2, color='purple', label='Normal Vector')

# Set plot labels and legend
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

# Set aspect ratio for equal scaling
ax.set_box_aspect([1, 1, 1])

# Display plot
plt.show()
