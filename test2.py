import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_points_with_plane(plane_points, other_points):
    assert len(plane_points) == 3, "Exactly three points are needed to define a plane."
    
    plane_points = np.array(plane_points)
    other_points = np.array(other_points)
    
    # Calculate the normal vector of the plane
    v1 = plane_points[1] - plane_points[0]
    v2 = plane_points[2] - plane_points[0]
    normal_vector = np.cross(v1, v2)
    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    
    # Define a meshgrid for the plane
    d = -plane_points[0].dot(normal_vector)
    xx, yy = np.meshgrid(range(int(np.min(other_points[:,0])-1), int(np.max(other_points[:,0])+2)), 
                         range(int(np.min(other_points[:,1])-1), int(np.max(other_points[:,1])+2)))
    zz = (-normal_vector[0] * xx - normal_vector[1] * yy - d) * 1. / normal_vector[2]
    
    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the plane
    ax.plot_surface(xx, yy, zz, alpha=0.5, color='cyan')
    
    # Plot the plane points
    ax.scatter(plane_points[:, 0], plane_points[:, 1], plane_points[:, 2], color='red', s=100, label='Plane Points')
    
    # Plot the other points
    if other_points.size > 0:
        ax.scatter(other_points[:, 0], other_points[:, 1], other_points[:, 2], color='blue', label='Other Points')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()

# Example usage
plane_points = [
    [25.063544575019147,-11.781345232666467, 0.0000000000000000],
    [25.063544575019147, -11.781345232666467, 30.000000000000000],
    [25.063544575019147, -11.781345232666467, -79.108699911410781]
]

other_points = [
    [45.012708915003820, -10.356269046533294, 1.0000000000000000],
    [40.025417830007655, -10.712538093066588, 10.000000000000000],
    [30.050835660015316, -11.425076186133174, 15.000000000000000]
]

plot_points_with_plane(plane_points, other_points)