import numpy as np
import trimesh
from copy import deepcopy
import open3d
from sklearn.neighbors import NearestNeighbors
from typing import List, Optional
import torch


def visualize_pointclouds(pc1_array: np.array, pc2_array: np.array = None, only_pointclouds: bool = False, manipulation_point: np.array = None, manipulation_point_2: np.array = None, other_points: list = [], action_translation: np.array = None, action_translation_2: np.array = None, action_rotation: np.array = None, action_rotation_2: np.array = None, plot_origin: bool = True):
    """
    Visualize point clouds for DeformerNet

    Parameters:
    - pc1_array (np.array): a point cloud array, usually the goal point cloud
    - pc2_array (np.array): a point cloud array, usually the initial point cloud
    - only_pointclouds (bool): if True, only visualize the point clouds
    - manipulation_point (np.array): a point in 3D space that represents the manipulation point
    - other_points (list[np.array]): a list of other points to visualize
    - action_translation (np.array): a vector representing the action, will be visualized as a line
    - action_rotation (np.array): a vector representing the rotation as XYZ Euler angles, will be visualized as a rotated axis
    - plot_origin (bool): whether to plot the origin

    Opens a open3D visualization window and blocks execution until window is closed
    """
    items = []

    pc1 = pcd_ize(pc1_array, color=[0, 0, 0])
    items.append(pc1)
    
    if pc2_array is not None:
        pc2 = pcd_ize(pc2_array, color = [1, 0, 0])
        items.append(pc2)

    if only_pointclouds:
        open3d.visualization.draw_geometries(items)
        return


    if manipulation_point_2 is not None:
        print("Visualizing bimanual manipulation")
        assert manipulation_point is not None, "manipulation_point must be provided if manipulation_point_2 is provided"
    
    if action_translation_2 is not None:
        assert action_translation is not None, "action_translation must be provided if action_translation_2 is provided"
    
    if action_rotation_2 is not None:
        assert action_rotation is not None, "action_rotation must be provided if action_rotation_2 is provided"


    if manipulation_point is None:
        print("pick a manipulation point for visualization")
        manipulation_point = pick_point(pc1_array)

    if action_translation_2 is not None and manipulation_point_2 is None:
        print("pick a second manipulation point for visualization (object right)")
        manipulation_point_2 = pick_point(pc1_array)

    sphere = open3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    sphere.paint_uniform_color([0, 0, 1])  
    sphere.translate(tuple(manipulation_point))
    items.append(sphere)

    if manipulation_point_2 is not None:
        sphere = open3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        sphere.paint_uniform_color([1, 0, 0])  
        sphere.translate(tuple(manipulation_point_2))
        items.append(sphere)
    
    if action_translation is not None:
        sphere = open3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        sphere.paint_uniform_color([0, 1, 0])  
        sphere.translate(tuple(manipulation_point + action_translation))
        items.append(sphere)
    
        lines = open3d.geometry.LineSet() # Create a LineSet
        lines.points = open3d.utility.Vector3dVector([manipulation_point, manipulation_point + action_translation]) # Add points to the LineSet
        lines.lines = open3d.utility.Vector2iVector([[0, 1]]) # Define the line by specifying the indices of the points in the LineSet
        lines.colors = open3d.utility.Vector3dVector([[1, 0, 0]])  # Red color
        items.append(lines)
    
    if action_rotation is not None:
        # first plot the un-rotated axis at the manipulation point
        coor = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
        coor.translate(manipulation_point)

        # then plot the rotated axis at the end of the action_translation vector
        r = coor.get_rotation_matrix_from_xyz(action_rotation)
        coor_rot = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
        coor_rot.rotate(r, center=(0, 0, 0)) # rotate first, then translate
        coor_rot.translate(manipulation_point + action_translation)
        items.append(coor)
        items.append(coor_rot)
    
    if action_translation_2 is not None:
        sphere = open3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        sphere.paint_uniform_color([0, 1, 0])  
        sphere.translate(tuple(manipulation_point_2 + action_translation_2))
        items.append(sphere)
    
        lines = open3d.geometry.LineSet() # Create a LineSet
        lines.points = open3d.utility.Vector3dVector([manipulation_point_2, manipulation_point_2 + action_translation_2]) # Add points to the LineSet
        lines.lines = open3d.utility.Vector2iVector([[0, 1]]) # Define the line by specifying the indices of the points in the LineSet
        lines.colors = open3d.utility.Vector3dVector([[1, 0, 0]])  # Red color
        items.append(lines)
    
    if action_rotation_2 is not None:
        # first plot the un-rotated axis at the manipulation point
        coor = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
        coor.translate(manipulation_point_2)

        # then plot the rotated axis at the end of the action_translation vector
        r = coor.get_rotation_matrix_from_xyz(action_rotation_2)
        coor_rot = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
        coor_rot.rotate(r, center=(0, 0, 0)) # rotate first, then translate
        coor_rot.translate(manipulation_point_2 + action_translation_2)
        items.append(coor)
        items.append(coor_rot)
            

    print(f"visualizing {len(other_points)} points")
    initial_color = np.array([0, 1, 0])  # Initial color (green)
    darkening_step = 0.1  # Amount to darken the color
    for i, point in enumerate(other_points):
        sphere = open3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        color = np.maximum(initial_color - i * darkening_step, 0)
        sphere.paint_uniform_color(color)  
        sphere.translate(tuple(point))
        items.append(sphere)

    if plot_origin:
        coor = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05) 
        items.append(coor)
    
    open3d.visualization.draw_geometries(items) # remains blocked here until visualization window is closed


def pick_point(pc_array: np.array):
    """
    Given a point cloud, allow the user to pick a point using the Open3D visualizer.

    Parameters:
    - pc_array (np.array): a point cloud array

    Returns:
    - (np.array): the picked point (x, y, z)

    """
    pc = pcd_ize(pc_array, color=[0, 0, 0])
    print("Please pick a point using [shift + left click]")
    vis = open3d.visualization.VisualizerWithEditing()
    vis.create_window()

    vis.add_geometry(pc)
    vis.get_render_option().point_size = 6
    vis.run()
    vis.destroy_window()
    picked_indices = vis.get_picked_points()

    if picked_indices:
        picked_points = pc_array[picked_indices[0]]
        
        return picked_points
    else:
        return None


def tensorize_pointcloud(
    pointcloud: np.ndarray,
    manipulation_points: Optional[List[np.ndarray]] = None,
    ) -> torch.Tensor:
    """
    Convert a numpy pointcloud to a PyTorch tensor. Optionally, add channels for manipulation points.

    Parameters:
        pointcloud (np.ndarray): The pointcloud. Shape (number of points, 3)
        manipulation_points (List[np.ndarray]): The manipulation points. Each manipulation point is a numpy array in the form [x, y, z].

    Returns:
        pointcloud (torch.Tensor): The pointcloud tensor.
    """
    
    assert pointcloud.shape[1] == 3, "Input point cloud should have shape (number of points, 3)"
    if manipulation_points is not None:     
        assert type(manipulation_points) == list, "manipulation_points should be a list of numpy arrays"          
        neigh = NearestNeighbors(n_neighbors=50)
        neigh.fit(pointcloud)
        
        mp_channels = []
        for mani_point in manipulation_points:
            _, nearest_idxs = neigh.kneighbors(mani_point.reshape(1, -1))
            mp_channel = np.zeros(pointcloud.shape[0])
            mp_channel[nearest_idxs.flatten()] = 1
            mp_channels.append(mp_channel)
            
        modified_pointcloud = np.vstack([pointcloud.transpose(1,0)] + mp_channels)
        pointcloud = torch.from_numpy(modified_pointcloud).float()        

    else:
        pointcloud = torch.from_numpy(pointcloud).permute(1,0).float()   
            
    return pointcloud     


def get_manipulation_point_channels(pointcloud: np.ndarray, manipulation_points: np.ndarray) -> List[np.ndarray]:
    """"
    Modifies pointcloud to include a channel for each manipulation point. The channel is 1 near the manipulation point and 0 elsewhere.

    Parameters:
        pointcloud (np.ndarray): The pointcloud.
        manipulation_points (np.ndarray): The manipulation points.

    Returns:
        modified_pc (np.ndarray): The modified pointcloud with the manipulation point channel(s).
    """
    
    assert type(manipulation_points) == list, "manipulation_points should be a list of numpy arrays"          
    neigh = NearestNeighbors(n_neighbors=50)
    neigh.fit(pointcloud)
    
    mp_channels = []
    for mani_point in manipulation_points:
        _, nearest_idxs = neigh.kneighbors(mani_point.reshape(1, -1))
        mp_channel = np.zeros(pointcloud.shape[0])
        mp_channel[nearest_idxs.flatten()] = 1
        mp_channels.append(mp_channel)
        
    modified_pc = np.vstack([pointcloud.transpose(1,0)] + mp_channels)

    return modified_pc


def down_sampling(pc, num_pts=1024, return_indices=False):
    # farthest_indices,_ = farthest_point_sampling(pc, num_pts)
    # pc = pc[farthest_indices.squeeze()]
    # return pc

    """
    Input:
        pc: point cloud data, [B, N, D] where B = num batches, N = num points, D = feature size (typically D=3)
        num_pts: number of samples
    Return:
        centroids: sampled pointcloud index, [num_pts, D]
        pc: down_sampled point cloud, [num_pts, D]
    """

    if pc.ndim == 2:
        # insert batch_size axis
        pc = deepcopy(pc)[None, ...]

    B, N, D = pc.shape
    xyz = pc[:, :, :3]
    centroids = np.zeros((B, num_pts))
    distance = np.ones((B, N)) * 1e10
    farthest = np.random.uniform(low=0, high=N, size=(B,)).astype(np.int32)

    for i in range(num_pts):
        centroids[:, i] = farthest
        centroid = xyz[np.arange(0, B), farthest, :]  # (B, D)
        centroid = np.expand_dims(centroid, axis=1)  # (B, 1, D)
        dist = np.sum((xyz - centroid) ** 2, -1)  # (B, N)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)  # (B,)

    pc = pc[np.arange(0, B).reshape(-1, 1), centroids.astype(np.int32), :]

    if return_indices:
        return pc.squeeze(), centroids.astype(np.int32)

    return pc.squeeze()


def down_sampling_torch(pc, num_pts=1024, return_indices=False):
    """
    Input:
        pc: point cloud data, [B, N, D] where B = num batches, N = num points, D = feature size (typically D=3)
        num_pts: number of samples
    Return:
        centroids: sampled point cloud index, [num_pts, D]
        pc: down-sampled point cloud, [num_pts, D]
    """
    import torch

    if pc.ndim == 2:
        # Insert batch_size axis
        pc = pc.unsqueeze(0)

    B, N, D = pc.shape
    xyz = pc[:, :, :3]
    centroids = torch.zeros((B, num_pts), dtype=torch.long, device=pc.device)
    distance = torch.ones((B, N), dtype=pc.dtype, device=pc.device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=pc.device)

    for i in range(num_pts):
        centroids[:, i] = farthest
        centroid = xyz[torch.arange(0, B), farthest, :]  # (B, D)
        centroid = centroid.unsqueeze(1)  # (B, 1, D)
        dist = torch.sum((xyz - centroid) ** 2, -1)  # (B, N)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.argmax(distance, -1)  # (B,)

    pc = pc[torch.arange(0, B).view(-1, 1), centroids, :]

    if return_indices:
        return pc.squeeze(), centroids

    return pc.squeeze()


def pcd_ize(pc, color=None, vis=False):
    """
    Convert point cloud numpy array to an open3d object (usually for visualization purpose).
    """

    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(pc)
    if color is not None:
        pcd.paint_uniform_color(color)
    if vis:
        open3d.visualization.draw_geometries([pcd])
    return pcd


def visualize_open3d_objects(open3d_objects: list) -> None:
    """
    Visualize a list of Open3D objects.
    """
    if len(open3d_objects) > 0:
        open3d.visualization.draw_geometries(open3d_objects)


def spherify_point_cloud_open3d(point_cloud, radius=0.002, color=None, vis=False):
    """
    Use Open3D to visualize a point cloud where each point is represented by a sphere.
    """
    """
    Visualize a point cloud where each point is represented by a sphere.
    
    Parameters:
    - point_cloud: NumPy array of shape (N, 3), representing the point cloud.
    - radius: float, the radius of each sphere used to represent a point.
    """
    # Create an empty list to hold the sphere meshes
    sphere_meshes = []

    # Iterate over the points in the point cloud
    for point in point_cloud:
        # Create a mesh sphere for the current point
        sphere = open3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sphere.translate(tuple(point))  # Move the sphere to the point's location
        if color is not None:
            sphere.paint_uniform_color(color)
        sphere_meshes.append(sphere)

    # Combine all spheres into one mesh
    combined_mesh = open3d.geometry.TriangleMesh()
    for sphere in sphere_meshes:
        combined_mesh += sphere
    if vis:
        open3d.visualization.draw_geometries([combined_mesh])
    return combined_mesh


def is_homogeneous_matrix(matrix):
    # Check matrix shape
    if matrix.shape != (4, 4):
        return False

    # Check last row
    if not np.allclose(matrix[3, :], [0, 0, 0, 1]):
        return False

    # Check rotational part (3x3 upper-left submatrix)
    rotational_matrix = matrix[:3, :3]
    if not np.allclose(
        np.dot(rotational_matrix, rotational_matrix.T), np.eye(3), atol=1.0e-6
    ) or not np.isclose(np.linalg.det(rotational_matrix), 1.0, atol=1.0e-6):

        print(np.linalg.inv(rotational_matrix), "\n")
        print(rotational_matrix.T)
        print(np.linalg.det(rotational_matrix))

        return False

    return True


def unit_vector(vector):
    """Returns the unit vector of the vector."""
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """Returns the angle in radians between vectors 'v1' and 'v2'::

    >>> angle_between((1, 0, 0), (0, 1, 0))
    1.5707963267948966
    >>> angle_between((1, 0, 0), (1, 0, 0))
    0.0
    >>> angle_between((1, 0, 0), (-1, 0, 0))
    3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def find_min_ang_vec(world_vec, cam_vecs):
    min_ang = float("inf")
    min_ang_idx = -1
    min_ang_vec = None
    for i in range(cam_vecs.shape[1]):
        angle = angle_between(world_vec, cam_vecs[:, i])
        larger_half_pi = False
        if angle > np.pi * 0.5:
            angle = np.pi - angle
            larger_half_pi = True
        if angle < min_ang:
            min_ang = angle
            min_ang_idx = i
            if larger_half_pi:
                min_ang_vec = -cam_vecs[:, i]
            else:
                min_ang_vec = cam_vecs[:, i]

    return min_ang_vec, min_ang_idx


def world_to_object_frame(points):

    """
    Compute 4x4 homogeneous transformation matrix to transform world frame to object frame.
    The object frame is obtained by fitting a bounding box to the object partial-view point cloud.
    The centroid of the bbox is the the origin of the object frame.
    x, y, z axes are the orientation of the bbox.
    We then compare these computed axes against the ground-truth axes ([1,0,0], [0,1,0], [0,0,1]) and align them properly.
    For example, if the computed x-axis is [0.3,0.0,0.95], which is most similar to [0,0,1], this axis would be set to be the new z-axis.

    **This function is used to define a new frame for the object point cloud. Crucially, it creates the training data and defines the pc for test time.

    (Input) points: object partial-view point cloud. Shape (num_pts, 3)
    """

    # Create a trimesh.Trimesh object from the point cloud
    point_cloud = trimesh.points.PointCloud(points)

    # Compute the oriented bounding box (OBB) of the point cloud
    obb = point_cloud.bounding_box_oriented

    homo_mat = obb.primitive.transform
    axes = obb.primitive.transform[:3, :3]  # x, y, z axes concat together

    # Find and align z axis
    z_axis = [0.0, 0.0, 1.0]
    align_z_axis, min_ang_axis_idx = find_min_ang_vec(z_axis, axes)
    axes = np.delete(axes, min_ang_axis_idx, axis=1)

    # Find and align x axis.
    x_axis = [1.0, 0.0, 0.0]
    align_x_axis, min_ang_axis_idx = find_min_ang_vec(x_axis, axes)
    axes = np.delete(axes, min_ang_axis_idx, axis=1)

    # Find and align y axis
    y_axis = [0.0, 1.0, 0.0]
    align_y_axis, min_ang_axis_idx = find_min_ang_vec(y_axis, axes)

    R_o_w = np.column_stack((align_x_axis, align_y_axis, align_z_axis))

    # Transpose to get rotation from world to object frame.
    R_w_o = np.transpose(R_o_w)
    d_w_o_o = np.dot(-R_w_o, homo_mat[:3, 3])

    homo_mat[:3, :3] = R_w_o
    homo_mat[:3, 3] = d_w_o_o

    assert is_homogeneous_matrix(homo_mat)

    return homo_mat


def transform_point_cloud(point_cloud, transformation_matrix):
    # Add homogeneous coordinate (4th component) of 1 to each point
    homogeneous_points = np.hstack((point_cloud, np.ones((point_cloud.shape[0], 1))))

    # Apply the transformation matrix to each point
    transformed_points = np.dot(homogeneous_points, transformation_matrix.T)

    # Remove the homogeneous coordinate (4th component) from the transformed points
    transformed_points = transformed_points[:, :3]

    return transformed_points


def transform_point_clouds(point_clouds, matrices):

    # If there's only one point cloud but multiple matrices, repeat and reshape point cloud to match matrices shape.
    if len(point_clouds.shape) == 2 and len(matrices.shape) == 3:
        num_matrices = matrices.shape[0]
        point_clouds = np.tile(point_clouds, (num_matrices, 1, 1))

    # If there's both only one point cloud and one matrix, add an extra dimension to both.
    elif len(point_clouds.shape) == 2:
        point_clouds = point_clouds[np.newaxis, ...]
        matrices = matrices[np.newaxis, ...]

    # Convert 3D points to homogeneous coordinates
    homogeneous_points = np.concatenate(
        (point_clouds, np.ones_like(point_clouds[..., :1])), axis=-1
    )

    # Perform matrix multiplication for all point clouds at once using broadcasting
    transformed_points = np.matmul(homogeneous_points, matrices.swapaxes(1, 2))

    return transformed_points[:, :, :3]


def random_transformation_matrix(translation_range=None, rotation_range=None):
    from scipy.spatial.transform import Rotation

    # Generate random translation vector
    if translation_range is None:
        translation = np.array([0, 0, 0])
    else:
        translation = np.random.uniform(
            translation_range[0], translation_range[1], size=3
        )

    if rotation_range is None:
        rotation_matrix = np.eye(3)
    else:
        # Generate random rotation angles
        rotation_angles = np.random.uniform(
            rotation_range[0], rotation_range[1], size=3
        )

        # Create rotation object
        rotation = Rotation.from_euler("xyz", rotation_angles, degrees=False)
        rotation_matrix = rotation.as_matrix()

    # Create 4x4 transformation matrix
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = translation

    return transformation_matrix
