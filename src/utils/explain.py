import matplotlib.pyplot as plt 
import numpy as np
from torch import Tensor
import open3d
import os
from utils.point_cloud_utils import pcd_ize

import torch
import numpy as np

def compute_action_gradients(model, current_pointcloud_tensor, goal_pointcloud_tensor, 
                           manipulation_point, ideal_goal_manipulation_point):
    """
    Compute gradients of input pointclouds with respect to action prediction error.
    
    Args:
        model: Neural network model
        current_pointcloud_tensor: Current scene pointcloud (should require grad)
        goal_pointcloud_tensor: Goal scene pointcloud (should require grad)
        manipulation_point: Current manipulation point coordinates
        ideal_goal_manipulation_point: Target manipulation point coordinates
    
    Returns:
        tuple: (current_pointcloud_gradients, goal_pointcloud_gradients)
    """

    # Ensure input tensors require gradients
    torch.set_grad_enabled(True)
    current_pointcloud_tensor.requires_grad_(True)
    goal_pointcloud_tensor.requires_grad_(True)
    
    # Forward pass
    pos, rot_mat = model(current_pointcloud_tensor.unsqueeze(0), 
                        goal_pointcloud_tensor.unsqueeze(0))
    
    # Convert ideal action to tensor
    ideal_action = torch.tensor(ideal_goal_manipulation_point - manipulation_point,
                                device=pos.device,
                                dtype=torch.float32)
    
    # Compute L2 distance between predicted and ideal action
    action_error = torch.norm(pos.squeeze() - ideal_action)
    
    # Zero existing gradients
    model.zero_grad()
    
    # Compute gradients
    action_error.backward()
    
    # Get gradients
    current_pointcloud_grad = current_pointcloud_tensor.grad.clone()
    goal_pointcloud_grad = goal_pointcloud_tensor.grad.clone()
    
    # Reset requires_grad
    current_pointcloud_tensor.requires_grad_(False)
    goal_pointcloud_tensor.requires_grad_(False)

    # Visualize gradients

    # turn pointcloud tensors into numpy arrays and remove the manipulation point channel if it exists
    current_pointcloud_numpy = current_pointcloud_tensor.squeeze().detach().cpu().numpy()[0:3, :]
    current_pointcloud_numpy = np.swapaxes(current_pointcloud_numpy, 0, 1)

    goal_pointcloud_numpy = goal_pointcloud_tensor.squeeze().detach().cpu().numpy()[0:3, :]
    goal_pointcloud_numpy = np.swapaxes(goal_pointcloud_numpy, 0, 1)

    current_pointcloud_grad = current_pointcloud_grad.squeeze().detach().cpu().numpy()[0:3, :]
    current_pointcloud_grad = np.swapaxes(current_pointcloud_grad, 0, 1)
    # current_pointcloud_grad = current_pointcloud_grad[:, 0] # take the gradients for only one cartesian direction for now
    current_pointcloud_grad = np.linalg.norm(current_pointcloud_grad, axis=1)
    current_pointcloud_mean = current_pointcloud_grad.mean()
    current_pointcloud_std = current_pointcloud_grad.std()
    cutoff = current_pointcloud_mean + 0.3*current_pointcloud_std
    print(f"mean: {current_pointcloud_mean}, std: {current_pointcloud_std}, cutoff: {cutoff}")
    current_pointcloud_grad[current_pointcloud_grad > cutoff] = cutoff # clip the gradients to remove outliers

    visualize_pointcloud_with_weights(current_pointcloud_numpy, current_pointcloud_grad, plot_origin=True)

    goal_pointcloud_grad = goal_pointcloud_grad.squeeze().detach().cpu().numpy()[0:3, :]
    goal_pointcloud_grad = np.swapaxes(goal_pointcloud_grad, 0, 1)
    # goal_pointcloud_grad = goal_pointcloud_grad[:, 0] # take the gradients for only one cartesian direction for now
    goal_pointcloud_grad = np.linalg.norm(goal_pointcloud_grad, axis=1)
    goal_pointcloud_mean = goal_pointcloud_grad.mean()
    goal_pointcloud_std = goal_pointcloud_grad.std()
    cutoff = goal_pointcloud_mean + 0.3*goal_pointcloud_std
    goal_pointcloud_grad[goal_pointcloud_grad > cutoff] = cutoff # clip the gradients to remove outliers

    visualize_pointcloud_with_weights(goal_pointcloud_numpy, goal_pointcloud_grad, plot_origin=True)
   
    return current_pointcloud_grad, goal_pointcloud_grad



def plot_points():
    print("hello")

def visualize_pointclouds_simple_from_tensor(point_cloud: Tensor, point_cloud_features: Tensor, plot_origin: bool = True):
    """
    
    """
    directory_path = "../../outputs"
    # os.makedirs(directory_path, exist_ok=True)

    items = []
    pc1_array = point_cloud[0].cpu().numpy()
    pc1_array = np.swapaxes(pc1_array, 0, 1)

    point_cloud_features = point_cloud_features[0].cpu().numpy() # remove the batch dimension and convert to numpy
    
    feature_of_interest = 0
    features_to_visualize = 20
    
    for feature_of_interest in range(features_to_visualize):
        feature = point_cloud_features[feature_of_interest]

        feature_max = feature.max()
        feature_min = feature.min()
        feature_mean = feature.mean()
        print(f"feature max: {feature_max}, feature min: {feature_min}, feature mean: {feature_mean}")
        feature = (feature - feature_min) / (feature_max - feature_min) # normalize the feature. Slide to zero and divide by range to get values between 0 and 1

        pc1 = array_to_pointcloud(pc1_array, values=feature, vis=False)
        items.append(pc1)

        if plot_origin:
            coor = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05) 
            items.append(coor)
        

        vis = open3d.visualization.Visualizer()
        vis.create_window(visible=True)
        vis.add_geometry(pc1)
        vis.update_geometry(pc1)
        vis.poll_events()
        vis.update_renderer()
        vis.run()
        vis.capture_screen_image(directory_path + f"/feature{feature_of_interest}.png", do_render=True) # save the image to the directory
        # vis.destroy_window()
        
        # open3d.visualization.draw_geometries(items) # remains blocked here until visualization window is closed
        # open3d.visualization.capture_screen_image(directory_path + f"/feature{feature_of_interest}.png", do_render=True) # save the image to the directory

def visualize_pointcloud_with_weights(point_cloud: np.ndarray, point_cloud_weights: np.ndarray, plot_origin: bool = True):
    
    feature_max = point_cloud_weights.max()
    feature_min = point_cloud_weights.min()
    feature_mean = point_cloud_weights.mean()
    print(f"feature max: {feature_max}, feature min: {feature_min}, feature mean: {feature_mean}")
    point_cloud_weights_normalized = (point_cloud_weights - feature_min) / (feature_max - feature_min) # normalize the feature. Slide to zero and divide by range to get values between 0 and 1

    pc1 = array_to_pointcloud(point_cloud, values=point_cloud_weights_normalized, vis=False)

    if plot_origin:
        coor = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05) 
    

    vis = open3d.visualization.Visualizer()
    vis.create_window(visible=True)
    vis.add_geometry(pc1)
    vis.update_geometry(pc1)
    vis.poll_events()
    vis.update_renderer()
    vis.run()

    directory_path = ""
    vis.capture_screen_image(directory_path + f"/weights_visualized.png", do_render=True) # save the image to the directory
    # vis.destroy_window()


def array_to_pointcloud(pc, color=None, vis=False, values=None):
    """
    Convert a point cloud numpy array to an open3d object (usually for visualization purposes).
    Optionally, colors points based on values ranging from 0 to 1, transitioning from red to green.
    
    Parameters:
    - pc: numpy array of shape (N, 3), representing point cloud coordinates.
    - color: list or tuple of 3 floats, uniform color for the entire point cloud.
    - vis: bool, if True, visualize the point cloud.
    - values: list or numpy array of length N, with values between 0 and 1 to determine point colors.
    
    Returns:
    - pcd: open3d.geometry.PointCloud object.
    """
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(pc)
    
    if values is not None:
        if len(values) != len(pc):
            raise ValueError("Length of values must match the number of points in pc.")
        # Create a color array based on the values
        colors = np.zeros((len(values), 3))
        colors[:, 0] = 1 - values  # Red decreases as values increase
        colors[:, 1] = values      # Green increases as values increase
        colors[:, 2] = 0           # No blue component
        pcd.colors = open3d.utility.Vector3dVector(colors)
    elif color is not None:
        pcd.paint_uniform_color(color)
    
    if vis:
        open3d.visualization.draw_geometries([pcd])
    
    return pcd



def visualize_weights(model):
    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f"Layer: {name} | Shape: {param.shape}")
            print(f"# filters {param.shape[0]}, # depth of filters {param.shape[1]}, filter size {param.shape[2]}x{param.shape[3]}")

            for filter_number in range(param.shape[0]):
                plt.figure()
                plt.title(f"{name} filter {filter_number}")
                plt.imshow(param[filter_number].detach().cpu().numpy().squeeze(), cmap='Purples')
                plt.colorbar()
                plt.show()


if __name__ == "__main__":
    print("Hello, world!")