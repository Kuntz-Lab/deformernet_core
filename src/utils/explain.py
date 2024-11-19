import matplotlib.pyplot as plt 
import numpy as np
from torch import Tensor
import open3d
import os
from utils.point_cloud_utils import pcd_ize

def plot_points():
    print("hello")

def visualize_pointclouds_simple_from_tensor(point_cloud: Tensor, point_cloud_features: Tensor, plot_origin: bool = True):
    """
    
    """
    directory_path = "../../outputs"
    os.makedirs(directory_path, exist_ok=True)

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
        
        open3d.visualization.draw_geometries(items) # remains blocked here until visualization window is closed
        open3d.visualization.capture_screen_image(directory_path + f"/feature{feature_of_interest}.png", do_render=True) # save the image to the directory


import numpy as np
import open3d


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