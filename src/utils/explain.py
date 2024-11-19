import matplotlib.pyplot as plt 
import numpy as np
from torch import Tensor
import open3d
from point_cloud_utils import pcd_ize

def plot_points():
    print("hello")

def visualize_pointclouds_simple_from_tensor(point_cloud: Tensor, plot_origin: bool = True):
    """
    
    """
    items = []

    pc1_array = point_cloud[0].cpu().numpy()

    pc1 = pcd_ize(pc1_array, color=[0, 0, 0])
    items.append(pc1)

    if plot_origin:
        coor = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05) 
        items.append(coor)
    
    open3d.visualization.draw_geometries(items) # remains blocked here until visualization window is closed

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