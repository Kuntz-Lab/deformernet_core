"""
Authors: Britton Jordan, Bao Thach
Kuntz Lab at the University of Utah

Date: October 2024

"""
import pickle
import numpy as np
import torch
import transformations
from typing import List, Tuple, Dict, Optional
from copy import deepcopy
import os
import sys

from bimanual_architecture import DeformerNetBimanual

# add the src directory to the path
src_path = os.path.join(os.path.dirname(__file__), '..')
print(f"Adding {src_path} to sys.path")
sys.path.append(src_path)
from utils.point_cloud_utils import tensorize_pointcloud, pick_point, visualize_pointclouds


def run_deformernet_prediction(current_pointcloud: np.ndarray, goal_pointcloud: np.ndarray, manipulation_points: List[np.ndarray], visualize: bool = False) -> np.ndarray:
    """
    Run a DeformerNetBimanual prediction to determine the robot action which will transform the current point cloud to the goal point cloud.

    Parameters:
        current_pointcloud (np.ndarray): The current point cloud.
        goal_pointcloud (np.ndarray): The goal point cloud.
        manipulation_points (List[np.ndarray]): The manipulation points. 
                                                Format [left_manipulation_point, right_manipulation_point].
        visualize (bool): Whether to visualize the action. Default is False.

    Returns:
        action (np.ndarray): The predicted robot action in the form of a vector of length 12. The action is in the same frame as the input point clouds.
                             Format: [left_translation, left_rotation, right_translation, right_rotation]. 
                             Left and right are defined by the frame of the point clouds. Negative X direction is left, positive X direction is right.
    """
      
    # Load network architecture and weights
    model = DeformerNetBimanual(normal_channel=False, use_mp_input=(manipulation_points is not None))      

    weight_path = "./data/weights/bimanual_deformernet_w_mp"   

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        print("Using GPU")
    else:
        print("Using CPU")
    
    model.load_state_dict(torch.load(weight_path))  
    model.to(device)
    model.eval()

    with torch.no_grad():

        # Convert the point clouds to tensors
        goal_pointcloud_tensor = tensorize_pointcloud(goal_pointcloud).to(device)

        if (manipulation_points is not None):                                 
            current_pointcloud_tensor = tensorize_pointcloud(current_pointcloud, manipulation_points=manipulation_points).to(device)
        else:
            current_pointcloud_tensor = tensorize_pointcloud(current_pointcloud).to(device)

        # Run the model
        pos, rot_mat_1, rot_mat_2 = model(current_pointcloud_tensor.unsqueeze(0), goal_pointcloud_tensor.unsqueeze(0)) # the magic line
        pos, rot_mat_1, rot_mat_2 = pos.detach().cpu().numpy(), rot_mat_1.detach().cpu().numpy(), rot_mat_2.detach().cpu().numpy()                       

        # Transform and scale the translation output
        desired_translation_action = pos[0] # extract the 3D translation action
        desired_translation_action *= 0.001 # unit conversion

        # negate the X and Y translation actions of just the right arm, transforming the simulation robot #2 to the world frame orientation. Robot #1 is already oriented the same as the world frame.
        desired_translation_action[3] *= -1
        desired_translation_action[4] *= -1

        # Convert the rotation matrix to euler angles
        desired_eulers = np.zeros(6)

        temp1 = np.eye(4)
        temp1[:3,:3] = rot_mat_1 
        desired_eulers[:3] =  transformations.euler_from_matrix(temp1, axes='szyx')            
        
        temp2 = np.eye(4)
        temp2[:3,:3] = rot_mat_2
        desired_eulers[3:6] = transformations.euler_from_matrix(temp2, axes='szyx')

        action = np.concatenate((desired_translation_action, desired_eulers), axis=None)    
        action[3:6], action[6:9] = deepcopy(action[6:9]), deepcopy(action[3:6]) # put each arm's translation and rotation actions next to each other
    

    if visualize:
        # Visualize the action
        visualize_pointclouds(current_pointcloud, goal_pointcloud, only_pointclouds=False, manipulation_point = manipulation_points[0], manipulation_point_2=manipulation_points[1], action_translation=action[:3], action_translation_2=action[6:9], action_rotation=action[3:6], action_rotation_2=action[9:12])
    
  
    return action

if __name__ == '__main__':
    goal_pointcloud_path = "data/pointclouds/filtered_goal_bimanual_goal1.pickle"
    initial_pointcloud_path = "data/pointclouds/filtered_goal_bimanual_initial1.pickle"
    
    with open(initial_pointcloud_path, 'rb') as handle:
        initial_pointcloud = pickle.load(handle)
        initial_pointcloud = initial_pointcloud.squeeze()

    with open(goal_pointcloud_path, 'rb') as handle:
        goal_pointcloud = pickle.load(handle)
        goal_pointcloud = goal_pointcloud.squeeze()
    
    # manipulation_point1 = pick_point(initial_pointcloud)
    # manipulation_point2 = pick_point(initial_pointcloud)
    manipulation_point1 = np.array([0.0123, -0.0964, 0.0392])
    manipulation_point2 = np.array([0.0267, 0.0377, 0.0507])
    manipulation_points = [manipulation_point1, manipulation_point2]

    action = run_deformernet_prediction(initial_pointcloud, goal_pointcloud, manipulation_points, visualize=True)