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
import os
import sys

from single_architecture import DeformerNetSingle as DeformerNet

# add the src directory to the path
src_path = os.path.join(os.path.dirname(__file__), '..')
print(f"Adding {src_path} to sys.path")
sys.path.append(src_path)
from utils.point_cloud_utils import tensorize_pointcloud, pick_point, visualize_pointclouds


def run_deformernet_prediction(current_pointcloud: np.ndarray, goal_pointcloud: np.ndarray, manipulation_point: np.ndarray, visualize: bool = False) -> np.ndarray:
    """
    Run a DeformerNet prediction to determine the robot action which will transform the current pointcloud to the goal pointcloud.

    Parameters:
        current_pointcloud (np.ndarray): The current pointcloud.
        goal_pointcloud (np.ndarray): The goal pointcloud.
        manipulation_point (np.ndarray): The manipulation point.
        visualize (bool): Whether to visualize the action. Default is False.

    Returns:
        action (np.ndarray): The predicted robot action in the form of a vector of length 6. The action is in the same frame as the input pointclouds.
    """
      
    # Load network architecture and weights
    model = DeformerNet(normal_channel=False, use_mp_input=(manipulation_point is not None))      

    weight_path = "./data/weights/deformernet_w_mp"   

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        print("Using GPU")
    else:
        print("Using CPU")
    
    model.load_state_dict(torch.load(weight_path))  
    model.to(device)
    model.eval()

    with torch.no_grad():

        # Convert the pointclouds to tensors
        goal_pointcloud_tensor = tensorize_pointcloud(goal_pointcloud).to(device)

        if (manipulation_point is not None):                                 
            current_pointcloud_tensor = tensorize_pointcloud(current_pointcloud, manipulation_points=[manipulation_point]).to(device)
        else:
            current_pointcloud_tensor = tensorize_pointcloud(current_pointcloud).to(device)

        # Run the model
        pos, rot_mat = model(current_pointcloud_tensor.unsqueeze(0), goal_pointcloud_tensor.unsqueeze(0)) # the magic line

        pos, rot_mat = pos.detach().cpu().numpy(), rot_mat.detach().cpu().numpy() # pos.shape = (1, 3)

        # Transform and scale the translation output
        desired_translation_action = pos[0] # extract the 3D translation action
        desired_translation_action *= 0.001 # unit conversion
        desired_translation_action[0] *= -1 # the network outputs the negative of the desired X and Y translation for the arm used in single-arm manipulation. This is a consequence of the way the network was trained.
        desired_translation_action[1] *= -1

        # Convert the rotation matrix to euler angles
        temp1 = np.eye(4)
        temp1[:3,:3] = rot_mat     
        desired_eulers =  transformations.euler_from_matrix(temp1)     
    
    action = np.concatenate((desired_translation_action, desired_eulers), axis=None)

    if visualize:
        # Visualize the action
        visualize_pointclouds(current_pointcloud, goal_pointcloud, only_pointclouds=False, manipulation_point = manipulation_point, action_translation=desired_translation_action, action_rotation=desired_eulers)
    
  
    return action   # 6D for single-arm, 12D for bimanual (first 6 for left arm, last 6 for right arm)

if __name__ == '__main__':
    goal_pointcloud_path = "data/pointclouds/filtered_goal_singlearm_goal1.pickle"
    initial_pointcloud_path = "data/pointclouds/filtered_goal_singlearm_initial1.pickle"
    
    with open(initial_pointcloud_path, 'rb') as handle:
        initial_pointcloud = pickle.load(handle)
        initial_pointcloud = initial_pointcloud.squeeze()

    with open(goal_pointcloud_path, 'rb') as handle:
        goal_pointcloud = pickle.load(handle)
        goal_pointcloud = goal_pointcloud.squeeze()
    
    # manipulation_point = pick_point(initial_pointcloud)
    manipulation_point = np.array([0.0146, 0.0569, 0.0286])

    action = run_deformernet_prediction(initial_pointcloud, goal_pointcloud, manipulation_point, visualize=True)