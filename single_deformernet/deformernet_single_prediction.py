"""
Authors: Britton Jordan, Bao Thach
Kuntz Lab at the University of Utah

Date: October 2024

"""

import numpy as np
import torch
import transformations
from sklearn.neighbors import NearestNeighbors
from typing import List, Tuple, Dict, Optional

from single_architecture import DeformerNetSingle as DeformerNet
from utils import visualize_pointclouds

def tensorize_pointcloud(
    pointcloud: np.ndarray,
    manipulation_points: Optional[List[np.ndarray]] = None,
    ) -> torch.Tensor:
    """
    Convert a numpy pointcloud to a PyTorch tensor. Optionally, add channels for manipulation points.

    Parameters:
        pointcloud (np.ndarray): The pointcloud.
        manipulation_points (List[np.ndarray]): The manipulation points.

    Returns:
        pointcloud (torch.Tensor): The pointcloud tensor.
    """
    
    assert pointcloud.shape[1] == 3, "Input point cloud should have shape (num_pts, 3)"

    if manipulation_points is not None:     
        mp_channels = get_manipulation_point_channels(pointcloud, manipulation_points) 
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

    weight_path = "./weights/deformernet_w_mp"   

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        print("Using GPU")
    else:
        print("Using CPU")
    
    model.load_state_dict(torch.load(weight_path))  
    model.to(device) # TODO: make this a parameter?
    model.eval()

    with torch.no_grad():

        # Convert the pointclouds to tensors
        goal_pointcloud_tensor = tensorize_pointcloud(goal_pointcloud).to(device)

        if (manipulation_point is not None):                                 
            current_pointcloud_tensor = tensorize_pointcloud(current_pointcloud, mani_points=[manipulation_point]).to(device)
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

    # if visualize:
    #     # Visualize the action
    #     visualize_pointclouds(current_pointcloud, goal_pointcloud, only_pointclouds=False, manipulation_point = manipulation_point, other_points = self.previous_mani_points, action_translation=desired_translation_action, action_rotation=desired_eulers, xyz_limits=xyz_lims)
    
  
    return action   # 6D for single-arm, 12D for bimanual (first 6 for left arm, last 6 for right arm)

if __name__ == '__main__':
    action = run_deformernet_prediction()