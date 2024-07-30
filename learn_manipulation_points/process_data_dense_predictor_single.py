import open3d
import os
import numpy as np
import pickle
import timeit
import sys

sys.path.append("../")
from utils.point_cloud_utils import down_sampling, pcd_ize, to_obj_frame_wrapper
from sklearn.neighbors import NearestNeighbors
import argparse

use_obj_frame = True ### default is False

parser = argparse.ArgumentParser(description=None)
parser.add_argument(
    "--obj_category", default="None", type=str, help="object category. Ex: box_1kPa"
)
args = parser.parse_args()


# data_recording_path = f"/home/baothach/shape_servo_data/manipulation_points/single_physical_dvrk/multi_{args.obj_category}/mp_data"
# data_recording_path = f"/home/baothach/shape_servo_data/rotation_extension/single_physical_dvrk/multi_{args.obj_category}/data"
# data_processed_path = f"/home/baothach/shape_servo_data/manipulation_points/single_physical_dvrk/multi_{args.obj_category}/processed_seg_data"
data_recording_path = f"/home/baothach/Documents/shinghei_mani_data"
data_processed_path = f"/home/baothach/Documents/processed_shinghei_mani_data"
os.makedirs(data_processed_path, exist_ok=True)

start_time = timeit.default_timer()
visualization = False

file_names = sorted(os.listdir(data_recording_path))


for i in range(0, 10000):
    if i % 50 == 0:
        print(
            "current count:", i, " , time passed:", timeit.default_timer() - start_time
        )

    file_name = os.path.join(data_recording_path, "sample " + str(i) + ".pickle")

    if not os.path.isfile(file_name):
        print(f"{file_name} not found")
        continue

    with open(file_name, "rb") as handle:
        data = pickle.load(handle)

    ### Down-sample point clouds
    pc_resampled = down_sampling(data["partial pcs"][0])  # shape (num_points, 3)
    pc_goal_resampled = down_sampling(data["partial pcs"][1])


    if use_obj_frame:
        ############## Using object frame #################
        pc = data["partial pcs"][0]
        pc_goal = data["partial pcs"][1]
        mp_pos = data["mani_point"]
        # convert point clouds to object frame
        pc, pc_goal, world2obj_mat = to_obj_frame_wrapper(p1=pc, p2=pc_goal, mode="obb")
        # downsample
        pc_resampled = down_sampling(pc)#.transpose(1, 0)
        pc_goal_resampled = down_sampling(pc_goal)#.transpose(1, 0)
        # convert mp_pos_1 to object frame
        temp = np.array([mp_pos[0], mp_pos[1], mp_pos[2], 1])
        mp_pos = np.matmul(world2obj_mat, temp.reshape(-1,1))[:3,0]
        ######################################################
    else:
        ############## Not using object frame ################
        pc = down_sampling(data["partial pcs"][0])#.transpose(1, 0)
        pc_goal = down_sampling(data["partial pcs"][1])#.transpose(1, 0)
        mp_pos = data["mani_point"]
        ######################################################

    ### Find 50 points nearest to the manipulation point
    #mp_pos = data["mani_point"]
    neigh = NearestNeighbors(n_neighbors=50)
    neigh.fit(pc_resampled)
    _, nearest_idxs = neigh.kneighbors(mp_pos.reshape(1, -1))
    mp_channel = np.zeros(pc_resampled.shape[0])
    mp_channel[
        nearest_idxs.flatten()
    ] = 1  # shape (num_points,). Get value of 1 at the 50 nearest point, value of 0 elsewhere.

    if visualization:
        pcd_goal = open3d.geometry.PointCloud()
        pcd_goal.points = open3d.utility.Vector3dVector(np.array(pc_goal_resampled))
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(np.array(pc_resampled))
        colors = np.zeros((1024, 3))
        colors[nearest_idxs.flatten()] = [1, 0, 0]
        pcd.colors = open3d.utility.Vector3dVector(colors)
        mani_point = open3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        mani_point.paint_uniform_color([0, 0, 1])
        # open3d.visualization.draw_geometries([pcd, mani_point.translate(tuple(mp_pos))])
        open3d.visualization.draw_geometries(
            [pcd, pcd_goal.translate((0.2, 0, 0)), mani_point.translate(tuple(mp_pos))]
        )

    pcs = (
        np.transpose(pc_resampled, (1, 0)),
        np.transpose(pc_goal_resampled, (1, 0)),
    )  # pcs[0] and pcs[1] shape: (3, num_points)
    processed_data = {
        "partial pcs": pcs,
        "mp_labels": mp_channel,
        "mani_point": data["mani_point"],
        "obj_name": data["obj_name"],
    }
    with open(os.path.join(data_processed_path, file_names[i]), "wb") as handle:
        pickle.dump(processed_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
