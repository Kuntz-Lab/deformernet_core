import torch
import os
import numpy as np
import ast
import random
from torch.utils.data import Dataset
import pickle5 as pickle


class SingleDeformernetDataset(Dataset):
    """
    Dataset for single-arm DeformerNet training.
    """

    def __init__(self, dataset_path=None, use_mp_input=True):
        self.dataset_path = dataset_path
        self.filenames = os.listdir(self.dataset_path)
        self.use_mp_input = use_mp_input

    def load_pickle_data(self, filename):
        with open(os.path.join(self.dataset_path, filename), "rb") as handle:
            return pickle.load(handle)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):

        sample = self.load_pickle_data(self.filenames[idx])

        if self.use_mp_input:
            pc = torch.tensor(
                sample["partial pcs"][0]
            ).float()  # original partial point cloud with MP input
        else:
            pc = torch.tensor(sample["partial pcs"][0][:3, :]).float()  # no MP input
        pc_goal = torch.tensor(sample["partial pcs"][1]).float()

        position = (torch.tensor(sample["pos"]) * 1000).squeeze().float()
        rot_mat = torch.tensor(sample["rot"]).float()
        sample = {"pcs": (pc, pc_goal), "pos": position, "rot": rot_mat}

        return sample


class SingleDeformernetAllObjectsDataset(Dataset):
    """
    Dataset for single-arm DeformerNet training for ALL OBJECTS.
    """

    def __init__(self, dataset_path, object_names, use_mp_input=True):
        self.dataset_path = dataset_path
        self.use_mp_input = use_mp_input

        self.filenames = []
        for object_name in object_names:
            single_object_category_dir = os.path.join(
                self.dataset_path, f"multi_{object_name}/processed_data"
            )
            self.filenames += [
                os.path.join(single_object_category_dir, file)
                for file in os.listdir(single_object_category_dir)
            ]
        random.shuffle(self.filenames)

    def load_pickle_data(self, filename):
        with open(os.path.join(self.dataset_path, filename), "rb") as handle:
            return pickle.load(handle)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):

        sample = self.load_pickle_data(self.filenames[idx])

        if self.use_mp_input:
            pc = torch.tensor(
                sample["partial pcs"][0]
            ).float()  # original partial point cloud with MP input
        else:
            pc = torch.tensor(sample["partial pcs"][0][:3, :]).float()  # no MP input

        pc_goal = torch.tensor(sample["partial pcs"][1]).float()

        position = (torch.tensor(sample["pos"]) * 1000).squeeze().float()
        rot_mat = torch.tensor(sample["rot"]).float()

        sample = {"pcs": (pc, pc_goal), "pos": position, "rot": rot_mat}

        return sample
