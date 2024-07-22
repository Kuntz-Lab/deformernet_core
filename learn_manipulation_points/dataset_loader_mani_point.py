import torch
import os
import random
from torch.utils.data import Dataset
import pickle


class DensePredictorAllObjectsDataset(Dataset):
    """
    Dataset for dense predictor training. Predict manipulation point using segmentation network.
    """

    def __init__(self, dataset_path, object_names, is_bimanual=False):
        self.dataset_path = dataset_path
        self.is_bimanual = is_bimanual

        self.filenames = []
        for object_name in object_names:
            single_object_category_dir = os.path.join(
                self.dataset_path, f"multi_{object_name}/processed_seg_data"
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
        pc = torch.tensor(sample["partial pcs"][0]).float()
        pc_goal = torch.tensor(sample["partial pcs"][1]).float()
        label = torch.tensor(sample["mp_labels"]).long()

        sample = {"pcs": (pc, pc_goal), "label": label}
        return sample
