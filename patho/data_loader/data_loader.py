from .dataset import DataSet
import torch
import torch.utils.data


class DataLoader:
    def __init__(self, root_data_dir, images_dir, masks_dir, batch_size=2, num_workers=4):
        dataset = DataSet(root_data_dir, images_dir, masks_dir)
        indices = torch.randperm(len(dataset)).tolist()
        dataset = torch.utils.data.Subset(dataset, indices)

        self.INSTANCE = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    def getInstance(self):
        return self.INSTANCE