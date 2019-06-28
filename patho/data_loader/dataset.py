import numpy as np
import os
from PIL import Image
import torch


class DataSet:
    def __init__(self, root, images_path, masks_path, transforms=None):
        self.root = root
        self.images_path = images_path
        self.masks_path = masks_path
        self.transforms = transforms

        self.imgs = list(
            sorted(os.listdir(os.path.join(root, images_path))))
        self.masks = list(
            sorted(os.listdir(os.path.join(root, masks_path))))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.images_path, self.imgs[idx])
        mask_path = os.path.join(self.root, self.masks_path, self.masks[idx])

        img = Image.open(img_path).convert("RGB")
        np_img = np.array(img).reshape(572, 572, 3)
        img = np_img.transpose((2, 0, 1)) / 255.
        img = torch.as_tensor(img, dtype=torch.float32)

        mask = Image.open(mask_path).convert("L")
        np_mask = np.array(mask).reshape(388, 388, 1)
        mask = np_mask.transpose((2, 0, 1)) / 255.
        mask = torch.as_tensor(mask, dtype=torch.float32)

        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        return img, mask

    def __len__(self):
        return len(self.imgs)