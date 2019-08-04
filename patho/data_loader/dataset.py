import numpy as np
import os
from PIL import Image
import torch
import random


class DataSet:
    def __init__(self,
                 root,
                 images_path,
                 masks_path,
                 input_size=572,
                 output_size=388):
        self.root = root
        self.images_path = images_path
        self.masks_path = masks_path
        self.input_size = input_size
        self.output_size = output_size

        self.imgs = list(sorted(os.listdir(os.path.join(root, images_path))))
        self.masks = list(sorted(os.listdir(os.path.join(root, masks_path))))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.images_path, self.imgs[idx])
        mask_path = os.path.join(self.root, self.masks_path, self.masks[idx])

        img = Image.open(img_path).convert("RGB")
        img.thumbnail((self.input_size, self.input_size), Image.ANTIALIAS)

        if self.input_size != self.output_size:
            img.thumbnail((self.output_size, self.output_size),
                          Image.ANTIALIAS)

            white_background = Image.new("RGB",
                                         (self.input_size, self.input_size),
                                         "white")
            white_background.paste(img, (92, 92))

            img_on_white_background = np.array(white_background).reshape(
                self.input_size, self.input_size, 3)

            img = img_on_white_background

        mask = Image.open(mask_path).convert("L")
        maks = mask.resize((self.output_size, self.output_size),
                           Image.ANTIALIAS)

        print(mask.size)
        np_mask = np.array(mask).reshape(self.output_size, self.output_size, 1)
        mask = np_mask.transpose((2, 0, 1)) / 255.

        img = np.array(img).reshape(self.input_size, self.input_size, 3)
        img = img.transpose((2, 0, 1)) / 255.

        img = torch.as_tensor(img, dtype=torch.float32)
        mask = torch.as_tensor(mask, dtype=torch.float32)

        return img, mask

    def __len__(self):
        return len(self.imgs)
