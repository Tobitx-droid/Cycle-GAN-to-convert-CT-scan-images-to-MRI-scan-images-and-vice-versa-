"""
    Custom Image Dataset class
"""

import os
import glob
import random
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from model.utils import convert_to_rgb


class ImageDataset(Dataset):
    """
        Custom Image Dataset class
    """

    def __init__(
        self,
        root_dir,
        transforms_=None,
        unaligned=False,
        mode="train",
        set_a="A",
        set_b="B",
    ):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        # self.files_a = sorted(
        #     glob.glob(os.path.join(root_dir, "%strainA" % mode) + "/*.*"))
        # self.files_b = sorted(
        #     glob.glob(os.path.join(root_dir, "%strainB" % mode) + "/*.*"))

        self.files_a = sorted(
            glob.glob(os.path.join(root_dir, f"{mode}{set_a}", "*.*"))
        )
        self.files_b = sorted(
            glob.glob(os.path.join(root_dir, f"{mode}{set_b}", "*.*"))
        )

    def __getitem__(self, index):
        image_a = Image.open(self.files_a[index % len(self.files_a)])
        # a % b => a is divided by b, and the remainder of that division is returned.

        if self.unaligned:
            image_b = Image.open(
                self.files_b[random.randint(0, len(self.files_b) - 1)])
        else:
            image_b = Image.open(self.files_b[index % len(self.files_b)])

        # Convert grayscale images to rgb
        if image_a.mode != "RGB":
            image_a = convert_to_rgb(image_a)
        if image_b.mode != "RGB":
            image_b = convert_to_rgb(image_b)

        item_a = self.transform(image_a)
        item_b = self.transform(image_b)

        # Finally return a dict
        return {
            "A": item_a,
            "B": item_b
        }

    def __len__(self):
        return max(len(self.files_a), len(self.files_b))
